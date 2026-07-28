#!/usr/bin/env python3
"""PHASE A -- read-only per-stage / per-tensor gradient-norm audit for P2.

Context: under flat clipping (C=1) 100% of per-sample gradients clip, with
raw norms in the hundreds-to->1000 range. Flat clipping rescales the WHOLE
per-sample gradient by one scalar (C/||g||), so the RELATIVE allocation of
the unit budget across stages is dictated entirely by the raw per-stage norm
composition -- a stage whose raw norm is small relative to a dominant stage
receives almost none of the clipped signal, yet (under DP) receives the same
per-coordinate Gaussian noise as everything else. Whether that imbalance is
severe is an EMPIRICAL question about the real per-stage norm distributions,
which have never been measured separately -- only the total norm has
(scripts/23's grad_norm_stats). This probe measures them, per trainable P2
stage (16, 19, 22, 23) AND per parameter tensor (the granularity Opacus's
DPPerLayerOptimizer actually clips at -- its max_grad_norm is a list with
exactly one threshold per trainable parameter tensor, verified against the
installed opacus 1.6.0 source), so a principled per-layer clipping
configuration can be DERIVED from measurement instead of guessed.

READ-ONLY GUARANTEES (same validated mechanism as scripts/25's probe):
  - reuses train_client_round_dp() completely unmodified (zero changes to
    src/fedxpalm/privacy/dp_sgd.py);
  - wraps two Opacus methods for this process only (restored in a finally
    block): PrivacyEngine.make_private (to capture the name map, snapshot
    every live parameter tensor, and replace original_optimizer.step with a
    no-op so NO parameter update can ever execute) and
    DPOptimizer.clip_and_accumulate (to read per-sample per-tensor norms
    from Opacus's own grad_samples BEFORE clipping runs -- the same public
    property scripts/23's bit-exact-verified instrumentation reads);
  - sigma is forced to 0.0: no Gaussian noise is generated, no epsilon is
    ever reported (dp_sgd.py's existing sigma<=0 guard), and this run makes
    NO privacy claim -- it is a measurement of gradient geometry only;
  - BYTE-IDENTITY VERIFIED, not assumed: every parameter tensor is cloned at
    make_private time and compared (torch.equal) against the live model's
    returned state_dict after the probe -- the result is in the JSON as
    weights_byte_identical (must be true) + any mismatching names (must be
    empty).

Measures, for >= --steps (default 2) LOGICAL optimizer steps on client0 at
logical batch 64 / physical batch 8 (Opacus BatchMemoryManager, the exact
path the real b64phys8 pilots used):
  per PARAMETER TENSOR and per STAGE (16/19/22/23):
    n params, n tensors, per-sample gradient L2 norm mean / median(=p50) /
    p75 / p90 / p95 / max, fraction of samples whose STAGE norm alone
    exceeds C=1 (hypothetical: if that stage had the whole flat budget),
    and the stage's mean share of the total squared norm (the actual
    allocation profile flat clipping preserves);
  GLOBAL: total per-sample norm stats + the fraction clipped under the
    production flat C=1 (this is the real clip fraction, and the only
    number here directly comparable to scripts/23's grad_norm_stats).

Outputs (NEW namespace; nothing under any existing run/results path):
  results/diag_p2_perstage_gradnorm.json

  python scripts/28_diag_p2_perstage_gradnorm.py --device 0

This is a DIAGNOSTIC: no training occurs (zero parameter updates, verified
per-tensor), no noise is added, no epsilon exists, and nothing here changes
any experiment configuration.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402

import fedxpalm  # noqa: E402,F401 (applies the GroupNorm-safe `fuse()` patch)
from fedxpalm.privacy.dp_sgd import train_client_round_dp  # noqa: E402
from fedxpalm.privacy.freeze_audit import stage_of  # noqa: E402

P2_FREEZE_STAGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21]
P2_TRAINABLE_STAGES = [16, 19, 22, 23]

_captured = {
    "per_chunk_norms": [],   # list of (n_samples_in_chunk, n_tensors) cpu tensors, pre-clipping
    "param_names": None,     # ordered clean names, aligned with optimizer params / norm columns
    "param_numels": None,
    "initial_params": None,  # clean name -> cloned tensor, snapshotted at make_private time
    "logical_steps": 0,
    "chunk_batch_sizes": [],
}


def _install_probe():
    """Wrap 2 Opacus methods for this process only; returns restore().
    The ONLY behavior change is neutralizing original_optimizer.step
    (guaranteeing zero parameter updates); clip_and_accumulate is observed
    by reading grad_samples BEFORE calling through to the real
    implementation, which then runs untouched."""
    import opacus.optimizers.optimizer as opt_mod
    import opacus.privacy_engine as pe_mod

    orig_clip = opt_mod.DPOptimizer.clip_and_accumulate
    orig_make_private = pe_mod.PrivacyEngine.make_private

    def wrapped_clip(self):
        # read per-sample per-tensor norms from Opacus's own grad_samples
        # BEFORE the real clipping consumes them. Fires once per PHYSICAL
        # micro-batch under BatchMemoryManager; per-sample statistics do not
        # depend on the logical grouping, so chunks are simply concatenated.
        gs = self.grad_samples  # list aligned with self.params
        if gs and len(gs[0]) > 0:
            per_tensor = torch.stack(
                [g.reshape(len(g), -1).norm(2, dim=-1) for g in gs], dim=1
            ).detach().cpu()  # (batch, n_tensors)
            _captured["per_chunk_norms"].append(per_tensor)
            _captured["chunk_batch_sizes"].append(int(per_tensor.shape[0]))
        # count completed LOGICAL steps: a step is skipped (accumulation
        # only) unless this is the last physical chunk of its logical group
        if not self._check_skip_next_step(pop_next=False):
            _captured["logical_steps"] += 1
        orig_clip(self)  # real Opacus clipping, untouched

    def wrapped_make_private(self, *args, **kwargs):
        result = orig_make_private(self, *args, **kwargs)
        dp_model_ret, dp_optimizer_ret, _ = result
        name_by_id = {id(p): n.replace("_module.", "", 1)
                      for n, p in dp_model_ret.named_parameters()}
        _captured["param_names"] = [name_by_id[id(p)] for p in dp_optimizer_ret.params]
        _captured["param_numels"] = [int(p.numel()) for p in dp_optimizer_ret.params]
        _captured["initial_params"] = {
            n.replace("_module.", "", 1): p.detach().clone().cpu()
            for n, p in dp_model_ret.named_parameters()
        }
        # the ONLY behavior modification: the real SGD update never executes
        dp_optimizer_ret.original_optimizer.step = lambda *a, **kw: None
        return result

    opt_mod.DPOptimizer.clip_and_accumulate = wrapped_clip
    pe_mod.PrivacyEngine.make_private = wrapped_make_private

    def restore():
        opt_mod.DPOptimizer.clip_and_accumulate = orig_clip
        pe_mod.PrivacyEngine.make_private = orig_make_private

    return restore


def _stats(x: torch.Tensor) -> dict:
    return {
        "mean": float(x.mean()),
        "median": float(x.median()),
        "p50": float(torch.quantile(x, 0.50)),
        "p75": float(torch.quantile(x, 0.75)),
        "p90": float(torch.quantile(x, 0.90)),
        "p95": float(torch.quantile(x, 0.95)),
        "max": float(x.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--client", default="0", help="client id from the K=4 partition (spec: client0)")
    parser.add_argument("--steps", type=int, default=2, help="minimum LOGICAL optimizer steps to observe")
    parser.add_argument("--logical-batch", type=int, default=64)
    parser.add_argument("--physical-batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=None, help="default: fl_config model.imgsz (960)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="the FLAT production C the clip-fraction columns are computed against "
                             "(reporting only -- clipping thresholds are not being changed here)")
    parser.add_argument("--base-weights", default="models/base_groupnorm.pt")
    parser.add_argument("--out-dir", default="runs/_diag_perstage_gradnorm_scratch",
                        help="scratch dir for Ultralytics' own run artifacts; NOT a real experiment path")
    parser.add_argument("--tag", default="",
                        help="suffix for the output JSON filename, e.g. --tag canonical so a "
                             "re-run under a fixed dp_sgd.py never silently overwrites a "
                             "measurement taken under a different generation of the code "
                             "(default: empty, byte-identical to the original filename)")
    args = parser.parse_args()

    if not Path(args.base_weights).exists():
        print(f"FAIL: {args.base_weights} not found"); return 1

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)
    with open("configs/dp_config.yaml") as f:
        dp_cfg = yaml.safe_load(f)

    splits_dir = Path(ds_cfg["output_dir"])
    data_yaml = str(splits_dir / "federated_partitions" / "k4_clients" / f"client{args.client}" / "data.yaml")
    if not Path(data_yaml).exists():
        print(f"FAIL: {data_yaml} not found"); return 1

    imgsz = args.imgsz or fl_cfg["model"]["imgsz"]
    hyp = dict(fl_cfg["local_training"], imgsz=imgsz, warmup_epochs=0.0, workers=0,
               batch_size=args.logical_batch)
    dp_hyp = {
        "sigma": 0.0,  # NO noise: gradient-geometry measurement only, no privacy claim
        "max_grad_norm": args.max_grad_norm,
        "delta": dp_cfg["dp_sgd"]["delta"],
        "accountant": dp_cfg["dp_sgd"]["accountant"],
    }

    physical_per_logical = -(-args.logical_batch // args.physical_batch)  # ceil
    # max_steps counts PHYSICAL iterations when physical_batch_size is set
    # (dp_sgd.py's documented semantics) -- generous headroom, truncated below
    max_steps_arg = (args.steps + 1) * physical_per_logical

    print(f"Per-stage gradient-norm audit: client={args.client}  freeze={P2_FREEZE_STAGES}  "
          f"trainable stages={P2_TRAINABLE_STAGES}  logical_batch={args.logical_batch}  "
          f"physical_batch={args.physical_batch}  imgsz={imgsz}  target logical steps>={args.steps}  "
          f"sigma=0 (no noise)  flat-C reference={args.max_grad_norm}")
    print("(optimizer update is a NO-OP for this entire run; byte-identity verified below)")

    restore = _install_probe()
    try:
        final_state, info = train_client_round_dp(
            global_weights_path=args.base_weights, client_data_yaml=data_yaml,
            hyp=hyp, dp_hyp=dp_hyp, round_idx=0, client_id=f"probe_{args.client}",
            out_dir=args.out_dir, device=args.device, freeze_stages=P2_FREEZE_STAGES,
            accountant_state=None, max_steps=max_steps_arg,
            physical_batch_size=(args.physical_batch if args.physical_batch < args.logical_batch else None),
        )
    finally:
        restore()

    # generation-identity check: this probe reuses train_client_round_dp()
    # unmodified, so it silently inherits whichever generation of dp_sgd.py's
    # loss-normalization mechanism is installed -- record + assert it here so
    # the output JSON can never be mistaken for a measurement taken under a
    # different (e.g. pre-canonical-fix) generation of the code.
    expected_loss_reduction = "mean"
    expected_upstream_convention = "ultralytics_sum"
    expected_explicit_normalization = "actual_microbatch_mean"
    for key, expected in (
        ("loss_reduction", expected_loss_reduction),
        ("upstream_loss_convention", expected_upstream_convention),
        ("explicit_loss_normalization", expected_explicit_normalization),
    ):
        got = info.get(key)
        if got != expected:
            print(f"FAIL: dp_sgd.py reported {key}={got!r}, expected {expected!r} -- "
                  f"this probe's per-tensor norms would not be comparable to the "
                  f"canonical-generation record; refusing to write output")
            return 1

    if not _captured["per_chunk_norms"]:
        print("FAIL: no per-sample gradients were observed (empty dataloader or all batches skipped)")
        return 1
    if _captured["logical_steps"] < args.steps:
        print(f"[!] only {_captured['logical_steps']} logical steps completed "
              f"(requested >= {args.steps}) -- stats below still use every observed sample")

    # ---- byte-identity verification (REAL check, not an assumption) ----
    mismatched = []
    for name, init_t in _captured["initial_params"].items():
        if name not in final_state:
            mismatched.append(f"{name} (missing from final state_dict)")
        elif not torch.equal(init_t, final_state[name].detach().cpu()):
            mismatched.append(name)
    weights_byte_identical = len(mismatched) == 0
    print(f"\nweights_byte_identical={weights_byte_identical}"
          + (f"  MISMATCHES: {mismatched[:8]}" if mismatched else "  (all parameter tensors torch.equal)"))

    all_norms = torch.cat(_captured["per_chunk_norms"], dim=0)  # (n_samples, n_tensors)
    names = _captured["param_names"]
    numels = _captured["param_numels"]
    n_samples = all_norms.shape[0]
    C = args.max_grad_norm

    # global (production-comparable): total per-sample norm across all tensors
    total_norms = all_norms.norm(2, dim=1)
    global_stats = _stats(total_norms)
    global_clip_frac = float((total_norms > C).float().mean())

    # per-tensor records (the granularity Opacus per-layer clipping uses)
    per_tensor = []
    for j, (name, numel) in enumerate(zip(names, numels)):
        col = all_norms[:, j]
        per_tensor.append({
            "name": name, "stage": stage_of(name), "n_params": numel,
            "per_sample_norm": _stats(col),
            "fraction_above_flat_C": float((col > C).float().mean()),
            "mean_share_of_total_sq_norm": float((col ** 2).mean() / (total_norms ** 2).mean()),
        })

    # per-stage aggregation: per-sample stage norm = sqrt(sum of member-tensor norm^2)
    per_stage = {}
    for s in P2_TRAINABLE_STAGES:
        idx = [j for j, name in enumerate(names) if stage_of(name) == s]
        if not idx:
            per_stage[str(s)] = {"n_tensors": 0, "n_params": 0,
                                 "note": "no trainable tensors observed for this stage"}
            continue
        stage_norms = all_norms[:, idx].norm(2, dim=1)
        per_stage[str(s)] = {
            "n_tensors": len(idx),
            "n_params": int(sum(numels[j] for j in idx)),
            "per_sample_norm": _stats(stage_norms),
            "fraction_stage_norm_above_flat_C": float((stage_norms > C).float().mean()),
            "fraction_note": "hypothetical: fraction of samples whose THIS-STAGE norm alone exceeds "
                             "the flat C -- under actual flat clipping the clip decision uses the "
                             "TOTAL norm (see global_clip_fraction), not per-stage norms",
            "mean_share_of_total_sq_norm": float((stage_norms ** 2).mean() / (total_norms ** 2).mean()),
        }

    unexpected = sorted({stage_of(n) for n in names} - set(P2_TRAINABLE_STAGES) - {None})
    if unexpected:
        print(f"[!] UNEXPECTED trainable stages in optimizer: {unexpected} -- investigate before trusting this audit")

    print(f"\nsamples observed: {n_samples}  (logical steps completed: {_captured['logical_steps']}, "
          f"physical chunks: {len(_captured['chunk_batch_sizes'])})")
    print(f"GLOBAL total per-sample norm: mean={global_stats['mean']:.1f} median={global_stats['median']:.1f} "
          f"p90={global_stats['p90']:.1f} max={global_stats['max']:.1f}  "
          f"clip_fraction@flatC={C}: {global_clip_frac:.3f}")
    print(f"\n{'stage':>6}{'tensors':>9}{'params':>10}{'mean':>10}{'median':>10}{'p75':>10}"
          f"{'p90':>10}{'p95':>10}{'max':>11}{'frac>C':>8}{'sq-share':>10}")
    for s in P2_TRAINABLE_STAGES:
        r = per_stage[str(s)]
        if r.get("n_tensors", 0) == 0:
            print(f"{s:>6}  (none)")
            continue
        st = r["per_sample_norm"]
        print(f"{s:>6}{r['n_tensors']:>9}{r['n_params']:>10}{st['mean']:>10.2f}{st['median']:>10.2f}"
              f"{st['p75']:>10.2f}{st['p90']:>10.2f}{st['p95']:>10.2f}{st['max']:>11.2f}"
              f"{r['fraction_stage_norm_above_flat_C']:>8.3f}{r['mean_share_of_total_sq_norm']:>10.4f}")

    record = {
        "diagnostic": "p2_perstage_gradnorm_audit",
        "note": "READ-ONLY DIAGNOSTIC -- zero parameter updates (byte-identity verified per tensor), "
                "sigma=0 (no noise, no epsilon, no privacy claim). Norms are PRE-clipping per-sample "
                "L2 norms read from Opacus's own grad_samples.",
        "client_id": args.client, "freeze_stages": P2_FREEZE_STAGES,
        "trainable_stages": P2_TRAINABLE_STAGES,
        "logical_batch": args.logical_batch, "physical_batch": args.physical_batch,
        "imgsz": imgsz, "flat_C_reference": C, "sigma": 0.0,
        "n_samples_observed": int(n_samples),
        "logical_steps_completed": int(_captured["logical_steps"]),
        "physical_chunk_batch_sizes": _captured["chunk_batch_sizes"],
        "weights_byte_identical": weights_byte_identical,
        "weight_mismatches": mismatched,
        "loss_normalization_generation": "canonical_loss_normalization",
        "dp_loss_reduction": info.get("loss_reduction"),
        "upstream_loss_convention": info.get("upstream_loss_convention"),
        "explicit_loss_normalization": info.get("explicit_loss_normalization"),
        "global_total_norm": {**global_stats, "clip_fraction_at_flat_C": global_clip_frac},
        "per_stage": per_stage,
        "per_tensor": per_tensor,
        "unexpected_trainable_stages": unexpected,
    }
    suffix = f"_{args.tag}" if args.tag else ""
    out_json = f"results/diag_p2_perstage_gradnorm{suffix}.json"
    Path("results").mkdir(exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nSaved {out_json}")
    print("Next step (Phase B) derives per-tensor clipping thresholds FROM this file's measured "
          "distributions -- do not pick thresholds by hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
