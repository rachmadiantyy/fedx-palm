#!/usr/bin/env python3
"""Noise-vs-signal probe: measures, for 1-2 REAL DP optimizer steps at the
exact production config (C=1, sigma=0.5, logical batch=8, freeze=[0..10],
lr0=0.01), whether the Gaussian noise numerically dominates the clipped
gradient signal -- BEFORE any parameter update ever happens.

Context: Diagnostic B (sigma=0, clipping only) climbed mAP50 0.3395 -> 0.7963
over 5 rounds -- clipping alone is aggressive (100% of per-sample gradients
clipped, median pre-clip norm ~462-617 vs C=1.0) but still usable. The real
E2 sigma=0.5 run collapsed (0.0247 -> 0.0544 over the same rounds). This
probe measures the actual noise/signal magnitudes to confirm (or refute)
that Gaussian noise is what destroys the signal, before touching C or lr0.

HOW: reuses train_client_round_dp() -- the exact same production function
E1/E2 and Diagnostic B/C call -- completely UNMODIFIED (zero changes to
src/fedxpalm/privacy/dp_sgd.py). Three of Opacus's OWN methods are wrapped
for the duration of this script's process only (restored in a finally
block), never touching this repo's code:

  1. PrivacyEngine.make_private -- after calling through to the real
     implementation, captures the (name -> id) map for the per-region
     breakdown, and replaces the returned optimizer's
     `original_optimizer.step` with a no-op. This is the ONLY behavior
     modification, and it is deliberate: it guarantees the real SGD update
     (p.data -= lr * p.grad) NEVER executes, for every step observed, so
     "measure before optimizer update" holds literally, not just in
     spirit -- verified empirically (see below) AND re-verified at the end
     of every run of this script by diffing the untouched checkpoint
     against the model's live state after the probe.
  2. DPOptimizer.clip_and_accumulate -- calls through to Opacus's real
     implementation (untouched), then reads the resulting p.summed_grad
     (the real clipped-and-summed gradient, exactly what production sees)
     and Opacus's own `expected_batch_size` / `len(grad_samples[0])`
     (actual sampled batch size for this specific Poisson draw).
  3. DPOptimizer.add_noise -- calls through to Opacus's real implementation
     (untouched; generates real Gaussian noise with std=sigma*C), then reads
     p.grad = summed_grad + noise and derives noise = p.grad - p.summed_grad.

Verified end-to-end (real PrivacyEngine.make_private()-wrapped model, both
1-step exception-based and 2-step no-op-based variants): the wrapped
clip_and_accumulate/add_noise readings are BIT-EXACT against Opacus's
internal computation (confirmed via a separate spy test), and the model's
parameters are BYTE-IDENTICAL before and after a 2-step probe.

Reports, per step: expected_batch_size, actual_sampled_batch_size, L2 norm
of the aggregated clipped gradient BEFORE noise, L2 norm of the generated
noise, L2 norm AFTER noise, noise_to_signal_norm_ratio, and a per-region
breakdown (neck / head -- NOT "late backbone", see note below).

NOTE on regions: freeze_stages=[0..10] means the ENTIRE backbone (stages
0-10) is frozen and was already stripped out of the optimizer before
PrivacyEngine.make_private() (verified in the freeze/optimizer audit) --
there is no "late backbone" region inside the optimizer to report on; its
gradient/noise contribution is exactly zero by construction, not a
measurement. The two trainable regions under the CURRENT E2 config are
neck (stages 11-22) and head (stage 23, DFL excluded since it stays frozen
even in the trainable regions -- see freeze_audit.py's stage map).

TASK 1 FINDING (baseline logical batch=8): the real probe reported
expected_batch_size=7, not 8. Traced to Opacus's OWN source, not a repo
bug: `DPDataLoader.from_data_loader` sets `sample_rate = 1/len(data_loader)`,
and a standard (non-drop-last) DataLoader has `len() = ceil(n/batch)`. For
client0 (n=860, batch=8): len=ceil(860/8)=108, sample_rate=1/108=0.009259.
`PrivacyEngine.make_private` then sets
`expected_batch_size = int(len(dataset) * sample_rate)` -- note the `int()`
TRUNCATION (not round) -- `int(860 * 1/108) = int(7.963) = 7`. Verified this
reproduces exactly for all 4 clients (n=860/775/5305/1997) at batch=8, all
of which round down to expected_batch_size=7 under this formula. This is
expected, intentional Opacus behavior (a structural consequence of deriving
a Poisson sample_rate from a discretized "epoch length", not specific to
this dataset), NOT an implementation or accounting error -- left unchanged.
It also does not explain the noise domination: the ~12% gap between 7 and 8
is negligible next to the observed ~100-190x noise_to_signal ratio.

LOGICAL-BATCH SWEEP (this task): --logical-batch/--physical-batch let the
Poisson sample_rate/expected_batch_size be computed for a LARGER logical
batch (32, 64, ...) while the physical per-step compute stays capped (<=8,
for GPU memory), via Opacus's own `BatchMemoryManager` -- wired into
train_client_round_dp() as an additive, opt-in `physical_batch_size` kwarg
(default None = unwrapped, unused by any real E1/E2 call site, so their
behavior is byte-for-byte unchanged). BatchMemoryManager splits each Poisson
logical draw into <=physical_batch_size chunks; Opacus's own skip-step
queue makes clip_and_accumulate() fire (and accumulate into p.summed_grad)
on EVERY physical chunk, while add_noise() (and a real step) fires only
ONCE per logical group -- verified end-to-end on a real model: the
accumulated actual_sampled_batch_size correctly reflects the FULL logical
batch (not just the last physical chunk -- an easy mistake, caught and
fixed during verification: reading `grad_samples` after the ALREADY-run
clip call, or without tracking accumulation across skipped calls, silently
under-reports it), and model parameters remain BYTE-IDENTICAL across
multiple physical iterations spanning multiple logical steps.

Reports "projected" privacy-accounting inputs (sample rate q, expected
total local steps) for each logical batch size via this repo's own
accounting helper (fedxpalm.privacy.accounting) -- NOT a final epsilon,
which is only ever computed by the same PRV accountant during a real
(or previously-run) training call, per instruction.

Outputs (NEW namespace, no B1/B2/E1/E2/diag_a/diag_b path touched, no
checkpoint saved anywhere -- every parameter update is a no-op):
  results/diag_noise_signal_probe_client{id}_sigma{sigma}_logB{L}_physB{P}.json

  python scripts/25_diag_noise_signal_probe.py --device 0                                    # baseline, logical=physical=8
  python scripts/25_diag_noise_signal_probe.py --device 0 --logical-batch 32 --physical-batch 8
  python scripts/25_diag_noise_signal_probe.py --device 0 --logical-batch 64 --physical-batch 8

This is a DIAGNOSTIC: no training occurs (zero parameter updates, verified),
no test split is read, and nothing here changes C, sigma, or lr0 for any
real experiment.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402

import fedxpalm  # noqa: E402,F401 (applies the GroupNorm-safe `fuse()` patch)
from fedxpalm.privacy.accounting import estimate_steps, training_sample_rate  # noqa: E402
from fedxpalm.privacy.dp_sgd import train_client_round_dp  # noqa: E402
from fedxpalm.privacy.freeze_audit import BACKBONE_MAX_STAGE, stage_of  # noqa: E402

_captured = {"steps": [], "name_by_id": {}, "_accum_bs": 0}


def _region_of(name: str) -> str:
    if "dfl.conv" in name:
        return "dfl"
    stage = stage_of(name)
    if stage is None:
        return "other"
    if stage <= BACKBONE_MAX_STAGE:
        return "backbone"  # expected to be EMPTY -- frozen, stripped from optimizer
    return "neck" if stage <= 22 else "head"


def _norm(tensors) -> float:
    if not tensors:
        return 0.0
    return float(torch.sqrt(sum((t.float() ** 2).sum() for t in tensors)))


def _install_probe():
    """Wrap 3 Opacus methods for the duration of this process. Returns a
    restore() callable. See module docstring for exactly what each wrap does
    and why the optimizer no-op is the only behavior change (guaranteeing no
    parameter update, not merely "probably none")."""
    import opacus.optimizers.optimizer as opt_mod
    import opacus.privacy_engine as pe_mod

    orig_clip = opt_mod.DPOptimizer.clip_and_accumulate
    orig_add_noise = opt_mod.DPOptimizer.add_noise
    orig_make_private = pe_mod.PrivacyEngine.make_private

    def wrapped_clip(self):
        # With BatchMemoryManager (physical_batch_size set), clip_and_accumulate()
        # fires on EVERY physical micro-batch, accumulating into p.summed_grad,
        # while add_noise() only fires once per LOGICAL group. `self.grad_samples`
        # at this point reflects only the CURRENT physical chunk (p.grad_sample was
        # reset to None after the previous clip call), so the actual accumulated
        # logical batch size must be tracked across calls, not read once here --
        # verified: reading it naively under-reports by ~4x at logical=32/physical=8.
        # p.summed_grad is None exactly when a NEW logical group starts (Opacus's
        # own zero_grad() only resets it to None when the PREVIOUS step wasn't skipped).
        starting_new_group = self.params[0].summed_grad is None
        this_call_bs = len(self.grad_samples[0]) if self.grad_samples and len(self.grad_samples) > 0 else 0
        if starting_new_group:
            _captured["_accum_bs"] = 0
        _captured["_accum_bs"] += this_call_bs
        orig_clip(self)  # real Opacus clipping -- p.summed_grad now holds the real clipped/summed gradient
        _captured["_pending"] = {
            "expected_batch_size": self.expected_batch_size,
            "actual_sampled_batch_size": _captured["_accum_bs"],  # running total across physical chunks
            "pre_noise": {id(p): p.summed_grad.detach().clone() for p in self.params},
        }

    def wrapped_add_noise(self):
        orig_add_noise(self)  # real Opacus noise generation -- p.grad = summed_grad + noise
        pend = _captured["_pending"]
        post = {id(p): p.grad.detach().clone() for p in self.params}
        noise = {i: post[i] - pend["pre_noise"][i] for i in pend["pre_noise"]}
        _captured["steps"].append({**pend, "post_noise": post, "noise": noise})

    def wrapped_make_private(self, *args, **kwargs):
        result = orig_make_private(self, *args, **kwargs)
        dp_model_ret, dp_optimizer_ret, _ = result
        _captured["name_by_id"] = {id(p): n.replace("_module.", "", 1) for n, p in dp_model_ret.named_parameters()}
        # snapshot every live parameter tensor so byte-identity can be
        # VERIFIED at the end (torch.equal against the returned state_dict),
        # not merely argued from the no-op mechanism
        _captured["initial_params"] = {n.replace("_module.", "", 1): p.detach().clone().cpu()
                                       for n, p in dp_model_ret.named_parameters()}
        # the ONLY behavior modification: neutralize the real parameter
        # update so every observed step is provably a no-op on the model
        dp_optimizer_ret.original_optimizer.step = lambda *a, **kw: None
        return result

    opt_mod.DPOptimizer.clip_and_accumulate = wrapped_clip
    opt_mod.DPOptimizer.add_noise = wrapped_add_noise
    pe_mod.PrivacyEngine.make_private = wrapped_make_private

    def restore():
        opt_mod.DPOptimizer.clip_and_accumulate = orig_clip
        opt_mod.DPOptimizer.add_noise = orig_add_noise
        pe_mod.PrivacyEngine.make_private = orig_make_private

    return restore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--client", default=None, help="client id from the K=4 partition (default: first one found)")
    parser.add_argument("--freeze-stages", type=int, nargs="*", default=None,
                        help="override configs/dp_config.yaml variants.partial.freeze_stages (default: read "
                             "from there, i.e. P0/current-E2 [0..10]). Pass explicit stage indices for P1 "
                             "(head-only: 0 1 2 ... 22) or P2 (graph-justified: all except 16,19,22,23)")
    parser.add_argument("--subset-label", default="P0",
                        help="label for this trainable-subset candidate (P0/P1/P2/...) -- purely for output "
                             "filename/record clarity, does not affect the probe itself")
    parser.add_argument("--steps", type=int, default=2, help="DP optimizer steps to probe (1-2 per spec)")
    parser.add_argument("--sigma", type=float, default=0.5, help="matches the real E2 collapsed run")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="default: configs/dp_config.yaml (C=1.0)")
    parser.add_argument("--lr0", type=float, default=None, help="default: fl_config local_training.lr0 (0.01)")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None, help="legacy alias for --logical-batch")
    parser.add_argument("--logical-batch", type=int, default=None,
                        help="logical (Poisson) batch size driving sample_rate/expected_batch_size "
                             "(default: fl_config local_training.batch_size = 8, i.e. the validated baseline)")
    parser.add_argument("--physical-batch", type=int, default=None,
                        help="physical per-step batch actually placed on device, capped for GPU memory "
                             "(default: same as the resolved logical batch, i.e. no BatchMemoryManager -- "
                             "byte-identical to the original baseline probe path)")
    parser.add_argument("--base-weights", default="models/base_groupnorm.pt")
    parser.add_argument("--out-dir", default="runs/_diag_noise_signal_probe_scratch",
                        help="scratch dir for Ultralytics' own run artifacts; NOT a real experiment path")
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
    manifest_path = splits_dir / "federated_partitions" / "manifest.json"
    if not manifest_path.exists():
        print(f"FAIL: {manifest_path} not found -- run scripts/03_partition_clients.py --k 4 first"); return 1
    with open(manifest_path) as f:
        manifest = json.load(f)["4"]
    client_id = args.client or sorted(manifest["sizes"], key=lambda c: int(c) if str(c).isdigit() else c)[0]
    data_yaml = str(splits_dir / "federated_partitions" / "k4_clients" / f"client{client_id}" / "data.yaml")
    if not Path(data_yaml).exists():
        print(f"FAIL: {data_yaml} not found"); return 1

    freeze_stages = (args.freeze_stages if args.freeze_stages is not None
                    else dp_cfg["variants"]["partial"]["freeze_stages"])  # default: same source as A0/E2/Diag B
    imgsz = args.imgsz or fl_cfg["model"]["imgsz"]

    default_batch = fl_cfg["local_training"]["batch_size"]
    logical_batch = args.logical_batch or args.batch or default_batch
    physical_batch = args.physical_batch or logical_batch  # default: no BatchMemoryManager needed
    if physical_batch > logical_batch:
        print(f"FAIL: --physical-batch ({physical_batch}) must not exceed --logical-batch ({logical_batch})")
        return 1
    # only engage BatchMemoryManager when actually needed -- when physical ==
    # logical, physical_batch_size stays None, byte-identical to the
    # originally-validated (unwrapped) baseline path
    physical_batch_size_arg = physical_batch if physical_batch < logical_batch else None

    hyp = dict(fl_cfg["local_training"], imgsz=imgsz, warmup_epochs=0.0, workers=0, batch_size=logical_batch)
    if args.lr0 is not None:
        hyp["lr0"] = args.lr0
    max_grad_norm = args.max_grad_norm if args.max_grad_norm is not None else dp_cfg["dp_sgd"]["max_grad_norm"]
    dp_hyp = {
        "sigma": args.sigma, "max_grad_norm": max_grad_norm,
        "delta": dp_cfg["dp_sgd"]["delta"], "accountant": dp_cfg["dp_sgd"]["accountant"],
    }

    n_client = manifest["sizes"][client_id]
    projected_q = training_sample_rate(n_client, logical_batch)
    print(f"Noise/signal probe: client={client_id} (n={n_client})  freeze={freeze_stages}  "
          f"lr0={hyp['lr0']}  C={max_grad_norm}  sigma={args.sigma}  "
          f"logical_batch={logical_batch}  physical_batch={physical_batch}  "
          f"imgsz={imgsz}  steps={args.steps}")
    print(f"projected sample_rate q = {projected_q:.6f} (NOT a final epsilon -- see note below)")
    print("(optimizer update is a NO-OP for this entire run -- verified below)")

    # If physical_batch_size_arg is set, max_steps counts PHYSICAL iterations
    # (see dp_sgd.py's note), so request generous headroom to guarantee
    # args.steps full LOGICAL groups complete, then truncate to args.steps below.
    physical_per_logical = -(-logical_batch // physical_batch)  # ceil division
    max_steps_arg = args.steps * physical_per_logical + physical_per_logical if physical_batch_size_arg else args.steps

    restore = _install_probe()
    try:
        final_state, _probe_info = train_client_round_dp(
            global_weights_path=args.base_weights, client_data_yaml=data_yaml,
            hyp=hyp, dp_hyp=dp_hyp, round_idx=0, client_id=f"probe_{client_id}",
            out_dir=args.out_dir, device=args.device, freeze_stages=freeze_stages,
            accountant_state=None, max_steps=max_steps_arg,
            physical_batch_size=physical_batch_size_arg,
        )
    finally:
        restore()

    # keep only the requested number of logical steps (extra ones may have
    # been captured due to the generous max_steps headroom above)
    _captured["steps"] = _captured["steps"][:args.steps]

    if not _captured["steps"]:
        print("FAIL: no DP optimizer steps were observed (empty dataloader or all batches skipped)")
        return 1

    name_by_id = _captured["name_by_id"]
    step_records = []
    print(f"\n{'step':>5}{'expected_bs':>12}{'actual_bs':>10}{'signal':>10}{'noise':>10}{'post':>10}{'ratio':>9}")
    for i, s in enumerate(_captured["steps"]):
        signal = _norm(list(s["pre_noise"].values()))
        noise = _norm(list(s["noise"].values()))
        post = _norm(list(s["post_noise"].values()))
        ratio = noise / signal if signal > 0 else None
        print(f"{i:>5}{s['expected_batch_size']:>12.2f}{s['actual_sampled_batch_size']:>10}"
              f"{signal:>10.3f}{noise:>10.3f}{post:>10.3f}{(ratio if ratio is not None else float('nan')):>9.3f}")

        # per-region breakdown -- backbone is expected EMPTY (frozen, stripped
        # from optimizer before make_private()); reported anyway for direct proof
        by_region = {"backbone": [], "neck": [], "head": [], "dfl": [], "other": []}
        for pid in s["pre_noise"]:
            by_region[_region_of(name_by_id.get(pid, ""))].append(pid)
        region_stats = {}
        for region, ids in by_region.items():
            sig_r = _norm([s["pre_noise"][i] for i in ids])
            noi_r = _norm([s["noise"][i] for i in ids])
            post_r = _norm([s["post_noise"][i] for i in ids])
            region_stats[region] = {
                "n_params": len(ids), "signal_norm": sig_r, "noise_norm": noi_r,
                "post_noise_norm": post_r, "ratio": (noi_r / sig_r if sig_r > 0 else None),
            }
        print(f"       per-region: " + ", ".join(
            f"{r}(n={region_stats[r]['n_params']}, sig={region_stats[r]['signal_norm']:.3f}, "
            f"noise={region_stats[r]['noise_norm']:.3f})"
            for r in ("backbone", "neck", "head", "dfl") if region_stats[r]["n_params"] > 0))

        # per-STAGE breakdown (finer than the neck/head regions above --
        # e.g. P2's trainable stages 16/19/22 are all "neck" but behave
        # differently, per the Phase A per-stage gradient-norm audit)
        by_stage: dict = {}
        for pid in s["pre_noise"]:
            st = stage_of(name_by_id.get(pid, ""))
            by_stage.setdefault(st, []).append(pid)
        stage_stats = {}
        for st, ids in sorted(by_stage.items(), key=lambda kv: (kv[0] is None, kv[0])):
            sig_s = _norm([s["pre_noise"][i] for i in ids])
            noi_s = _norm([s["noise"][i] for i in ids])
            stage_stats[str(st)] = {
                "n_param_tensors": len(ids), "signal_norm": sig_s, "noise_norm": noi_s,
                "ratio": (noi_s / sig_s if sig_s > 0 else None),
            }
        print(f"       per-stage:  " + ", ".join(
            f"s{st}(sig={v['signal_norm']:.3f}, noise={v['noise_norm']:.3f})"
            for st, v in stage_stats.items()))

        step_records.append({
            "step": i, "expected_batch_size": s["expected_batch_size"],
            "actual_sampled_batch_size": s["actual_sampled_batch_size"],
            "signal_norm_before_noise": signal, "noise_norm": noise, "norm_after_noise": post,
            "noise_to_signal_norm_ratio": ratio, "per_region": region_stats,
            "per_stage": stage_stats,
        })

    # byte-identity VERIFICATION (upgraded from the earlier mechanism-only
    # argument): every parameter tensor snapshotted at make_private time is
    # torch.equal-compared against the live model's returned state_dict.
    _mismatched = []
    for _pname, _init_t in _captured.get("initial_params", {}).items():
        if _pname not in final_state:
            _mismatched.append(f"{_pname} (missing from final state_dict)")
        elif not torch.equal(_init_t, final_state[_pname].detach().cpu()):
            _mismatched.append(_pname)
    weights_byte_identical = len(_mismatched) == 0
    weight_update_occurred = not weights_byte_identical
    print(f"weights_byte_identical={weights_byte_identical}"
          + (f"  MISMATCHES: {_mismatched[:8]}" if _mismatched else "  (all parameter tensors torch.equal)"))

    # projected privacy-accounting inputs ONLY -- q and an expected step count
    # for a full run at this logical batch size; NOT an epsilon. Final epsilon
    # must come from the same PRV accountant during an actual (or previously
    # completed) training call, per instruction -- never computed here.
    rounds = fl_cfg["federated"]["rounds"]
    epochs_per_round = fl_cfg["local_training"]["epochs_per_round"]
    projected_expected_steps = estimate_steps(n_client, logical_batch, epochs_per_round, rounds)
    projected_accounting = {
        "sample_rate_q": projected_q,
        "expected_total_local_steps_full_run": projected_expected_steps,
        "epochs_per_round": epochs_per_round, "communication_rounds": rounds,
        "epsilon": None,
        "epsilon_note": "NOT COMPUTED -- q/steps only, per instruction; requires the same PRV "
                        "accountant run over an actual (or previously completed) training call",
    }

    record = {
        "diagnostic": "noise_signal_probe",
        "note": "DIAGNOSTIC ONLY -- not a thesis result; zero parameter updates occurred (verified)",
        "subset_label": args.subset_label,
        "client_id": client_id, "freeze_stages": freeze_stages,
        "lr0": hyp["lr0"], "max_grad_norm": max_grad_norm, "sigma": args.sigma,
        "logical_batch_size": logical_batch, "physical_batch_size": physical_batch, "imgsz": imgsz,
        "weight_update_occurred": weight_update_occurred,
        "weights_byte_identical": weights_byte_identical,
        "projected_privacy_accounting_inputs": projected_accounting,
        "steps": step_records,
    }
    out_json = (f"results/diag_noise_signal_probe_client{client_id}_sigma{args.sigma}"
               f"_logB{logical_batch}_physB{physical_batch}_{args.subset_label}.json")
    Path("results").mkdir(exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nprojected_total_local_steps (full 40-round run @ logical_batch={logical_batch}): "
          f"{projected_expected_steps}  (epsilon NOT computed -- see note in JSON)")
    print(f"Saved {out_json}")
    print("Do NOT change C, sigma, or lr0 based on this probe alone -- report back for the next step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
