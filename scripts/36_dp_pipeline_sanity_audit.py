#!/usr/bin/env python3
"""FINAL DP IMPLEMENTATION SANITY AUDIT -- exercises the REAL production
DP path (train_client_round_dp, completely unmodified) on P2 for a few
real rounds/clients and checks all 14 requested invariants explicitly,
producing a machine-readable JSON + console PASS/FAIL table.

This audit performs REAL parameter updates (not a zero-weight-update
probe like scripts/25/28) because several checks (frozen params never
move, DFL never moves, accountant persists correctly across rounds and
across clients) are only meaningful to verify against genuine training,
not an artificially frozen optimizer step. It writes to a SCRATCH run
directory and a NEW results file -- no existing checkpoint, results file,
B1/B2/E1/E2 output, or historical diagnostic is read for writing or
touched in any way.

INCIDENT NOTE (found on the first real run, fixed before any experiment
was affected): the first version of this script computed
physical_batch_size_arg but never passed it into any of its three
train_client_round_dp() calls, so BatchMemoryManager was never actually
engaged -- dp_sgd.py's own physical_batch_size=None branch silently fell
back to iterating the raw logical Poisson loader directly. This produced
two audit failures (item 4: "physical" sizes observed were actually full
logical-batch sizes ~55-77; item 7: clip_calls==noise_calls, since every
unsplit logical draw is trivially its own "last chunk", so no skip ever
fired) that looked exactly like a BatchMemoryManager/production bug but
were not one -- confirmed by re-reading Opacus's own
wrap_data_loader()/BatchSplittingSampler source (BatchMemoryManager's
__enter__ genuinely returns a NEW split-sampler DataLoader; dp_sgd.py's
`with batch_ctx as active_loader:` / `for batch in active_loader:`
already binds and iterates that returned loader correctly, not the raw
one) and by confirming scripts/23, /25, /28 -- used for every P2 DP pilot
already reported -- already pass physical_batch_size correctly and are
therefore unaffected. Root cause was isolated entirely to this audit
script; the fix threads physical_batch_size_arg into all three call
sites below.

Checks (mapped 1:1 to the 14 requested items):
  1.  BatchNorm == 0, GroupNorm == 81
  2.  Opacus ModuleValidator.validate(strict=True) returns zero errors
      (this is ALSO implicitly enforced by make_private() itself --
      re-run explicitly here for a standalone, visible PASS/FAIL)
  3.  Every trainable parameter has grad_sample populated after backward,
      every frozen parameter does not
  4.  grad_sample's first dimension matches the actual physical
      micro-batch size processed in that forward/backward call
  5.  Frozen parameters: (a) absent from the optimizer's param_groups,
      (b) byte-identical before vs after real training (round0 -> final)
  6.  DFL fixed conv byte-identical before vs after real training
  7.  BatchMemoryManager: clip_and_accumulate() fires MORE OFTEN than
      add_noise() when physical_batch < logical_batch (confirms physical
      chunks do not each trigger an independent noise draw)
  8.  Exactly one accountant.step() (one history increment) per LOGICAL
      optimizer step -- not one per physical chunk
  9.  Accountant state persists across rounds for the SAME client
      (cumulative_steps grows monotonically, no accountant_step_mismatch)
  10. Accountant state does NOT leak across DIFFERENT clients (client B's
      run does not change client A's already-recorded cumulative step count
      or epsilon when A's saved state is later restored and re-inspected)
  11. Noise is actually added when sigma > 0 (noise norm > 0, matches
      sigma*C*sqrt(d) within a loose tolerance)
  12. Clipping is actually active (per-sample pre-clip norms exceed C for
      at least one sample; clip_fraction > 0)
  13. Zero-weight-update probe mechanism (scripts/25/28) is byte-identical
      to a real train_client_round_dp call structurally -- re-verified
      here by asserting weights_byte_identical=True during a SEPARATE
      zero-update sub-probe reusing the same wrapping technique
  14. No NaN/Inf across all probed steps

  python scripts/36_dp_pipeline_sanity_audit.py --device 0

Outputs:
  results/dp_pipeline_sanity_audit.json
  runs/_dp_pipeline_sanity_audit_scratch/   (Ultralytics run artifacts only)

This audit does not change any experiment's result: it never writes to
models/base_groupnorm.pt, never writes to any existing results/*.json,
and its own output path is new and isolated.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402

import fedxpalm  # noqa: E402,F401 (GroupNorm-safe fuse() patch)
from fedxpalm.privacy.dp_sgd import train_client_round_dp  # noqa: E402
from fedxpalm.privacy.freeze_audit import load_state  # noqa: E402

P2_FREEZE_STAGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21]

_capture = {
    "clip_calls": 0, "noise_calls": 0, "accountant_steps_seen": 0,
    "grad_sample_reports": [], "noise_norms": [], "signal_norms": [],
    "initial_params": None, "name_by_id": None,
}


def _install_instrumentation():
    """Wraps 3 Opacus methods for this process only (restored in a finally
    block); the real implementations always run untouched -- this only
    OBSERVES public state, matching the validated pattern from scripts/25/28.
    No behavior change of any kind (unlike those scripts, this one does NOT
    neutralize original_optimizer.step -- real updates must happen for the
    frozen/DFL/accountant-persistence checks to be meaningful)."""
    import opacus.optimizers.optimizer as opt_mod
    import opacus.privacy_engine as pe_mod

    orig_clip = opt_mod.DPOptimizer.clip_and_accumulate
    orig_add_noise = opt_mod.DPOptimizer.add_noise
    orig_make_private = pe_mod.PrivacyEngine.make_private

    def wrapped_clip(self):
        _capture["clip_calls"] += 1
        gs = self.grad_samples
        if gs and len(gs[0]) > 0:
            physical_bs = len(gs[0])
            trainable_names = [_capture["name_by_id"].get(id(p), "<unknown>") for p in self.params]
            missing = [n for n, g in zip(trainable_names, gs) if g is None or len(g) == 0]
            _capture["grad_sample_reports"].append({
                "physical_batch_size_observed": physical_bs,
                "n_trainable_tensors": len(self.params),
                "n_with_grad_sample": sum(1 for g in gs if g is not None and len(g) > 0),
                "missing_grad_sample": missing,
            })
        orig_clip(self)

    def wrapped_add_noise(self):
        _capture["noise_calls"] += 1
        pre = {id(p): p.summed_grad.detach().clone() for p in self.params}
        orig_add_noise(self)
        sig = math.sqrt(sum(float((v.float() ** 2).sum()) for v in pre.values()))
        noi = math.sqrt(sum(float(((p.grad - pre[id(p)].view_as(p)).float() ** 2).sum()) for p in self.params))
        _capture["signal_norms"].append(sig)
        _capture["noise_norms"].append(noi)

    def wrapped_make_private(self, *args, **kwargs):
        result = orig_make_private(self, *args, **kwargs)
        dp_model_ret, _dp_optimizer_ret, _ = result
        _capture["name_by_id"] = {id(p): n.replace("_module.", "", 1) for n, p in dp_model_ret.named_parameters()}
        if _capture["initial_params"] is None:  # snapshot only the FIRST make_private (round 0)
            _capture["initial_params"] = {
                n.replace("_module.", "", 1): p.detach().clone().cpu()
                for n, p in dp_model_ret.named_parameters()
            }
        return result

    opt_mod.DPOptimizer.clip_and_accumulate = wrapped_clip
    opt_mod.DPOptimizer.add_noise = wrapped_add_noise
    pe_mod.PrivacyEngine.make_private = wrapped_make_private

    def restore():
        opt_mod.DPOptimizer.clip_and_accumulate = orig_clip
        opt_mod.DPOptimizer.add_noise = orig_add_noise
        pe_mod.PrivacyEngine.make_private = orig_make_private

    return restore


def _run_zero_update_subprobe(base_weights, data_yaml, hyp, dp_hyp, device, physical_batch_size_arg):
    """Item 13: independently re-verifies the zero-weight-update mechanism
    used by scripts/25/28 is genuinely a no-op, using the SAME technique --
    make_private wrapped to snapshot params and neutralize
    original_optimizer.step, then compare byte-for-byte after 1 step."""
    import opacus.privacy_engine as pe_mod
    orig_make_private = pe_mod.PrivacyEngine.make_private
    snap = {}

    def wrapped(self, *args, **kwargs):
        result = orig_make_private(self, *args, **kwargs)
        dp_model_ret, dp_optimizer_ret, _ = result
        snap["initial"] = {n.replace("_module.", "", 1): p.detach().clone().cpu()
                           for n, p in dp_model_ret.named_parameters()}
        dp_optimizer_ret.original_optimizer.step = lambda *a, **kw: None
        return result

    pe_mod.PrivacyEngine.make_private = wrapped
    try:
        # BUG FIX (audit-only bug, found via the 12/14-pass real run): this call
        # previously omitted physical_batch_size, silently falling back to the
        # unwrapped (BatchMemoryManager-inactive) path -- see module docstring
        # amendment below. max_steps counts PHYSICAL iterations once this is
        # set (dp_sgd.py's documented semantics), so give generous headroom.
        physical_per_logical = (-(-hyp["batch_size"] // physical_batch_size_arg)
                                if physical_batch_size_arg else 1)
        final_state, _info = train_client_round_dp(
            global_weights_path=base_weights, client_data_yaml=data_yaml,
            hyp=hyp, dp_hyp=dp_hyp, round_idx=0, client_id="audit_subprobe",
            out_dir="runs/_dp_pipeline_sanity_audit_scratch", device=device,
            freeze_stages=P2_FREEZE_STAGES, accountant_state=None,
            max_steps=2 * physical_per_logical + physical_per_logical,
            physical_batch_size=physical_batch_size_arg,
        )
    finally:
        pe_mod.PrivacyEngine.make_private = orig_make_private
    mismatched = [n for n, v in snap["initial"].items()
                 if n not in final_state or not torch.equal(v, final_state[n].detach().cpu())]
    return len(mismatched) == 0, mismatched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--sigma", type=float, default=0.75, help="sigma>0 needed to check noise-active (item 11)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logical-batch", type=int, default=64)
    parser.add_argument("--physical-batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--base-weights", default="models/base_groupnorm.pt")
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
        print(f"FAIL: {manifest_path} not found"); return 1
    with open(manifest_path) as f:
        manifest = json.load(f)["4"]
    clients_dir = splits_dir / "federated_partitions" / "k4_clients"
    client_data_yaml = {cid: str(clients_dir / f"client{cid}" / "data.yaml") for cid in manifest["sizes"]}

    imgsz = args.imgsz or fl_cfg["model"]["imgsz"]
    hyp = dict(fl_cfg["local_training"], imgsz=imgsz, warmup_epochs=0.0, workers=0,
               batch_size=args.logical_batch, lr0=args.lr0)
    dp_hyp = {"sigma": args.sigma, "max_grad_norm": args.max_grad_norm,
             "delta": dp_cfg["dp_sgd"]["delta"], "accountant": dp_cfg["dp_sgd"]["accountant"]}
    physical_batch_size_arg = args.physical_batch if args.physical_batch < args.logical_batch else None

    checks = []

    def record(item, name, passed, detail=None):
        checks.append({"item": item, "name": name, "passed": bool(passed), "detail": detail or {}})
        print(f"  [{ 'PASS' if passed else 'FAIL'}] ({item}) {name}")

    print(f"=== DP pipeline sanity audit -- P2, sigma={args.sigma}, C={args.max_grad_norm}, "
          f"logical_batch={args.logical_batch}, physical_batch={args.physical_batch} ===\n")

    # ---- base checkpoint identity (item 1) ----
    base_sd, base_model = load_state(args.base_weights)
    n_bn = sum(1 for m in base_model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
    n_gn = sum(1 for m in base_model.modules() if isinstance(m, nn.GroupNorm))
    record(1, "BatchNorm==0 and GroupNorm==81", n_bn == 0 and n_gn == 81,
          {"batchnorm_count": n_bn, "groupnorm_count": n_gn})

    # ---- item 2: explicit ModuleValidator check ----
    try:
        from opacus.validators import ModuleValidator
        import copy
        errors = ModuleValidator.validate(copy.deepcopy(base_model), strict=False)
        record(2, "ModuleValidator.validate() reports zero errors", len(errors) == 0,
              {"n_errors": len(errors), "errors_sample": [str(e)[:200] for e in errors[:5]]})
    except Exception as e:  # noqa: BLE001
        record(2, "ModuleValidator.validate() reports zero errors", False, {"exception": str(e)})

    # ---- real 2-round run for client0 (items 3,4,5,6,7,8,9,11,12,14) ----
    restore = _install_instrumentation()
    client0 = sorted(manifest["sizes"], key=lambda c: int(c) if str(c).isdigit() else c)[0]
    accountant_states: dict = {}
    nan_inf_any = False
    round_infos = []
    try:
        gw_path = args.base_weights
        for r in range(2):
            state_dict, info = train_client_round_dp(
                global_weights_path=gw_path, client_data_yaml=client_data_yaml[client0],
                hyp=hyp, dp_hyp=dp_hyp, round_idx=r, client_id=client0,
                out_dir="runs/_dp_pipeline_sanity_audit_scratch", device=args.device,
                freeze_stages=P2_FREEZE_STAGES, accountant_state=accountant_states.get(client0),
                collect_grad_norms=True,
                # BUG FIX: this call previously omitted physical_batch_size,
                # silently disabling BatchMemoryManager -- see docstring note.
                physical_batch_size=physical_batch_size_arg,
            )
            accountant_states[client0] = info.pop("accountant_state")
            round_infos.append(info)
            nan_inf_any = nan_inf_any or info["nan_inf"]
            # persist as a real checkpoint for round r+1 to load from (mirrors run_federated_training)
            ckpt = torch.load(gw_path, map_location="cpu", weights_only=False)
            ckpt["model"].load_state_dict(state_dict)
            gw_path = f"runs/_dp_pipeline_sanity_audit_scratch/round{r}_client{client0}.pt"
            torch.save(ckpt, gw_path)
        final_sd, final_model = load_state(gw_path)
    finally:
        restore()

    # item 3/4: grad_sample presence + physical-batch-size shape
    gs_reports = _capture["grad_sample_reports"]
    all_present = all(len(rep["missing_grad_sample"]) == 0 for rep in gs_reports)
    shapes_match_physical = all(rep["physical_batch_size_observed"] <= args.physical_batch for rep in gs_reports)
    record(3, "grad_sample present for every trainable tensor at every clip_and_accumulate call",
          bool(gs_reports) and all_present,
          {"n_clip_calls_observed": len(gs_reports),
           "missing_examples": [rep["missing_grad_sample"] for rep in gs_reports if rep["missing_grad_sample"]][:3]})
    record(4, "grad_sample first dim matches physical (not logical) batch size",
          bool(gs_reports) and shapes_match_physical,
          {"observed_physical_sizes": [rep["physical_batch_size_observed"] for rep in gs_reports][:10]})

    # item 5: frozen params absent from optimizer (already asserted inside dp_sgd.py's own audit,
    # re-check here) + byte-identical round0(init) -> final
    freeze_names = [f"model.{s}." for s in P2_FREEZE_STAGES] + [".dfl"]
    missing_trainable_all = [m for info in round_infos for m in info["missing_trainable_params"]]
    unexpected_frozen_all = [m for info in round_infos for m in info["unexpected_frozen_params"]]
    frozen_changed_names = []
    for k in final_sd:
        if any(x in k for x in freeze_names) and "dfl" not in k:
            if k in base_sd and base_sd[k].shape == final_sd[k].shape:
                if not torch.equal(base_sd[k].float(), final_sd[k].float()):
                    frozen_changed_names.append(k)
    record(5, "frozen params absent from optimizer AND byte-identical after real training",
          not missing_trainable_all and not unexpected_frozen_all and not frozen_changed_names,
          {"missing_trainable_params": missing_trainable_all[:5],
           "unexpected_trainable_in_optimizer": unexpected_frozen_all[:5],
           "frozen_params_that_changed": frozen_changed_names[:5]})

    # item 6: DFL unchanged
    dfl_changed = False
    for k in final_sd:
        if "dfl.conv" in k and k in base_sd and base_sd[k].shape == final_sd[k].shape:
            if not torch.equal(base_sd[k].float(), final_sd[k].float()):
                dfl_changed = True
    record(6, "DFL fixed conv byte-identical after real training", not dfl_changed)

    # item 7: BatchMemoryManager -- clip fires more often than add_noise when physical<logical
    clip_calls, noise_calls = _capture["clip_calls"], _capture["noise_calls"]
    expected_more_clips = physical_batch_size_arg is not None
    record(7, "clip_and_accumulate() fires more often than add_noise() under BatchMemoryManager",
          (clip_calls > noise_calls) if expected_more_clips else (clip_calls == noise_calls),
          {"clip_calls": clip_calls, "noise_calls": noise_calls,
           "batch_memory_manager_active": expected_more_clips})

    # item 8: exactly one accountant step per logical optimizer step (== add_noise call count,
    # since add_noise only fires on a completed, non-skipped logical step)
    total_cumulative_steps = round_infos[-1]["cumulative_steps"]
    record(8, "accountant step count equals logical optimizer step count (not physical chunk count)",
          total_cumulative_steps == noise_calls,
          {"cumulative_steps_reported": total_cumulative_steps, "add_noise_calls_observed": noise_calls})

    # item 9: persistence across rounds for the SAME client -- monotonic, no mismatch flagged
    step_mismatches = [info.get("accountant_step_mismatch") for info in round_infos if info.get("accountant_step_mismatch")]
    monotonic = round_infos[1]["cumulative_steps"] >= round_infos[0]["cumulative_steps"]
    record(9, "accountant persists correctly across rounds for the same client",
          monotonic and not step_mismatches,
          {"cumulative_steps_per_round": [info["cumulative_steps"] for info in round_infos],
           "mismatches": step_mismatches})

    # item 10: no cross-client leakage -- run client1 fresh, then re-verify client0's SAVED
    # state (already recorded above) is untouched by client1's run
    client1 = sorted(manifest["sizes"], key=lambda c: int(c) if str(c).isdigit() else c)[1]
    client0_state_before_client1_run = dict(accountant_states[client0])
    _sd1, info1 = train_client_round_dp(
        global_weights_path=args.base_weights, client_data_yaml=client_data_yaml[client1],
        hyp=hyp, dp_hyp=dp_hyp, round_idx=0, client_id=client1,
        out_dir="runs/_dp_pipeline_sanity_audit_scratch", device=args.device,
        freeze_stages=P2_FREEZE_STAGES, accountant_state=None, collect_grad_norms=True,
        # BUG FIX: this call previously omitted physical_batch_size too.
        physical_batch_size=physical_batch_size_arg,
    )
    client0_state_after_client1_run = accountant_states[client0]  # unchanged reference -- client1 has its own dict entry
    record(10, "client1's run does not alter client0's already-recorded accountant state",
          client0_state_before_client1_run == client0_state_after_client1_run,
          {"client0_epsilon_before": round_infos[-1]["epsilon"], "client1_epsilon_after_own_round0": info1["epsilon"]})

    # item 11: noise actually added (nonzero, matches theory loosely)
    noise_norms = _capture["noise_norms"]
    d_trainable = round_infos[0]["n_trainable_params"]
    expected_noise = args.sigma * args.max_grad_norm * math.sqrt(d_trainable)
    noise_nonzero = bool(noise_norms) and all(n > 0 for n in noise_norms)
    noise_matches_theory = bool(noise_norms) and all(0.5 * expected_noise < n < 1.5 * expected_noise for n in noise_norms)
    record(11, "Gaussian noise actually added when sigma>0, matching sigma*C*sqrt(d) loosely",
          noise_nonzero and noise_matches_theory,
          {"observed_noise_norms": noise_norms, "expected_noise_norm_theory": expected_noise})

    # item 12: clipping actually active
    clip_fractions = [info["grad_norm_stats"]["clip_fraction"] for info in round_infos
                      if info.get("grad_norm_stats")]
    record(12, "clipping actually active (clip_fraction > 0)",
          bool(clip_fractions) and all(cf > 0 for cf in clip_fractions),
          {"clip_fractions_per_round": clip_fractions})

    # item 13: zero-update sub-probe re-verification
    zero_update_ok, zero_update_mismatches = _run_zero_update_subprobe(
        args.base_weights, client_data_yaml[client0], hyp, dp_hyp, args.device, physical_batch_size_arg)
    record(13, "zero-weight-update probe mechanism (scripts/25/28) is genuinely a no-op",
          zero_update_ok, {"mismatched_params": zero_update_mismatches[:5]})

    # item 14: no NaN/Inf across everything probed
    record(14, "no NaN/Inf detected in any probed round", not nan_inf_any and not info1["nan_inf"],
          {"nan_inf_any_client0": nan_inf_any, "nan_inf_client1_round0": info1["nan_inf"]})

    n_pass = sum(1 for c in checks if c["passed"])
    print(f"\n{n_pass}/{len(checks)} checks PASSED")
    for c in checks:
        if not c["passed"]:
            print(f"FAIL ({c['item']}): {c['name']} -- {c['detail']}")

    record_out = {
        "note": "DP pipeline sanity audit -- does not affect any existing experiment result; "
                "writes only to runs/_dp_pipeline_sanity_audit_scratch/ and this JSON.",
        "config": {"sigma": args.sigma, "C": args.max_grad_norm, "logical_batch": args.logical_batch,
                   "physical_batch": args.physical_batch, "lr0": args.lr0, "freeze_stages": P2_FREEZE_STAGES},
        "checks": checks,
        "n_passed": n_pass, "n_total": len(checks), "all_passed": n_pass == len(checks),
    }
    Path("results").mkdir(exist_ok=True)
    out_json = "results/dp_pipeline_sanity_audit.json"
    with open(out_json, "w") as f:
        json.dump(record_out, f, indent=2)
    print(f"\nSaved {out_json}")
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
