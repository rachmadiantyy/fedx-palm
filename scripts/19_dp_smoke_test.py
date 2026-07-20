#!/usr/bin/env python3
"""Cheap DP mechanism smoke test -- run BEFORE any full E1/E2 sweep.

Runs a SHORT 2-round federated DP run (K=4, one sigma, small imgsz/batch so it
is fast -- this is a mechanism check, NOT a result) for the requested variant,
then asserts the accounting + freezing behavior. It never touches the test
split and never claims a privacy-utility result.

  python scripts/19_dp_smoke_test.py --variant partial --device 0   # E2 (default: also full)
  python scripts/19_dp_smoke_test.py --variant full    --device 0
  python scripts/19_dp_smoke_test.py --skip-run --variant partial    # analyze existing outputs

Assertions:
  - epsilon after round 2 > epsilon after round 1 (per client, cumulative)
  - cumulative_steps after round 2 == round-1 steps + round-2 steps (per client)
  - clients do not share accountant state (per-client cumulative steps track
    each client's own dataset size, not a shared counter)
  - the test split is never evaluated (results test fields null, no test key in history)
  - best checkpoint is selected from validation (best_global.pt exists; best_round
    matches the argmax of validation mAP@0.5 in history)
  - no NaN/Inf in any round
  - E2 only: backbone weights are byte-identical before vs after training, while
    neck+head weights change, and the optimizer received no frozen params
  - E1: no BatchNorm remains / GroupNorm present in the trained checkpoint
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

BACKBONE_MAX_STAGE = 10  # model.0..model.10 = backbone; 11..22 neck; 23 head

# Parameters that are frozen BY ARCHITECTURE, not a methodology error:
# Ultralytics always freezes the DFL fixed conv (weight = arange(reg_max),
# never trained) in every YOLO run, DP or not. A frozen param whose name
# matches one of these is expected; anything else that is frozen-but-in-the-
# optimizer is a real problem (a param that should train but won't).
EXPECTED_FROZEN_PATTERNS = ("dfl.conv",)


def _classify_frozen(names):
    expected = [n for n in names if any(p in n for p in EXPECTED_FROZEN_PATTERNS)]
    unexpected = [n for n in names if not any(p in n for p in EXPECTED_FROZEN_PATTERNS)]
    return expected, unexpected


def _run(variant, device, imgsz, batch, sigma, rounds, tag_suffix, workers):
    from _dp_sweep_common import run_dp_sweep  # noqa: E402
    # workers=0: keep dataloading in the main process. Opacus' _dict_safe_init
    # collate fn is a local (unpicklable) closure, so a spawned DataLoader
    # worker (workers>0) crashes on Windows -- workers=0 avoids it. Smoke path
    # only; the real E1/E2 runs use the fl_config worker count unchanged.
    return run_dp_sweep(variant, device=device, k_override=4, sigma_override=sigma,
                        rounds_override=rounds, imgsz_override=imgsz, batch_override=batch,
                        tag_suffix=tag_suffix, workers_override=workers)


def _stage_of(param_key: str) -> int | None:
    # keys look like "model.5.cv1.conv.weight"
    parts = param_key.split(".")
    if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
        return int(parts[1])
    return None


def _load_state(path):
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    return model.state_dict(), model


def audit_freeze(init_weights, final_weights):
    """Returns (backbone_changed, neck_head_changed, n_bn, n_gn)."""
    import torch
    import torch.nn as nn

    sd0, _ = _load_state(init_weights)
    sd1, model1 = _load_state(final_weights)
    backbone_changed = neck_head_changed = False
    for k in sd1:
        if k not in sd0 or sd0[k].shape != sd1[k].shape:
            continue
        stage = _stage_of(k)
        if stage is None:
            continue
        differs = not torch.equal(sd0[k].float(), sd1[k].float())
        if stage <= BACKBONE_MAX_STAGE:
            backbone_changed = backbone_changed or differs
        else:
            neck_head_changed = neck_head_changed or differs
    # DFL fixed conv must be byte-identical before/after: proof the DPOptimizer
    # never updated the architectural constant despite it sitting in a group.
    dfl_changed = False
    for k in sd1:
        if "dfl.conv" in k and k in sd0 and sd0[k].shape == sd1[k].shape:
            if not torch.equal(sd0[k].float(), sd1[k].float()):
                dfl_changed = True
    n_bn = sum(1 for m in model1.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
    n_gn = sum(1 for m in model1.modules() if isinstance(m, nn.GroupNorm))
    return backbone_changed, neck_head_changed, n_bn, n_gn, dfl_changed


def analyze(variant, tag_suffix):
    tag = ("e1_dp_full" if variant == "full" else "e2_dp_partial") + (f"_{tag_suffix}" if tag_suffix else "")
    results_path = Path("results") / f"{tag}_k4_sigma1.0.json"
    if not results_path.exists():
        # sigma may render without trailing .0 depending on input; find it
        cands = list(Path("results").glob(f"{tag}_k4_sigma*.json"))
        if not cands:
            print(f"FAIL: no results JSON for tag {tag}")
            return False
        results_path = cands[0]
    record = json.loads(results_path.read_text())
    history = json.loads(Path(record["history"]).read_text())

    failures = []

    # test hygiene
    if record.get("test_evaluated") is not False or any(record.get(f) is not None
                                                        for f in ("map50", "map50_95", "precision", "recall")):
        failures.append("test fields present / test_evaluated not False")
    if any("test" in (info or {}) for r in history for info in r["clients"].values()):
        failures.append("a client log references 'test'")

    # need >= 2 rounds
    if len(history) < 2:
        failures.append(f"history has {len(history)} round(s), need >= 2")
        _report(failures, None)
        return not failures

    r1, r2 = history[0]["clients"], history[1]["clients"]
    per_client_rows = []
    seen_cumulatives = {}
    expected_frozen_seen, unexpected_frozen_seen = set(), set()
    frozen_names_available = False
    for cid in r1:
        e1 = r1[cid]["epsilon"]; e2 = r2[cid]["epsilon"]
        s1 = r1[cid]["cumulative_steps"]; s2 = r2[cid]["cumulative_steps"]
        st1 = r1[cid]["steps_this_round"]; st2 = r2[cid]["steps_this_round"]
        per_client_rows.append((cid, r1[cid]["n_samples"], r1[cid]["sample_rate_q"],
                                st1, e1, st2, s2, e2))
        if not (e2 > e1):
            failures.append(f"client {cid}: epsilon not increasing ({e1} -> {e2})")
        if s1 != st1:
            failures.append(f"client {cid}: round-1 cumulative {s1} != round-1 steps {st1}")
        if s2 != st1 + st2:
            failures.append(f"client {cid}: cumulative {s2} != {st1}+{st2}")
        if r1[cid]["nan_inf"] or r2[cid]["nan_inf"]:
            failures.append(f"client {cid}: NaN/Inf detected")
        # optimizer<->model set audit.
        # Post-filter runs (have removed_frozen_from_optimizer_count): the
        # optimizer must contain EXACTLY the trainable set -- any frozen param
        # left (even the DFL fixed conv) means the pre-Opacus filter failed,
        # and any missing trainable param means the filter dropped too much.
        # Pre-filter artifacts: fall back to the allowlist (DFL expected).
        for rr in (r1[cid], r2[cid]):
            names = rr.get("frozen_in_optimizer")
            if names is None:
                continue  # pre-patch artifact: no names logged (see note below)
            frozen_names_available = True
            post_filter = "removed_frozen_from_optimizer_count" in rr
            missing = rr.get("missing_trainable_params") or []
            if missing:
                failures.append(f"client {cid}: trainable params MISSING from optimizer: {missing[:8]}")
            if post_filter:
                if names:
                    failures.append(f"client {cid}: frozen params still in optimizer after filter: {names[:8]}")
            else:
                exp, unexp = _classify_frozen(names)
                expected_frozen_seen.update(exp)
                unexpected_frozen_seen.update(unexp)
        seen_cumulatives[cid] = s2

    # accountants not shared: with different dataset sizes, per-client cumulative
    # steps must not all be identical
    if len(set(seen_cumulatives.values())) == 1 and len(seen_cumulatives) > 1:
        failures.append("all clients share identical cumulative steps -- accountants may be shared")

    # only UNEXPECTED frozen-in-optimizer params fail the audit
    if unexpected_frozen_seen:
        failures.append(f"unexpected frozen params in optimizer: {sorted(unexpected_frozen_seen)}")
    if not frozen_names_available:
        print("[!] this run's logs predate the frozen-name patch (only a boolean was logged); "
              "rerun the 2-round smoke with the patched dp_sgd.py to capture exact names.")

    # best checkpoint from validation
    val_maps = [(r["round"], (r.get("val") or {}).get("map50")) for r in history]
    val_maps = [(rd, m) for rd, m in val_maps if m is not None]
    if val_maps:
        best_by_val = max(val_maps, key=lambda x: x[1])[0]
        if record.get("best_round") != best_by_val:
            failures.append(f"best_round {record.get('best_round')} != val argmax {best_by_val}")
    else:
        failures.append("no validation mAP recorded in history")
    if not Path(record["best_weights"]).name.endswith("best_global.pt"):
        failures.append("best_weights is not best_global.pt")

    # freeze / norm audit
    freeze_info = None
    init_w = "models/base_groupnorm.pt"
    if Path(init_w).exists() and Path(record["final_weights"]).exists():
        bb_changed, nh_changed, n_bn, n_gn, dfl_changed = audit_freeze(init_w, record["final_weights"])
        freeze_info = (bb_changed, nh_changed, n_bn, n_gn, dfl_changed)
        if n_bn != 0:
            failures.append(f"{n_bn} BatchNorm layers remain (must be 0)")
        if n_gn == 0:
            failures.append("no GroupNorm layers found")
        # architectural constant must never be updated, in either variant
        if dfl_changed:
            failures.append("DFL fixed conv weight CHANGED (must stay constant)")
        if variant == "partial":
            if bb_changed:
                failures.append("E2: backbone weights CHANGED (must be frozen/identical)")
            if not nh_changed:
                failures.append("E2: neck+head weights did NOT change (should train)")
        elif variant == "full":
            if not (bb_changed or nh_changed):
                failures.append("E1: no weights changed at all")
    else:
        print(f"[!] skip freeze audit: missing {init_w} or {record['final_weights']}")

    _report(failures, per_client_rows, record, freeze_info, variant,
            sorted(expected_frozen_seen), sorted(unexpected_frozen_seen))
    return not failures


def _report(failures, rows, record=None, freeze_info=None, variant=None,
            expected_frozen=None, unexpected_frozen=None):
    if rows:
        print(f"\nPer-client epsilon table ({variant}):")
        print(f"{'client':<8}{'n':>7}{'q':>9}{'st_r1':>7}{'eps_r1':>10}{'st_r2':>7}{'cum_r2':>8}{'eps_r2':>10}")
        for cid, n, q, st1, e1, st2, s2, e2 in rows:
            print(f"{cid:<8}{n:>7}{q:>9.4f}{st1:>7}{e1:>10.4f}{st2:>7}{s2:>8}{e2:>10.4f}")
    if expected_frozen is not None:
        print(f"\nfrozen-in-optimizer (expected/architectural): {expected_frozen or 'none'}")
        print(f"frozen-in-optimizer (UNEXPECTED): {unexpected_frozen or 'none'}")
    if freeze_info is not None:
        bb, nh, n_bn, n_gn, dfl_changed = freeze_info
        print(f"freeze/norm audit: backbone_changed={bb}  neck_head_changed={nh}  "
              f"DFL_changed={dfl_changed}  BatchNorm={n_bn}  GroupNorm={n_gn}")
    if record is not None:
        print(f"test_evaluated={record.get('test_evaluated')}  "
              f"epsilon_max={record.get('epsilon_max_over_clients')}  best_round={record.get('best_round')}")
    print()
    for f in failures:
        print(f"FAIL: {f}")
    print("\nDP SMOKE TEST: PASSED" if not failures else "\nDP SMOKE TEST: FAILED")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["full", "partial", "both"], default="both")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=320, help="small for speed; smoke is a mechanism check")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=2, help="must be >= 2 to test accumulation")
    parser.add_argument("--tag-suffix", default="smoke2r")
    parser.add_argument("--workers", type=int, default=0,
                        help="dataloader workers for the smoke run (default 0: avoids the Windows "
                             "unpicklable-collate crash; does not affect the real E1/E2 pipeline)")
    parser.add_argument("--skip-run", action="store_true", help="analyze existing outputs only")
    args = parser.parse_args()

    if args.rounds < 2:
        print("rounds must be >= 2 for the accumulation test"); return 1

    variants = ["full", "partial"] if args.variant == "both" else [args.variant]
    all_ok = True
    for v in variants:
        print("=" * 72)
        print(f"  DP smoke test -- variant={v}")
        print("=" * 72)
        if not args.skip_run:
            _run(v, args.device, args.imgsz, args.batch, args.sigma, args.rounds,
                 args.tag_suffix, args.workers)
        ok = analyze(v, args.tag_suffix)
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
