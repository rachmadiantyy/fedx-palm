#!/usr/bin/env python3
"""PHASE B -- derive per-tensor clipping thresholds from Phase A's MEASURED
per-sample gradient-norm distributions (results/diag_p2_perstage_gradnorm.json).

Pre-registered rule (committed before any threshold value existed):

    C_i = m_i * C_total / ||m||_2

where m_i is tensor i's MEASURED median per-sample gradient L2 norm from the
Phase A audit and C_total = 1.0 -- deliberately equal to the flat baseline C,
so ||C_vec||_2 = C_total = 1.0 exactly: total sensitivity, Gaussian noise
scale (sigma * ||C_vec||_2), and privacy accounting are all IDENTICAL to the
flat C=1 runs, making the per-layer experiment a pure signal-REALLOCATION
change with everything else held fixed.

Floor (only if needed): any tensor whose measured median is zero/degenerate
would otherwise get C_i = 0, permanently silencing it. Such tensors receive
floor = 1e-4 * C_total / sqrt(T) (a vanishing share -- squared-budget
contribution T * floor^2 = 1e-8 * C_total^2 in the worst case), then the
whole vector is renormalized so ||C_vec||_2 = C_total EXACTLY. The number of
floored tensors is reported and saved; with Phase A's observed norms
(medians in the hundreds) it should be zero.

TENSOR-ORDER AUDIT: Opacus's DPPerLayerOptimizer takes a bare ordered list
(one C_i per optimizer parameter tensor, in optimizer order). To make
correctness independent of ordering, the thresholds are SAVED and CONSUMED
as a name->C_i dict -- dp_sgd.py's per_layer_max_grad_norms path rebuilds
the ordered list from the optimizer's own canonical sequence
(opacus.optimizers.utils.params) with exact-set-match assertions. This
script ADDITIONALLY proves the mapping end-to-end by rebuilding the real
trainer/optimizer exactly as train_client_round_dp does (same checkpoint,
same freeze, same frozen-param stripping) and asserting the live optimizer's
ordered names are exactly the Phase A audit's recorded tensor list -- so
threshold[i] provably corresponds to optimizer_param[i].

Output: results/p2_perlayer_clip_thresholds.json
  { thresholds: {name: C_i}, ordered_names, ordered_C, C_vec_l2_norm,
    per_stage_summary, n_floored, rule, source }

  python scripts/29_derive_perlayer_thresholds.py --device 0

No training. No modification to any existing output.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

P2_FREEZE_STAGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21]
P2_TRAINABLE_STAGES = [16, 19, 22, 23]


def derive_thresholds(per_tensor: list[dict], c_total: float = 1.0,
                      floor_scale: float = 1e-4) -> tuple[dict, dict]:
    """Pure derivation (unit-testable without any model/dataset): applies the
    pre-registered rule C_i = m_i * C_total / ||m||_2, floors zero/degenerate
    medians at floor_scale * C_total / sqrt(T), renormalizes so
    ||C_vec||_2 == C_total exactly. Returns (name->C_i, meta)."""
    names = [t["name"] for t in per_tensor]
    if len(set(names)) != len(names):
        raise ValueError("duplicate tensor names in per_tensor records")
    medians = [float(t["per_sample_norm"]["median"]) for t in per_tensor]
    if any(m < 0 for m in medians):
        raise ValueError("negative median norm -- corrupt input")
    T = len(medians)
    floor_val = floor_scale * c_total / math.sqrt(T)

    m_l2 = math.sqrt(sum(m * m for m in medians))
    if m_l2 == 0:
        raise ValueError("all measured medians are zero -- cannot derive thresholds")
    raw = [m * c_total / m_l2 for m in medians]
    floored_idx = [i for i, c in enumerate(raw) if c < floor_val]
    c = [max(v, floor_val) for v in raw]
    # renormalize so ||C||_2 == c_total EXACTLY (no-op when nothing floored)
    c_l2 = math.sqrt(sum(v * v for v in c))
    c = [v * c_total / c_l2 for v in c]
    c_l2_final = math.sqrt(sum(v * v for v in c))

    meta = {
        "rule": "C_i = median_i * C_total / ||median||_2, floor at "
                f"{floor_scale} * C_total / sqrt(T), renormalized to ||C||_2 == C_total",
        "C_total": c_total, "n_tensors": T,
        "n_floored": len(floored_idx),
        "floored_names": [names[i] for i in floored_idx],
        "C_vec_l2_norm": c_l2_final,
        "C_min": min(c), "C_median": sorted(c)[T // 2], "C_max": max(c),
    }
    return dict(zip(names, c)), meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--audit-json", default="results/diag_p2_perstage_gradnorm.json")
    parser.add_argument("--c-total", type=float, default=1.0,
                        help="total sensitivity budget ||C_vec||_2 (default 1.0 == the flat "
                             "baseline C, so noise scale and accounting stay identical)")
    parser.add_argument("--base-weights", default="models/base_groupnorm.pt")
    parser.add_argument("--skip-order-audit", action="store_true",
                        help="skip the live optimizer rebuild (order audit). NOT recommended -- "
                             "only for machines without the dataset/checkpoint")
    args = parser.parse_args()

    if not Path(args.audit_json).exists():
        print(f"FAIL: {args.audit_json} not found -- run scripts/28_diag_p2_perstage_gradnorm.py first")
        return 1
    with open(args.audit_json) as f:
        audit = json.load(f)
    per_tensor = audit["per_tensor"]
    audit_names = [t["name"] for t in per_tensor]

    thresholds, meta = derive_thresholds(per_tensor, c_total=args.c_total)

    # per-stage aggregate of the derived thresholds
    from fedxpalm.privacy.freeze_audit import stage_of
    per_stage = {}
    for s in P2_TRAINABLE_STAGES:
        cs = sorted(thresholds[n] for n in audit_names if stage_of(n) == s)
        if not cs:
            per_stage[str(s)] = {"n_tensors": 0}
            continue
        per_stage[str(s)] = {
            "n_tensors": len(cs), "C_min": cs[0], "C_median": cs[len(cs) // 2], "C_max": cs[-1],
            "sum_C_sq": sum(v * v for v in cs),
            "share_of_sq_budget": sum(v * v for v in cs) / (meta["C_vec_l2_norm"] ** 2),
        }

    print(f"derived {meta['n_tensors']} thresholds from {args.audit_json}")
    print(f"rule: {meta['rule']}")
    print(f"C_min={meta['C_min']:.6f}  C_median={meta['C_median']:.6f}  C_max={meta['C_max']:.6f}  "
          f"n_floored={meta['n_floored']}")
    print(f"||C_vec||_2 = {meta['C_vec_l2_norm']:.9f}  (must equal C_total={args.c_total})")
    print(f"\n{'stage':>6}{'tensors':>9}{'C_min':>10}{'C_median':>11}{'C_max':>10}{'sq-budget-share':>17}")
    for s in P2_TRAINABLE_STAGES:
        r = per_stage[str(s)]
        if r["n_tensors"] == 0:
            print(f"{s:>6}  (none)"); continue
        print(f"{s:>6}{r['n_tensors']:>9}{r['C_min']:>10.5f}{r['C_median']:>11.5f}{r['C_max']:>10.5f}"
              f"{r['share_of_sq_budget']:>17.4f}")

    # ---- tensor-order audit against the LIVE optimizer ----
    order_audit = {"performed": False}
    if args.skip_order_audit:
        print("\n[!] order audit SKIPPED by flag -- dp_sgd's name-dict mapping still enforces "
              "exact set-match at run time, but run the audit on the real machine before Phase C.")
    else:
        if not Path(args.base_weights).exists():
            print(f"FAIL: {args.base_weights} not found (needed for the order audit; "
                  f"use --skip-order-audit only on machines without it)")
            return 1
        import yaml
        import fedxpalm  # noqa: F401
        from opacus.optimizers.utils import params as opacus_params
        from fedxpalm.federated.trainer_utils import build_trainer_from_checkpoint
        from fedxpalm.privacy.dp_sgd import _strip_frozen_from_optimizer

        with open("configs/dataset.yaml") as f:
            ds_cfg = yaml.safe_load(f)
        with open("configs/fl_config.yaml") as f:
            fl_cfg = yaml.safe_load(f)
        splits_dir = Path(ds_cfg["output_dir"])
        data_yaml = str(splits_dir / "federated_partitions" / "k4_clients"
                        / f"client{audit['client_id']}" / "data.yaml")
        overrides = dict(
            data=data_yaml, model=args.base_weights, epochs=1,
            batch=audit["logical_batch"], imgsz=audit["imgsz"], optimizer="SGD",
            lr0=fl_cfg["local_training"]["lr0"], momentum=fl_cfg["local_training"]["momentum"],
            weight_decay=fl_cfg["local_training"]["weight_decay"], warmup_epochs=0.0,
            device=args.device, workers=0, amp=False, plots=False, val=False, verbose=False,
            freeze=P2_FREEZE_STAGES, exist_ok=True,
            project="runs/_derive_perlayer_scratch", name="order_audit",
        )
        trainer = build_trainer_from_checkpoint(args.base_weights, overrides)
        trainer._setup_train()
        _strip_frozen_from_optimizer(trainer.optimizer, trainer.model)
        name_by_id = {id(p): n for n, p in trainer.model.named_parameters()}
        live_names = [name_by_id[id(p)] for p in opacus_params(trainer.optimizer)]

        order_matches = live_names == audit_names
        set_matches = set(live_names) == set(audit_names)
        order_audit = {
            "performed": True,
            "n_live_optimizer_tensors": len(live_names),
            "n_audit_tensors": len(audit_names),
            "set_match": set_matches,
            "order_match": order_matches,
            "first_mismatch": next((i for i, (a, b) in enumerate(zip(live_names, audit_names))
                                    if a != b), None) if not order_matches else None,
        }
        print(f"\norder audit: live optimizer tensors={len(live_names)}  audit tensors={len(audit_names)}  "
              f"set_match={set_matches}  order_match={order_matches}")
        if not set_matches:
            print("FAIL: live optimizer's trainable set differs from the audited set -- "
                  "do NOT use these thresholds")
            return 1
        if not order_matches:
            print("[!] ordering differs from the audit capture -- harmless for dp_sgd's "
                  "name-dict consumption (order-independent), but investigate why")

    out = {
        "diagnostic": "p2_perlayer_clip_thresholds",
        "note": "Derived via the pre-registered rule from MEASURED Phase A medians. "
                "||C_vec||_2 == C_total == flat baseline C: total sensitivity, noise scale "
                "(sigma*||C_vec||_2) and privacy accounting are identical to flat clipping.",
        "source_audit_json": args.audit_json,
        "meta": meta,
        "per_stage_summary": per_stage,
        "order_audit": order_audit,
        "thresholds": thresholds,          # name -> C_i (what dp_sgd consumes)
        "ordered_names": audit_names,      # Phase A capture order (== optimizer order per audit)
        "ordered_C": [thresholds[n] for n in audit_names],
    }
    out_json = "results/p2_perlayer_clip_thresholds.json"
    Path("results").mkdir(exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
