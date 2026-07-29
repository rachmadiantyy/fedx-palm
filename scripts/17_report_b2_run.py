#!/usr/bin/env python3
"""Full post-run report for one B2 federated run: validation curve, best vs
final round, runtime, per-client participation, FedAvg weights, and a
test-hygiene check (test fields must be null on tuning runs).

    python scripts/17_report_b2_run.py --results results/b2_k4_seed42_leakagefree_stageA.json

Reads the run's history.json (via the checkpoint paths inside the results
JSON) and the partition manifest. Read-only; exits non-zero if test hygiene
is violated on a run without --eval-test.
"""
import argparse
import json
import statistics as st
from pathlib import Path

TEST_FIELDS = ("map50", "map50_95", "precision", "recall", "per_class")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="results/b2_*.json from scripts/06")
    parser.add_argument("--last-n", type=int, default=5, help="window for the plateau check")
    args = parser.parse_args()

    with open(args.results) as f:
        rec = json.load(f)

    run_dir = Path(rec["checkpoint"]).parent
    hist_path = run_dir / "history.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() else []

    print("=" * 72)
    print(f"  B2 run report: {args.results}")
    print("=" * 72)
    print(f"K={rec['num_clients']}  seed={rec['seed']}  partition_seed={rec['partition_seed']}  "
          f"alpha={rec['dirichlet_alpha']}")
    print(f"rounds={rec['communication_rounds']}  local_epochs={rec['local_epochs']}  "
          f"lr0={rec['learning_rate']}  lr_round_decay={rec['lr_round_decay']}  "
          f"warmup={rec['warmup_epochs']}  batch={rec['batch_size']}  imgsz={rec['imgsz']}  "
          f"opt={rec['optimizer']}  norm={rec['normalization']}")
    print(f"sample exposure={rec.get('total_sample_exposure'):,}  "
          f"optimizer steps={rec.get('total_optimizer_steps'):,}")

    # FedAvg weights from the partition manifest
    manifest_path = rec.get("split_manifest")
    if manifest_path and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text())[str(rec["num_clients"])]
        sizes = manifest["sizes"]
        total = sum(sizes.values())
        print(f"\nFedAvg weights (n_k / {total}):")
        for cid in sorted(sizes, key=lambda c: int(c) if str(c).isdigit() else c):
            print(f"  client{cid}: {sizes[cid]:>6} images  w={sizes[cid] / total:.6f}")
        print(f"  sum(w) = {sum(n / total for n in sizes.values()):.10f}")

    # participation + val curve from history
    curve = []
    if history:
        n_clients_expected = rec["num_clients"]
        bad_rounds = [h["round"] for h in history if len(h.get("clients", {})) != n_clients_expected]
        print(f"\nrounds logged: {len(history)}; every round has all {n_clients_expected} clients: "
              f"{'YES' if not bad_rounds else f'NO -- rounds {bad_rounds}'}")
        print(f"\n{'round':>6}{'val mAP50':>11}{'mAP50-95':>10}{'P':>8}{'R':>8}"
              f"{'sec':>8}  best?  lr(client0)")
        for h in history:
            v = h.get("val") or {}
            c0 = next(iter(h.get("clients", {}).values()), {})
            lr = c0.get("lr0_this_round", c0.get("lr/pg0", ""))
            star = "  *" if h.get("best_updated") else ""
            if v:
                curve.append((h["round"], v["map50"]))
            print(f"{h['round']:>6}{v.get('map50', float('nan')):>11.4f}"
                  f"{v.get('map50_95', float('nan')):>10.4f}{v.get('precision', float('nan')):>8.4f}"
                  f"{v.get('recall', float('nan')):>8.4f}{h.get('elapsed_sec', 0):>8.1f}{star:>7}  {lr}")

    best_r, best_v = rec.get("best_round"), rec.get("best_val_map50")
    final_v = rec.get("final_val_map50")
    if final_v is None and curve:
        final_v = curve[-1][1]
    print(f"\nbest round           : {best_r}")
    print(f"best val mAP@0.5     : {best_v}")
    bv = rec.get("best_val") or {}
    if bv:
        print(f"best val mAP@0.5:0.95: {bv.get('map50_95')}")
        print(f"best val precision   : {bv.get('precision')}")
        print(f"best val recall      : {bv.get('recall')}")
    print(f"final-round val mAP50: {final_v}")
    if best_v is not None and final_v is not None:
        print(f"final - best         : {final_v - best_v:+.4f}")
    print(f"total runtime        : {rec.get('total_runtime_sec', '?')} s "
          f"(mean {rec.get('mean_round_sec', '?')} s/round)")
    print(f"best_global.pt       : {rec.get('checkpoint')}")
    print(f"final_global.pt      : {rec.get('final_checkpoint')}")

    # curve diagnostics (descriptive only -- interpretation stays with you)
    if len(curve) >= 3:
        vals = [v for _, v in curve]
        drops = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
        last_n = vals[-args.last_n:]
        print(f"\ncurve: max {max(vals):.4f} @ round {curve[vals.index(max(vals))][0]}, "
              f"rounds since best: {curve[-1][0] - best_r if best_r is not None else '?'}")
        print(f"biggest round-to-round drop: {min(drops):+.4f}; "
              f"mean of last {len(last_n)} rounds: {st.mean(last_n):.4f} "
              f"(vs best {max(vals):.4f}, gap {max(vals) - st.mean(last_n):.4f})")

    # test hygiene (step 9): tuning runs must keep test fields null
    filled = [k for k in TEST_FIELDS if rec.get(k) is not None]
    if filled:
        print(f"\nTEST HYGIENE: test fields FILLED: {filled} -- fine ONLY if this run was "
              f"an explicit --eval-test run on the locked config; otherwise a violation.")
        return 1
    print("\nTEST HYGIENE: OK -- all test fields null (test set untouched).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
