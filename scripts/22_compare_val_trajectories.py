#!/usr/bin/env python3
"""Round-by-round validation comparison between two federated runs' history.json
files (produced by fedxpalm.federated.server.run_federated_training -- B2,
E1, E2, and the Diagnostic A/B/... scripts all share this exact schema, so
any two runs' early-round trajectories can be compared fairly).

Purely read-only: opens two existing history.json files, prints a side-by-side
table for the first N rounds of each, and writes nothing anywhere by default.
Does not train, does not touch any B1/B2/E1/E2 checkpoint or results file,
does not read the test split (history.json never contains test metrics).

    python scripts/22_compare_val_trajectories.py \\
        --a runs/diag_a_partial_nodp/k4_seed42/history.json --a-label A0_partial_nodp \\
        --b runs/b2_federated/k4_seed42_leakagefree_stageA_seedfix/history.json --b-label B2_seed42 \\
        --rounds 5

This purposefully prints RAW numbers and mechanical deltas only -- it does
not assert a root cause (e.g. "LR too high"). Use the printed trajectories
as evidence for the A0-vs-B2 comparison; draw conclusions separately.
"""
import argparse
import json
from pathlib import Path


def load_history(path):
    with open(path) as f:
        return json.load(f)


def val_row(round_record):
    v = round_record.get("val") or {}
    return {
        "round": round_record["round"],
        "map50": v.get("map50"),
        "map50_95": v.get("map50_95"),
        "precision": v.get("precision"),
        "recall": v.get("recall"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="path to first run's history.json")
    parser.add_argument("--b", required=True, help="path to second run's history.json")
    parser.add_argument("--a-label", default="A")
    parser.add_argument("--b-label", default="B")
    parser.add_argument("--rounds", type=int, default=5, help="compare the first N rounds of each (0-indexed 0..N-1)")
    args = parser.parse_args()

    if not Path(args.a).exists():
        print(f"FAIL: {args.a} not found"); return 1
    if not Path(args.b).exists():
        print(f"FAIL: {args.b} not found"); return 1

    hist_a = load_history(args.a)
    hist_b = load_history(args.b)
    rows_a = {r["round"]: val_row(r) for r in hist_a}
    rows_b = {r["round"]: val_row(r) for r in hist_b}

    n = args.rounds
    print(f"Comparing first {n} rounds (round index 0..{n-1}, i.e. communication round 1..{n})")
    print(f"  A = {args.a_label:<20} ({args.a})  [{len(hist_a)} rounds total]")
    print(f"  B = {args.b_label:<20} ({args.b})  [{len(hist_b)} rounds total]")
    print()
    header = (f"{'round':>6} | {args.a_label+' mAP50':>16} {args.a_label+' mAP50-95':>18} | "
             f"{args.b_label+' mAP50':>16} {args.b_label+' mAP50-95':>18} | {'delta(mAP50)':>13}")
    print(header)
    print("-" * len(header))

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "  n/a "

    comparison_rows = []
    for rd in range(n):
        ra = rows_a.get(rd)
        rb = rows_b.get(rd)
        m50_a = ra["map50"] if ra else None
        m95_a = ra["map50_95"] if ra else None
        m50_b = rb["map50"] if rb else None
        m95_b = rb["map50_95"] if rb else None
        delta = (m50_a - m50_b) if isinstance(m50_a, (int, float)) and isinstance(m50_b, (int, float)) else None
        print(f"{rd:>6} | {fmt(m50_a):>16} {fmt(m95_a):>18} | {fmt(m50_b):>16} {fmt(m95_b):>18} | {fmt(delta):>13}")
        comparison_rows.append({"round": rd, f"{args.a_label}_map50": m50_a, f"{args.a_label}_map50_95": m95_a,
                                f"{args.b_label}_map50": m50_b, f"{args.b_label}_map50_95": m95_b, "delta_map50": delta})

    valid_a = [r[f"{args.a_label}_map50"] for r in comparison_rows if r[f"{args.a_label}_map50"] is not None]
    valid_b = [r[f"{args.b_label}_map50"] for r in comparison_rows if r[f"{args.b_label}_map50"] is not None]
    if valid_a and valid_b:
        print(f"\nmean mAP50 over these {n} rounds: {args.a_label}={sum(valid_a)/len(valid_a):.4f}  "
              f"{args.b_label}={sum(valid_b)/len(valid_b):.4f}")
        print(f"{args.a_label} round-0 -> round-{n-1}: {valid_a[0]:.4f} -> {valid_a[-1]:.4f}  "
              f"({'up' if valid_a[-1] > valid_a[0] else 'down' if valid_a[-1] < valid_a[0] else 'flat'})")
        print(f"{args.b_label} round-0 -> round-{n-1}: {valid_b[0]:.4f} -> {valid_b[-1]:.4f}  "
              f"({'up' if valid_b[-1] > valid_b[0] else 'down' if valid_b[-1] < valid_b[0] else 'flat'})")
    print("\n(raw comparison only -- no root-cause conclusion asserted here)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
