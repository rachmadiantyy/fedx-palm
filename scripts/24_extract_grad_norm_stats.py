#!/usr/bin/env python3
"""Print the per-round per-client clipping-severity table
(clipping_severity_per_client_round) already saved inside a Diagnostic B
(scripts/23_diag_b_clipping_only.py) results JSON. Purely read-only -- does
not train, does not touch any checkpoint.

    python scripts/24_extract_grad_norm_stats.py results/diag_b_clipping_only_seed42.json
"""
import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json")
    args = parser.parse_args()

    if not Path(args.results_json).exists():
        print(f"FAIL: {args.results_json} not found"); return 1

    with open(args.results_json) as f:
        record = json.load(f)

    rows = record.get("clipping_severity_per_client_round") or []
    if not rows:
        print("no clipping_severity_per_client_round found in this results JSON "
              "(was it produced with collect_grad_norms=True?)")
        return 1

    print(f"{'round':>6}{'client':>8}{'n_obs':>7}{'median':>9}{'p75':>9}{'p90':>9}"
          f"{'p95':>9}{'max':>9}{'frac>C':>9}{'clip_frac':>11}")
    for row in rows:
        gs = row.get("grad_norm_stats")
        if not gs:
            print(f"{row['round']:>6}{row['client_id']:>8}  (no grad_norm_stats)")
            continue
        print(f"{row['round']:>6}{row['client_id']:>8}{gs['n_samples_observed']:>7}"
              f"{gs['median']:>9.3f}{gs['p75']:>9.3f}{gs['p90']:>9.3f}{gs['p95']:>9.3f}"
              f"{gs['max']:>9.3f}{gs['fraction_norm_above_C']:>9.3f}{gs['clip_fraction']:>11.3f}")

    print(f"\nC (max_grad_norm) = {record.get('max_grad_norm')}")
    print("(fraction_norm_above_C and clip_fraction are identical by construction under L2 clipping)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
