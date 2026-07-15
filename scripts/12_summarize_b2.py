#!/usr/bin/env python3
"""Aggregate results/b2_k{K}_seed*.json into b2_k{K}_summary.csv + .md
(mean +/- SD across seeds, best/worst run), per the multi-seed
reproducibility protocol (target: mean test mAP@0.5, not one lucky seed).
"""
import argparse
import csv
import glob
import json
import statistics as st
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    paths = sorted(glob.glob(f"{args.results_dir}/b2_k{args.k}_seed*.json"))
    if not paths:
        print(f"No results/b2_k{args.k}_seed*.json found -- run scripts/06 first.")
        return

    runs = []
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        runs.append({
            "file": Path(p).name, "seed": r.get("seed"),
            "map50": r["map50"], "map50_95": r["map50_95"],
            "precision": r["precision"], "recall": r["recall"],
            "best_round": r.get("best_round"), "rounds": r.get("communication_rounds"),
            "local_epochs": r.get("local_epochs"), "lr0": r.get("learning_rate"),
        })

    csv_path = f"{args.results_dir}/b2_k{args.k}_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
        w.writeheader()
        w.writerows(runs)

    maps = [r["map50"] for r in runs]
    maps95 = [r["map50_95"] for r in runs]
    mean50 = st.mean(maps)
    sd50 = st.stdev(maps) if len(maps) > 1 else 0.0
    mean95 = st.mean(maps95)
    sd95 = st.stdev(maps95) if len(maps95) > 1 else 0.0
    best = max(runs, key=lambda r: r["map50"])
    worst = min(runs, key=lambda r: r["map50"])

    md = [f"# B2 K={args.k} — multi-seed summary\n",
          f"Runs: {len(runs)} (seeds: {', '.join(str(r['seed']) for r in runs)})\n",
          "| Seed | test mAP@0.5 | test mAP@0.5:0.95 | Precision | Recall | Best round |",
          "|---|---|---|---|---|---|"]
    for r in runs:
        md.append(f"| {r['seed']} | {r['map50']:.4f} | {r['map50_95']:.4f} "
                  f"| {r['precision']:.4f} | {r['recall']:.4f} | {r['best_round']} |")
    md += ["",
           f"- **mean mAP@0.5 = {mean50:.4f} ± {sd50:.4f}**",
           f"- mean mAP@0.5:0.95 = {mean95:.4f} ± {sd95:.4f}",
           f"- best run: seed {best['seed']} ({best['map50']:.4f}); "
           f"worst run: seed {worst['seed']} ({worst['map50']:.4f})",
           f"- target check: mean mAP@0.5 > 0.85 → **{'TERCAPAI' if mean50 > 0.85 else 'BELUM'}**"]
    md_path = f"{args.results_dir}/b2_k{args.k}_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\nSaved {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
