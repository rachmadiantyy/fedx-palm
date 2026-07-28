#!/usr/bin/env python3
"""Plot B2's real per-round validation mAP50 trajectory for each training
seed, straight from history.json (written every round by
fedxpalm.federated.server.run_federated_training) -- a genuine convergence
curve from the actual training run, not a bar chart of only the final
three numbers.

This is the intended source for Figure 6 (manuscript) / Gambar 4.3 (Bab
4.2): it shows the same "best round varied a lot per seed (9 / 11 / 40)"
story in Table 4.2 as an actual curve, which a bar chart of endpoints alone
cannot show.

Read-only: only reads history.json files already written by scripts/06; does
not train or touch any checkpoint.

    python scripts/43_plot_b2_convergence.py \\
        --history runs/b2_federated/k4_seed42/history.json --label "seed 42" \\
        --history runs/b2_federated/k4_seed123/history.json --label "seed 123" \\
        --history runs/b2_federated/k4_seed2026/history.json --label "seed 2026" \\
        --b1-map50 0.8820

(pass one --history per seed, in the same order as --label; history.json
itself only stores per-round/per-client derived seeds -- not the base
experiment seed 42/123/2026 -- so the label isn't inferred automatically.
If --label is omitted for a --history, its parent directory name is used,
e.g. "k4_seed42".)
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_curve(path: str) -> tuple[list[int], list[float]]:
    with open(path) as f:
        history = json.load(f)
    rounds, map50s = [], []
    for record in history:
        val = record.get("val")
        if val and val.get("map50") is not None:
            rounds.append(record["round"] + 1)  # 1-indexed communication round, matches Table 4.2
            map50s.append(val["map50"])
    return rounds, map50s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", action="append", required=True,
                        help="path to one seed's history.json (repeat per seed)")
    parser.add_argument("--label", action="append", default=[],
                        help="legend label for the --history at the same position "
                             "(defaults to that file's parent directory name)")
    parser.add_argument("--b1-map50", type=float, default=None,
                        help="B1 held-out test mAP50 (Table 4.1), drawn as a reference line")
    parser.add_argument("--out", default="results/figure_b2_convergence.png")
    args = parser.parse_args()

    labels = args.label + [Path(p).parent.name for p in args.history[len(args.label):]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for path, label in zip(args.history, labels):
        rounds, map50s = load_curve(path)
        if not rounds:
            print(f"WARNING: no rounds with 'val' in {path}, skipping")
            continue
        ax.plot(rounds, map50s, marker="o", markersize=3, linewidth=1.3, label=label)
        best_idx = max(range(len(map50s)), key=lambda i: map50s[i])
        ax.scatter([rounds[best_idx]], [map50s[best_idx]], s=45, zorder=5,
                  edgecolors="black", linewidths=0.8)

    if args.b1_map50 is not None:
        ax.axhline(args.b1_map50, color="grey", linestyle="--", linewidth=1,
                  label=f"B1 held-out test mAP50 = {args.b1_map50:.4f}")

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation mAP@0.5")
    ax.set_title("B2 (FedAvg, Non-IID, K=4) validation mAP50 per round")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")
    print("Filled markers show each seed's best-validation round (matches the 'Round' "
          "column in Table 4.2). Use this as Figure 6 / Gambar 4.3 instead of a bar "
          "chart of the three final numbers -- it's the real per-round trajectory.")
