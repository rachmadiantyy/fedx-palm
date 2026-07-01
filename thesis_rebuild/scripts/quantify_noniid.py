"""Quantify Non-IID heterogeneity of a Dirichlet client partition.

Computes Chi-Square test, per-pair and average Jensen-Shannon Divergence
(JSD), and a per-client class distribution table. Use to substantiate
"Non-IID" claims in the paper with formal statistics (Riyadi et al. 2026
JIKI uses the same approach).

Use:
  python thesis_rebuild/scripts/quantify_noniid.py --K 4
  python thesis_rebuild/scripts/quantify_noniid.py --K 4 --out thesis_rebuild/tables/noniid_K4.md
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES = ["Abnormal", "Empty Bunch", "Overripe", "Ripe", "Underripe", "Unripe"]


def read_yolo_label_counts(labels_dir: Path) -> np.ndarray:
    """Return [n_class_0, n_class_1, ...] counts of bbox labels in a dir."""
    counts = np.zeros(len(CLASS_NAMES), dtype=int)
    if not labels_dir.exists():
        return counts
    for f in labels_dir.glob("*.txt"):
        for line in f.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            if 0 <= cls < len(CLASS_NAMES):
                counts[cls] += 1
    return counts


def chi_square(observed: np.ndarray) -> tuple[float, int, float]:
    """Chi-Square test of independence on a contingency table (K x C).

    Returns (chi2, dof, p_value_approx).
    p computed via a simple chi2-CDF approximation good enough for
    reporting orders of magnitude (Wilson-Hilferty).
    """
    obs = observed.astype(float)
    row = obs.sum(axis=1, keepdims=True)
    col = obs.sum(axis=0, keepdims=True)
    total = obs.sum()
    expected = row @ col / total
    mask = expected > 0
    chi2 = float(np.sum((obs[mask] - expected[mask]) ** 2 / expected[mask]))
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    # Wilson-Hilferty: ((chi2/dof)^(1/3) - (1 - 2/(9*dof))) / sqrt(2/(9*dof)) ~ N(0,1)
    if dof <= 0:
        return chi2, dof, 1.0
    z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    # One-sided p (upper tail of N(0,1))
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return chi2, dof, p


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2, in [0, 1])."""
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    eps = 1e-12

    def kl(a, b):
        a = a + eps
        b = b + eps
        return float(np.sum(a * np.log2(a / b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out", default=None,
                    help="optional markdown output path (default: stdout only)")
    args = ap.parse_args()

    clients_root = Path(args.data_root) / f"clients_K{args.K}"
    if not clients_root.exists():
        raise SystemExit(f"Client partition not found: {clients_root}")

    # Gather per-client class counts (from labels under each client dir).
    counts = np.zeros((args.K, len(CLASS_NAMES)), dtype=int)
    client_dirs = sorted(d for d in clients_root.iterdir()
                         if d.is_dir() and d.name.startswith("client_"))
    if len(client_dirs) != args.K:
        raise SystemExit(f"Expected {args.K} client dirs, found {len(client_dirs)}")
    for k, cdir in enumerate(client_dirs):
        # labels usually under client_k/labels/ (flat) per partition script
        labels_dir = cdir / "labels"
        counts[k] = read_yolo_label_counts(labels_dir)

    # Per-client proportions
    proportions = counts / counts.sum(axis=1, keepdims=True).clip(min=1)

    # Chi-Square
    chi2, dof, p = chi_square(counts)

    # JSD: all pairs + average
    pair_jsd = []
    for i in range(args.K):
        for j in range(i + 1, args.K):
            pair_jsd.append((i + 1, j + 1, jsd(proportions[i], proportions[j])))
    mean_jsd = float(np.mean([v for _, _, v in pair_jsd])) if pair_jsd else 0.0

    # Report
    lines: list[str] = []
    lines.append(f"# Non-IID quantification (K={args.K})\n")
    lines.append("## Per-client class counts (bounding-box instances)\n")
    header = "| Client | " + " | ".join(CLASS_NAMES) + " | Total |"
    sep = "|---|" + "|".join("---" for _ in CLASS_NAMES) + "|---|"
    lines += [header, sep]
    for k in range(args.K):
        row = " | ".join(str(int(v)) for v in counts[k])
        lines.append(f"| {k + 1} | {row} | {int(counts[k].sum())} |")
    lines.append("")
    lines.append("## Per-client class proportions\n")
    lines += [header, sep]
    for k in range(args.K):
        row = " | ".join(f"{v:.3f}" for v in proportions[k])
        lines.append(f"| {k + 1} | {row} | {1.0:.3f} |")
    lines.append("")
    lines.append("## Heterogeneity statistics\n")
    lines.append(f"- Chi-Square test on (K x C) counts: chi^2 = {chi2:.2f}, "
                 f"df = {dof}, p ~ {p:.3e}")
    lines.append(f"- Mean pairwise Jensen-Shannon Divergence (base 2): "
                 f"{mean_jsd:.4f} (range [0, 1])")
    lines.append("")
    lines.append("### Pairwise JSD")
    for i, j, v in pair_jsd:
        lines.append(f"  - JSD(client {i}, client {j}) = {v:.4f}")

    out_text = "\n".join(lines)
    print(out_text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out_text)
        print(f"\n[md] saved -> {args.out}")


if __name__ == "__main__":
    main()
