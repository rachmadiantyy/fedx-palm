"""
Aggregate all experiment results into one master table + Bab 4 artefacts.

Reads:
  thesis_rebuild/runs/
    b1_centralized/results.csv                 (Ultralytics native format)
    b2_fl_K{K}_seed{N}/rounds.csv              (custom FL loop format)
    e1_fl_K{K}_sigma{S}_seed{N}/rounds.csv
    e2_fl_K{K}_sigma{S}_seed{N}/rounds.csv
    b2_fl_summary_seed{N}.csv                  (per-script summaries)
    e1_fl_full_grid_seed{N}.csv
    e2_fl_full_grid_seed{N}.csv

Writes:
  thesis_rebuild/tables/runs_master.csv        (one row per run)
  thesis_rebuild/tables/e1_grid_map50.md       (markdown table for thesis)
  thesis_rebuild/tables/e2_grid_map50.md
  thesis_rebuild/figures/privacy_utility_e1.png
  thesis_rebuild/figures/privacy_utility_e2.png
  thesis_rebuild/figures/K_curve_e1.png
  thesis_rebuild/figures/K_curve_e2.png

Use:
  python thesis_rebuild/scripts/aggregate_results.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "thesis_rebuild" / "runs"
TABLES_DIR = REPO_ROOT / "thesis_rebuild" / "tables"
FIGURES_DIR = REPO_ROOT / "thesis_rebuild" / "figures"

# Operational thresholds (Bab 3.8 / 4.0)
THRESH = [
    (0.05, "collapsed"),
    (0.70, "degraded"),
    (0.90, "acceptable"),
    (float("inf"), "excellent"),
]


def classify(map50: float) -> str:
    for cap, label in THRESH:
        if map50 < cap:
            return label
    return "excellent"


# ---------- readers ----------------------------------------------------------

def read_b1(runs_dir: Path) -> Optional[dict]:
    """B1 uses Ultralytics' results.csv — read the LAST epoch's metrics."""
    csv_path = runs_dir / "b1_centralized" / "results.csv"
    if not csv_path.exists():
        return None
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    last = rows[-1]

    def pick(*candidates: str) -> float:
        for c in candidates:
            for k in last:
                if k.strip() == c:
                    return float(last[k])
        return float("nan")

    return {
        "exp": "B1",
        "K": 1,
        "sigma": 0.0,
        "seed": 42,
        "rounds": int(float(last.get("epoch", len(rows)))),
        "best_mAP50": pick("metrics/mAP50(B)", "metrics/mAP50"),
        "best_mAP5095": pick("metrics/mAP50-95(B)", "metrics/mAP50-95"),
        "precision": pick("metrics/precision(B)", "metrics/precision"),
        "recall": pick("metrics/recall(B)", "metrics/recall"),
        "final_epsilon": 0.0,
        "freeze_backbone": False,
        "ckpt": str(runs_dir / "b1_centralized" / "weights" / "best.pt"),
    }


# Run dir name patterns (incremental crash-safe rounds.csv lives inside each)
B2_DIR_RE = re.compile(r"^b2_fl_K(?P<K>\d+)_seed(?P<seed>\d+)$")
E_DIR_RE = re.compile(
    r"^(?P<exp>e[12])_fl_K(?P<K>\d+)_sigma(?P<sigma>[0-9.]+)_seed(?P<seed>\d+)$"
)


def read_round_csv(run_dir: Path) -> Optional[dict]:
    """Return the row of the BEST mAP@0.5 in rounds.csv, plus final eps."""
    csv_path = run_dir / "rounds.csv"
    if not csv_path.exists():
        return None
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    rows_f = [
        {**r,
         "round": int(r["round"]),
         "epsilon": float(r["epsilon"]),
         "mAP50": float(r["mAP50"]),
         "mAP5095": float(r["mAP5095"]),
         "precision": float(r.get("precision", "nan") or "nan"),
         "recall": float(r.get("recall", "nan") or "nan")}
        for r in rows
    ]
    best = max(rows_f, key=lambda r: r["mAP50"])
    last_eps = rows_f[-1]["epsilon"]
    return {
        "rounds": len(rows_f),
        "best_mAP50": best["mAP50"],
        "best_mAP5095": best["mAP5095"],
        "precision": best["precision"],
        "recall": best["recall"],
        "final_epsilon": last_eps,
    }


def read_fl_runs(runs_dir: Path) -> list[dict]:
    """Walk runs/ and collect all B2/E1/E2 run dirs."""
    out: list[dict] = []
    for d in sorted(runs_dir.iterdir()) if runs_dir.exists() else []:
        if not d.is_dir():
            continue
        m = B2_DIR_RE.match(d.name)
        if m:
            res = read_round_csv(d)
            if res is None:
                continue
            out.append({
                "exp": "B2", "K": int(m["K"]), "sigma": 0.0,
                "seed": int(m["seed"]), "freeze_backbone": False,
                "ckpt": str(d / "best.pt"), **res,
            })
            continue
        m = E_DIR_RE.match(d.name)
        if m:
            res = read_round_csv(d)
            if res is None:
                continue
            out.append({
                "exp": m["exp"].upper(), "K": int(m["K"]),
                "sigma": float(m["sigma"]), "seed": int(m["seed"]),
                "freeze_backbone": m["exp"] == "e2",
                "ckpt": str(d / "best.pt"), **res,
            })
    return out


# ---------- writers ----------------------------------------------------------

MASTER_COLS = ["exp", "K", "sigma", "seed", "freeze_backbone", "rounds",
               "best_mAP50", "best_mAP5095", "precision", "recall",
               "final_epsilon", "regime", "ckpt"]


def write_master(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r["regime"] = classify(r["best_mAP50"])
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MASTER_COLS})
    print(f"[master] {len(rows)} runs -> {out}")


def grid_md_table(rows: list[dict], exp: str, out: Path) -> None:
    """Markdown table for Bab 4: rows = K, cols = sigma, cell = best mAP@0.5."""
    sub = [r for r in rows if r["exp"] == exp]
    if not sub:
        print(f"[md] no rows for {exp}, skip")
        return
    Ks = sorted({r["K"] for r in sub})
    sigmas = sorted({r["sigma"] for r in sub})
    # Average over seeds if multiple
    cell: dict[tuple[int, float], list[float]] = {}
    for r in sub:
        cell.setdefault((r["K"], r["sigma"]), []).append(r["best_mAP50"])

    lines: list[str] = []
    lines.append(f"# {exp} — best mAP@0.5 across (K, sigma)")
    lines.append("")
    header = "| K \\ sigma | " + " | ".join(f"{s}" for s in sigmas) + " |"
    sep = "|---|" + "|".join("---" for _ in sigmas) + "|"
    lines += [header, sep]
    for K in Ks:
        cells = []
        for s in sigmas:
            vals = cell.get((K, s))
            if not vals:
                cells.append("—")
            elif len(vals) == 1:
                cells.append(f"{vals[0]:.3f}")
            else:
                mean = sum(vals) / len(vals)
                spread = (max(vals) - min(vals)) / 2
                cells.append(f"{mean:.3f}±{spread:.3f}")
        lines.append(f"| {K} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Cells with multiple seeds show mean±half-range.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"[md] {exp} grid -> {out}")


def make_plots(rows: list[dict], exp: str) -> None:
    """Generate privacy-utility (mAP vs eps) and K-curve plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; skipping plots")
        return
    sub = [r for r in rows if r["exp"] == exp]
    if not sub:
        print(f"[plot] no rows for {exp}, skip")
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Privacy-utility: x = eps, y = mAP50, line per K
    fig, ax = plt.subplots(figsize=(7, 4.5))
    Ks = sorted({r["K"] for r in sub})
    for K in Ks:
        pts = sorted(
            [(r["final_epsilon"], r["best_mAP50"]) for r in sub if r["K"] == K]
        )
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=f"K={K}")
    ax.axhline(0.05, color="grey", linestyle=":", linewidth=0.8, label="collapse")
    ax.set_xlabel(r"$\varepsilon$ (privacy budget, $\delta=10^{-5}$)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title(f"{exp}: privacy–utility trade-off")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = FIGURES_DIR / f"privacy_utility_{exp.lower()}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] {p}")

    # K-curve: x = K, y = mAP50, line per sigma
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sigmas = sorted({r["sigma"] for r in sub})
    for s in sigmas:
        pts = sorted(
            [(r["K"], r["best_mAP50"]) for r in sub if r["sigma"] == s]
        )
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="s", label=f"σ={s}")
    ax.set_xlabel("K (number of clients)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title(f"{exp}: K-curve (H2-K: K↑ → utility↓ for per-sample DP-SGD)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = FIGURES_DIR / f"K_curve_{exp.lower()}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] {p}")


# ---------- main -------------------------------------------------------------

def main() -> None:
    if not RUNS_DIR.exists():
        sys.exit(f"runs dir not found: {RUNS_DIR}")

    rows: list[dict] = []
    b1 = read_b1(RUNS_DIR)
    if b1:
        rows.append(b1)
        print(f"[B1] mAP@0.5={b1['best_mAP50']:.4f} (upper bound)")
    else:
        print("[B1] not found yet")

    fl = read_fl_runs(RUNS_DIR)
    rows.extend(fl)
    counts: dict[str, int] = {}
    for r in fl:
        counts[r["exp"]] = counts.get(r["exp"], 0) + 1
    print(f"[FL] {counts}")

    if not rows:
        sys.exit("No results found. Has the grid run yet?")

    write_master(rows, TABLES_DIR / "runs_master.csv")
    for exp in ("B2", "E1", "E2"):
        grid_md_table(rows, exp, TABLES_DIR / f"{exp.lower()}_grid_map50.md")
    for exp in ("E1", "E2"):
        make_plots(rows, exp)

    # Quick summary
    print("\n=== SUMMARY (best mAP@0.5 per experiment) ===")
    for exp in ("B1", "B2", "E1", "E2"):
        sub = [r for r in rows if r["exp"] == exp]
        if not sub:
            continue
        best = max(sub, key=lambda r: r["best_mAP50"])
        print(f"  {exp}: best {best['best_mAP50']:.3f} "
              f"@ K={best['K']}, σ={best['sigma']}, ε={best['final_epsilon']:.2f} "
              f"({classify(best['best_mAP50'])})")


if __name__ == "__main__":
    main()
