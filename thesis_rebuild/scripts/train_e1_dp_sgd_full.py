"""
E1: Full DP-SGD on YOLOv11n-GN.

All 2.6M parameters are trainable and protected by per-sample
gradient clipping + Gaussian noise via Opacus.

Sweeps σ ∈ {0.5, 1.0, 1.5, 2.0, 3.0} to produce the privacy-utility
curve for Bab 4.3 of the thesis.

Run (single σ):
    python thesis_rebuild/scripts/train_e1_dp_sgd_full.py \\
        --sigma 1.0 --epochs 50

Run (full sweep):
    python thesis_rebuild/scripts/train_e1_dp_sgd_full.py --sweep

Note: For Day 2 this is CENTRALIZED DP-SGD (not federated). Day 3
adds federation. Reason: validate the loop end-to-end before adding
FL plumbing on top.
"""
import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from thesis_rebuild.scripts.utils.yolo_dp_loop import (  # noqa: E402
    DPSGDConfig,
    train_one_run,
)


SIGMA_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1 Full DP-SGD sweep")
    p.add_argument("--data", type=str, default="configs/data_sample.yaml")
    p.add_argument("--weights", type=str, default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=None,
                   help="Run a single σ. If omitted with --sweep, runs grid.")
    p.add_argument("--sweep", action="store_true",
                   help=f"Sweep σ over {SIGMA_GRID}")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--project", type=str, default="thesis_rebuild/runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.sweep:
        sigmas = SIGMA_GRID
    elif args.sigma is not None:
        sigmas = [args.sigma]
    else:
        raise SystemExit("Specify --sigma X or --sweep")

    results = []
    for sigma in sigmas:
        cfg = DPSGDConfig(
            data_yaml=args.data,
            weights=args.weights,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            lr0=args.lr0,
            noise_multiplier=sigma,
            max_grad_norm=args.max_grad_norm,
            freeze_backbone=False,           # E1 = full
            seed=args.seed,
            device=args.device,
            project=args.project,
            name=f"e1_full_sigma{sigma}",
        )
        result = train_one_run(cfg)
        results.append(result)

    # Write CSV summary
    summary_path = Path(args.project) / "e1_full_sweep.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sigma", "C", "freeze_backbone", "n_trainable",
                        "final_loss", "final_epsilon", "best_mAP50", "ckpt"],
        )
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in writer.fieldnames}
            writer.writerow(row)
    print(f"\nSummary CSV: {summary_path}")


if __name__ == "__main__":
    main()
