"""
E2: Partial DP-SGD on YOLOv11n-GN — backbone frozen, head-only training.

Only the detection head (~0.2M params) is trainable and DP-protected.
The backbone (model.0 - model.9) is loaded from pretrained weights
and kept frozen during DP-SGD.

Why this should win at low ε (high privacy):
  Per-sample noise scales with √(num_trainable_params). Freezing the
  backbone reduces noise impact by ~10x. See Tramer & Boneh (2021),
  "Differentially Private Learning Needs Better Features", ICLR.

Sweep σ identical to E1 → side-by-side comparison in Bab 4.4.

Run:
    python thesis_rebuild/scripts/train_e2_dp_sgd_partial.py --sweep
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
    p = argparse.ArgumentParser(description="E2 Partial DP-SGD (head only)")
    p.add_argument("--data", type=str, default="configs/data_sample.yaml")
    p.add_argument("--weights", type=str, default="yolo11n.pt",
                   help="Pretrained init for backbone (kept frozen)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--sweep", action="store_true")
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
            freeze_backbone=True,            # E2 = partial
            seed=args.seed,
            device=args.device,
            project=args.project,
            name=f"e2_partial_sigma{sigma}",
        )
        result = train_one_run(cfg)
        results.append(result)

    summary_path = Path(args.project) / "e2_partial_sweep.csv"
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
