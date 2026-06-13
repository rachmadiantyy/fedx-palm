"""
E2-FL: Federated Partial DP-SGD (backbone frozen, head only).

Backbone model.0-9 frozen (pretrained, no DP). Only head ~0.2M params
trainable + DP-protected. Should dominate E1-FL at low epsilon.

Full grid: K {2,4,8,16} x sigma {0.5,1.0,1.5,2.0,3.0} = 20 runs.

Run:
    python thesis_rebuild/scripts/train_e2_fl_dp_sgd_partial.py --K 4 --sigma 1.0
    python thesis_rebuild/scripts/train_e2_fl_dp_sgd_partial.py --full-grid
"""
import argparse
import csv
import itertools
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from thesis_rebuild.scripts.utils.fl_dp_loop import (  # noqa: E402
    FedDPConfig,
    run_federated,
)


K_GRID = [2, 4, 8, 16]
SIGMA_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2-FL Partial DP-SGD federated")
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--weights", type=str, default="yolo11n.pt")
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--full-grid", action="store_true")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--project", type=str, default="thesis_rebuild/runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.full_grid:
        combos = list(itertools.product(K_GRID, SIGMA_GRID))
    elif args.K is not None and args.sigma is not None:
        combos = [(args.K, args.sigma)]
    else:
        sys.exit("Specify --K K --sigma S, or --full-grid")

    results = []
    for K, sigma in combos:
        cfg = FedDPConfig(
            K=K,
            clients_root=f"{args.data_root}/clients_K{K}",
            global_val_yaml=f"{args.data_root}/global_val.yaml",
            weights=args.weights,
            imgsz=args.imgsz,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch,
            lr0=args.lr0,
            noise_multiplier=sigma,
            max_grad_norm=args.max_grad_norm,
            use_dp=True,
            freeze_backbone=True,          # E2 = partial
            seed=args.seed,
            device=args.device,
            project=args.project,
            name=f"e2_fl_K{K}_sigma{sigma}",
        )
        results.append(run_federated(cfg))

    out = Path(args.project) / "e2_fl_full_grid.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["K", "sigma", "freeze_backbone",
                                          "rounds", "best_mAP50",
                                          "final_epsilon", "ckpt"])
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"\nE2-FL summary -> {out}")


if __name__ == "__main__":
    main()
