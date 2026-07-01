"""
B2: Federated YOLOv11n-GN baseline (no DP).

Federated FedAvg without any DP noise. Measures the 'FL cost' relative
to B1 (centralized): how much utility is lost just by federating?

Sweeps K in {2, 4, 8, 12, 16} per user's full-grid decision.

Run single K:
    python thesis_rebuild/scripts/train_b2_fl.py --K 4 --rounds 5

Run all K:
    python thesis_rebuild/scripts/train_b2_fl.py --all-K
"""
import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from thesis_rebuild.scripts.utils.fl_dp_loop import (  # noqa: E402
    FedDPConfig,
    run_federated,
)


K_GRID = [2, 4, 8, 12, 16]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B2 Federated baseline (no DP)")
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--weights", type=str, default="yolo11n.pt")
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--all-K", action="store_true")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--project", type=str, default="thesis_rebuild/runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.all_K:
        Ks = K_GRID
    elif args.K is not None:
        Ks = [args.K]
    else:
        sys.exit("Specify --K X or --all-K")

    results = []
    for K in Ks:
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
            noise_multiplier=0.0,
            use_dp=False,            # B2 = no DP
            freeze_backbone=False,
            seed=args.seed,
            device=args.device,
            project=args.project,
            name=f"b2_fl_K{K}_seed{args.seed}",
        )
        results.append(run_federated(cfg))

    out = Path(args.project) / f"b2_fl_summary_seed{args.seed}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["K", "sigma", "freeze_backbone",
                                          "rounds", "best_mAP50",
                                          "final_epsilon", "ckpt", "seed"])
        w.writeheader()
        for r in results:
            row = {k: r[k] for k in w.fieldnames if k != "seed"}
            row["seed"] = args.seed
            w.writerow(row)
    print(f"\nSummary -> {out}")


if __name__ == "__main__":
    main()
