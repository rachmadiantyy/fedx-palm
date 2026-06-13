"""
B1: Centralized YOLOv11n-GN baseline (no DP, no FL).

This is the UPPER BOUND for utility — it sees all data, no FL cost,
no DP noise. Provides the reference point for all FL/DP comparisons.

Architecture: YOLOv11n with all BatchNorm replaced by GroupNorm.
We use GN here (not BN) for fair comparison with E1/E2 which require
GN for Opacus compatibility.

Run:
    python thesis_rebuild/scripts/train_b1_centralized.py \\
        --data configs/data_sample.yaml \\
        --epochs 50 \\
        --batch 16 \\
        --imgsz 640 \\
        --project thesis_rebuild/runs \\
        --name b1_centralized

Expected runtime: ~2 hours on RTX 4080.
Expected outcome: mAP@0.5 ≥ 0.95 (matches ablation C1b ~0.977).
"""
import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

# Make repo root importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from thesis_rebuild.scripts.utils.gn_convert import (  # noqa: E402
    count_bn_layers,
    replace_bn_with_gn,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B1 Centralized YOLOv11n-GN")
    p.add_argument("--data", type=str, default="data/resplit/data.yaml")
    p.add_argument("--weights", type=str, default="yolo11n.pt",
                   help="Initial weights (will have BN converted to GN)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--optimizer", type=str, default="SGD")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--project", type=str, default="thesis_rebuild/runs")
    p.add_argument("--name", type=str, default="b1_centralized")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_groups", type=int, default=8,
                   help="GroupNorm groups (auto-reduces if needed)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("B1 — Centralized YOLOv11n-GN Baseline (no DP, no FL)")
    print("=" * 70)
    print(f"  Data:    {args.data}")
    print(f"  Weights: {args.weights}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Batch:   {args.batch}")
    print(f"  Image:   {args.imgsz}")
    print(f"  Seed:    {args.seed}")
    print()

    torch.manual_seed(args.seed)

    # Load YOLO and convert BN → GN BEFORE training so Ultralytics
    # Trainer trains the GN-version (not BN-version).
    yolo = YOLO(args.weights)
    n_bn_before, _ = count_bn_layers(yolo.model)
    print(f"Loaded {args.weights}: {n_bn_before} BatchNorm layers")

    n_converted = replace_bn_with_gn(yolo.model, num_groups=args.num_groups)
    n_bn_after, n_gn_after = count_bn_layers(yolo.model)
    print(f"Converted {n_converted} BN → GN")
    print(f"  Remaining BN: {n_bn_after} (should be 0)")
    print(f"  Total GN:     {n_gn_after}")
    assert n_bn_after == 0, "BN conversion incomplete"

    # Ultralytics' Trainer will pick up our modified yolo.model
    # (the model object is mutated in-place above)
    results = yolo.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        optimizer=args.optimizer,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        plots=True,
        save=True,
        exist_ok=False,
    )

    print()
    print("=" * 70)
    print("B1 COMPLETE")
    print("=" * 70)
    if hasattr(results, "box"):
        print(f"  mAP@0.5:      {float(results.box.map50):.4f}")
        print(f"  mAP@0.5:0.95: {float(results.box.map):.4f}")
        print(f"  Precision:    {float(results.box.mp):.4f}")
        print(f"  Recall:       {float(results.box.mr):.4f}")
    print()
    print(f"Best weights: {args.project}/{args.name}/weights/best.pt")
    print("Use this as init for B2 (federated) and as the centralized")
    print("upper bound in Bab 4.1 of the thesis.")


if __name__ == "__main__":
    main()
