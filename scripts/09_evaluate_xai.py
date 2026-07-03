#!/usr/bin/env python3
"""Grad-CAM++ faithfulness evaluation (Average Drop, Focus Retention Rate)
on a chosen checkpoint -- run this on whichever run you designate as the
"operational" model (e.g. B2's best K), mirrors thesis Table 4.9/4.10."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

import fedxpalm  # noqa: E402,F401 (applies the GroupNorm-safe `fuse()` patch)
from fedxpalm.xai.evaluate import evaluate_faithfulness  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--tag", required=True, help="label for the output file, e.g. b2_k4")
    parser.add_argument("--target-layer-idx", type=int, default=22)
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--save-overlays", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)

    splits_dir = Path(ds_cfg["output_dir"])
    yolo = YOLO(args.weights)

    results = evaluate_faithfulness(
        yolo.model,
        images_dir=str(splits_dir / "test" / "images"),
        labels_dir=str(splits_dir / "test" / "labels"),
        class_names=ds_cfg["names"],
        imgsz=fl_cfg["model"]["imgsz"],
        target_layer_idx=args.target_layer_idx,
        top_fraction=args.top_fraction,
        max_images=args.max_images,
        save_overlays_dir=f"results/xai_overlays_{args.tag}" if args.save_overlays else None,
    )

    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / f"xai_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    for name, m in results.items():
        print(f"{name:>15}: n={m['n']:4d}  AD={m['average_drop']:.3f}  FRR={m['focus_retention_rate']:.3f}")
    print(f"Saved {out_path}")
