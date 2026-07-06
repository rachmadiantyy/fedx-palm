#!/usr/bin/env python3
"""B1 -- centralized baseline: fine-tune on the *entire* pooled train split
(no federation, no DP). Upper bound the federated/private runs are compared
against (mirrors thesis Table 4.1)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

import fedxpalm  # noqa: E402,F401 (applies the GroupNorm-safe `fuse()` patch)
from fedxpalm.eval.detection_metrics import evaluate_detector  # noqa: E402
from fedxpalm.federated.trainer_utils import build_trainer_from_checkpoint  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0", help="'0' for first GPU, 'cpu' for CPU-only")
    parser.add_argument("--out-dir", default="runs/b1_centralized")
    parser.add_argument("--imgsz", type=int, default=None, help="override configs/fl_config.yaml model.imgsz (e.g. 960)")
    parser.add_argument("--batch", type=int, default=None, help="override baseline_centralized.batch_size")
    args = parser.parse_args()

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open("configs/fl_config.yaml") as f:
        fl_cfg = yaml.safe_load(f)

    hyp = fl_cfg["baseline_centralized"]
    data_yaml = str(Path(ds_cfg["output_dir"]) / "data.yaml")
    base_weights = "models/base_groupnorm.pt"
    imgsz = args.imgsz or fl_cfg["model"]["imgsz"]
    batch = args.batch or hyp["batch_size"]

    overrides = dict(
        data=data_yaml,
        model=base_weights,
        epochs=hyp["epochs"],
        patience=hyp.get("patience", 30),
        batch=batch,
        imgsz=imgsz,
        optimizer=hyp["optimizer"],
        lr0=hyp["lr0"],
        device=args.device,
        project=args.out_dir,
        name="train",
        exist_ok=True,
        plots=True,
    )
    # NOT YOLO(base_weights).train(...) -- see trainer_utils.py docstring for why
    # that silently reverts our GroupNorm conversion back to BatchNorm.
    trainer = build_trainer_from_checkpoint(base_weights, overrides)
    trainer.train()
    best_weights = str(Path(args.out_dir) / "train" / "weights" / "best.pt")

    metrics = evaluate_detector(best_weights, data_yaml, split="test", imgsz=imgsz, device=args.device)

    Path("results").mkdir(exist_ok=True)
    # Named after --out-dir (default "b1_centralized") so re-runs with a different
    # --out-dir (e.g. an imgsz=960 ablation) don't clobber a prior run's results file.
    results_path = f"results/{Path(args.out_dir).name}.json"
    with open(results_path, "w") as f:
        json.dump({"weights": best_weights, "imgsz": imgsz, "batch": batch, "metrics": metrics}, f, indent=2)
    print(f"B1 centralized: mAP@0.5={metrics['map50']:.3f}  mAP@0.5:0.95={metrics['map50_95']:.3f}")
    print(f"Saved {results_path}")
