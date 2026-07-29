#!/usr/bin/env python3
"""ONE-TIME held-out TEST-split evaluation of B2's three validation-selected
checkpoints (seeds 42/123/2026, K=4, FedAvg, no DP) -- fills the manuscript's
Table 8 (matched held-out B1/B2 comparison).

Checkpoint identity (confirmed from docs/thesis/manuscript/b2_k4_seed{S}_
history.json's own recorded "checkpoint" paths -- NOT assumed):
  runs/b2_federated/k4_seed{42,123,2026}_leakagefree_stageA_seedfix/best_global.pt
best_global.pt is saved by fedxpalm.federated.server.run_federated_training
whenever a new best VALIDATION round is found, so it already IS each seed's
best round (9 / 11 / 40 respectively, matching the manuscript's Table 4.2) --
no round-number bookkeeping is done in this script beyond reporting it back.

READ-ONLY: calls fedxpalm.eval.detection_metrics.evaluate_detector() --
YOLO(weights_path) + model.val(...), no optimizer, no backward(), no model
update. Checkpoints are only opened for reading.

Runs EXACTLY ONCE per seed: refuses to overwrite an existing output file.

Usage:
    python scripts/46_eval_b2_holdout_test.py --device 0

Outputs (refuses to overwrite):
    results/b2_holdout_test/k4_seed42.json
    results/b2_holdout_test/k4_seed123.json
    results/b2_holdout_test/k4_seed2026.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

import fedxpalm  # noqa: E402,F401 (applies the GroupNorm-safe `fuse()` patch)
from fedxpalm.eval.detection_metrics import evaluate_detector  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
IMGSZ = 960  # matches B1/B2/E1/E2 -- fl_config.yaml model.imgsz, applied globally
CLASS_NAMES = ["Abnormal", "Empty Bunch", "Overripe", "Ripe", "Underripe", "Unripe"]

SEEDS = [42, 123, 2026]
CHECKPOINT_TEMPLATE = "runs/b2_federated/k4_seed{seed}_leakagefree_stageA_seedfix/best_global.pt"
OUT_DIR = REPO_ROOT / "results/b2_holdout_test"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    out_paths = {seed: OUT_DIR / f"k4_seed{seed}.json" for seed in SEEDS}
    for seed, out_path in out_paths.items():
        if out_path.exists():
            print(f"FAIL: {out_path} already exists -- this script runs the held-out test "
                  f"EXACTLY ONCE per seed and refuses to overwrite or re-run. Move/rename it "
                  f"first if you have a specific, deliberate reason to redo this")
            return 1

    ckpt_paths = {}
    for seed in SEEDS:
        p = REPO_ROOT / CHECKPOINT_TEMPLATE.format(seed=seed)
        if not p.exists():
            print(f"FAIL: checkpoint not found for seed {seed}: {p}")
            return 1
        ckpt_paths[seed] = p

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    splits_dir = Path(ds_cfg["output_dir"])
    data_yaml = str(splits_dir / "data.yaml")

    all_results = {}
    for seed in SEEDS:
        ckpt_path = ckpt_paths[seed]
        ckpt_sha256 = sha256_of(ckpt_path)
        print(f"\n=== B2 seed={seed}: evaluating {ckpt_path} (sha256={ckpt_sha256[:16]}...) on "
              f"HELD-OUT TEST split, imgsz={IMGSZ}, device={args.device} ===")
        metrics = evaluate_detector(str(ckpt_path), data_yaml, split="test",
                                    imgsz=IMGSZ, device=args.device)

        missing_classes = sorted(set(CLASS_NAMES) - set(metrics["per_class"]))
        unexpected_classes = sorted(set(metrics["per_class"]) - set(CLASS_NAMES))
        if missing_classes or unexpected_classes:
            print(f"[!] class-name mismatch for seed {seed}: missing={missing_classes} "
                  f"unexpected={unexpected_classes}")

        print(f"overall: mAP50={metrics['map50']:.4f}  mAP50-95={metrics['map50_95']:.4f}  "
              f"precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  "
              f"n_boxes_total={metrics['n_boxes_total']}")
        print(f"{'class':>14}{'AP50':>9}{'AP50-95':>10}{'precision':>11}{'recall':>9}{'n_inst':>8}")
        for cname in CLASS_NAMES:
            pc = metrics["per_class"].get(cname)
            if pc is None:
                print(f"{cname:>14}  (no instances / not reported)")
                continue
            print(f"{cname:>14}{pc['ap50']:>9.4f}{pc['ap50_95']:>10.4f}"
                  f"{pc['precision']:>11.4f}{pc['recall']:>9.4f}{pc['n_instances']:>8}")

        record = {
            "diagnostic": "b2_holdout_test_eval",
            "note": "ONE-TIME held-out test evaluation of B2's validation-selected best_global.pt "
                    "(FedAvg, K=4, no DP). Checkpoint choice was made ENTIRELY on the validation "
                    "split during training -- this script does not select, tune, or retrain.",
            "experiment": "B2", "seed": seed, "num_clients": 4,
            "checkpoint_path": str(ckpt_path.relative_to(REPO_ROOT)),
            "checkpoint_sha256": ckpt_sha256,
            "eval_config": {"split": "test", "imgsz": IMGSZ, "device": args.device},
            "n_boxes_total": metrics["n_boxes_total"],
            "overall": {
                "map50": metrics["map50"], "map50_95": metrics["map50_95"],
                "precision": metrics["precision"], "recall": metrics["recall"],
            },
            "per_class": metrics["per_class"],
            "class_name_audit": {"missing": missing_classes, "unexpected": unexpected_classes},
            "no_model_update": True, "no_optimizer_step": True,
            "no_test_based_model_selection": True,
        }
        all_results[seed] = record

        out_path = out_paths[seed]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"Saved {out_path}")

    print("\n=== B2 held-out test evaluation complete for all 3 seeds ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
