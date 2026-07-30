#!/usr/bin/env python3
"""ONE-TIME held-out TEST-split evaluation of B2's locked seed-42 checkpoint
(see scripts/47_lock_b2_checkpoint.py, which must have already produced
results/final_b2/b2_checkpoint_manifest.json -- this script refuses to run
without it). Uses the IDENTICAL eval configuration as E1/E2's
scripts/45_eval_final_dp_test.py (imgsz=960, split=test) so the three are
directly comparable.

READ-ONLY: evaluate_detector() only does YOLO(weights_path) + model.val(...)
-- no optimizer, no backward(), no model update. Re-hashes the checkpoint
first and hard-stops if it no longer matches the locked manifest.

Runs EXACTLY ONCE: refuses to overwrite its output.

Usage:
    python scripts/48_eval_b2_final_test.py --device 0

Output (refuses to overwrite):
    results/final_b2/test_b2_seed42_best_r9.json
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
MANIFEST_PATH = REPO_ROOT / "results/final_b2/b2_checkpoint_manifest.json"
OUT_JSON = REPO_ROOT / "results/final_b2/test_b2_seed42_best_r9.json"
IMGSZ = 960  # matches scripts/45's E1/E2 held-out test config exactly
EXPECTED_TEST_IMAGES = 1051
CLASS_NAMES = ["Abnormal", "Empty Bunch", "Overripe", "Ripe", "Underripe", "Unripe"]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        print(f"FAIL: {MANIFEST_PATH} not found -- run scripts/47_lock_b2_checkpoint.py first")
        return 1
    if OUT_JSON.exists():
        print(f"FAIL: {OUT_JSON} already exists -- this script runs the held-out test EXACTLY "
              f"ONCE and refuses to overwrite or re-run")
        return 1

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    ckpt_path = REPO_ROOT / manifest["checkpoint_path"]
    if not ckpt_path.exists():
        print(f"FAIL: {ckpt_path} not found")
        return 1
    live_sha256 = sha256_of(ckpt_path)
    if live_sha256 != manifest["checkpoint_sha256"]:
        print(f"FAIL: checkpoint sha256 changed since the lock manifest was built -- "
              f"live={live_sha256} locked={manifest['checkpoint_sha256']}. Refusing to evaluate "
              f"a checkpoint that no longer matches the locked, audited artifact")
        return 1

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    splits_dir = Path(ds_cfg["output_dir"])
    data_yaml = str(splits_dir / "data.yaml")

    test_images_dir = splits_dir / "test" / "images"
    if not test_images_dir.exists():
        print(f"FAIL: {test_images_dir} not found")
        return 1
    n_test_images_on_disk = sum(1 for p in test_images_dir.iterdir() if p.is_file())
    if n_test_images_on_disk != EXPECTED_TEST_IMAGES:
        print(f"FAIL: test image count on disk = {n_test_images_on_disk}, "
              f"expected {EXPECTED_TEST_IMAGES}")
        return 1

    print(f"=== B2 seed=42: evaluating {ckpt_path} (sha256={live_sha256[:16]}...) on "
          f"HELD-OUT TEST split, imgsz={IMGSZ}, device={args.device} ===")
    metrics = evaluate_detector(str(ckpt_path), data_yaml, split="test", imgsz=IMGSZ, device=args.device)

    missing_classes = sorted(set(CLASS_NAMES) - set(metrics["per_class"]))
    unexpected_classes = sorted(set(metrics["per_class"]) - set(CLASS_NAMES))
    if missing_classes or unexpected_classes:
        print(f"[!] class-name mismatch: missing={missing_classes} unexpected={unexpected_classes}")

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
        "diagnostic": "b2_seed42_holdout_test_eval",
        "note": "ONE-TIME held-out test evaluation of B2 seed42's manifest-locked, "
                "validation-selected checkpoint (round 9). No model update, no optimizer, "
                "no test-based model/checkpoint selection.",
        "experiment": "B2", "seed": 42, "selected_round_number": manifest["selected_round_number"],
        "checkpoint_path": manifest["checkpoint_path"], "checkpoint_sha256": live_sha256,
        "lock_manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "eval_config": {"split": "test", "imgsz": IMGSZ, "device": args.device},
        "dataset_test_split_path": str(test_images_dir.resolve().relative_to(REPO_ROOT)),
        "n_test_images_on_disk": n_test_images_on_disk,
        "n_test_images_expected": EXPECTED_TEST_IMAGES,
        "n_boxes_total": metrics["n_boxes_total"],
        "overall": {
            "map50": metrics["map50"], "map50_95": metrics["map50_95"],
            "precision": metrics["precision"], "recall": metrics["recall"],
        },
        "per_class": metrics["per_class"],
        "class_name_audit": {"missing": missing_classes, "unexpected": unexpected_classes},
        "no_model_update": True, "no_optimizer_step": True, "no_test_based_model_selection": True,
        "val_best_map50_for_reference": manifest["best_validation_map50"],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nSaved {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
