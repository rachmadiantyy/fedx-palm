#!/usr/bin/env python3
"""ONE-TIME held-out TEST-split evaluation for the locked E1_full / E2_partial_P2
canonical 20-round checkpoints (see scripts/44_audit_and_manifest_final_dp.py,
which must have already produced results/final_dp_canonical/final_artifact_manifest.json
-- this script refuses to run without it).

READ-ONLY: calls fedxpalm.eval.detection_metrics.evaluate_detector(), which
only does `YOLO(weights_path)` + `model.val(...)` -- no optimizer, no
backward(), no privacy accountant, no training step, no model update of any
kind. The checkpoint files themselves are never opened for writing.

Before evaluating, re-hashes both checkpoints and compares against the
sha256 already locked in final_artifact_manifest.json -- hard-stops if
either checkpoint has changed since the manifest was built (this is the
mechanism that makes "no test-based model selection happened after the
manifest was locked" a verified fact, not an assumption).

This script is meant to be run EXACTLY ONCE per checkpoint. It refuses to
overwrite either output path, and does not accept a --confidence/--iou/
--imgsz override for one model but not the other -- both are evaluated with
IDENTICAL settings (imgsz=960, device, batch=default) by construction, not
by convention.

Usage:
    python scripts/45_eval_final_dp_test.py --device 0

Outputs (refuses to overwrite):
    results/final_dp_canonical/test_e1_full_seed42_best_r19.json
    results/final_dp_canonical/test_e2_partial_p2_seed42_best_r19.json
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
MANIFEST_PATH = REPO_ROOT / "results/final_dp_canonical/final_artifact_manifest.json"
IMGSZ = 960
EXPECTED_TEST_IMAGES = 1051
CLASS_NAMES = ["Abnormal", "Empty Bunch", "Overripe", "Ripe", "Underripe", "Unripe"]

OUTPUTS = {
    "E1_full": REPO_ROOT / "results/final_dp_canonical/test_e1_full_seed42_best_r19.json",
    "E2_partial_P2": REPO_ROOT / "results/final_dp_canonical/test_e2_partial_p2_seed42_best_r19.json",
}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        print(f"FAIL: {MANIFEST_PATH} not found -- run scripts/44_audit_and_manifest_final_dp.py "
              f"first. Held-out test must only run against a manifest-locked checkpoint")
        return 1
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    for variant, out_path in OUTPUTS.items():
        if out_path.exists():
            print(f"FAIL: {out_path} already exists -- this script runs the held-out test EXACTLY "
                  f"ONCE per checkpoint and refuses to overwrite or re-run. Move/rename it first "
                  f"if you have a specific, deliberate reason to redo this")
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

    results = {}
    for variant in ("E1_full", "E2_partial_P2"):
        exp = manifest["experiments"][variant]
        ckpt_path = REPO_ROOT / exp["selected_checkpoint_path"]
        if not ckpt_path.exists():
            print(f"FAIL: {ckpt_path} not found")
            return 1
        live_sha256 = sha256_of(ckpt_path)
        locked_sha256 = exp["selected_checkpoint_sha256"]
        if live_sha256 != locked_sha256:
            print(f"FAIL: {variant} checkpoint sha256 changed since the manifest was built -- "
                  f"live={live_sha256} locked={locked_sha256}. Refusing to evaluate a checkpoint "
                  f"that does not match the locked, audited artifact")
            return 1

        print(f"\n=== {variant}: evaluating {ckpt_path} (sha256={live_sha256[:16]}...) on "
              f"HELD-OUT TEST split, imgsz={IMGSZ}, device={args.device} ===")
        metrics = evaluate_detector(str(ckpt_path), data_yaml, split="test",
                                    imgsz=IMGSZ, device=args.device)

        missing_classes = sorted(set(CLASS_NAMES) - set(metrics["per_class"]))
        unexpected_classes = sorted(set(metrics["per_class"]) - set(CLASS_NAMES))
        if missing_classes or unexpected_classes:
            print(f"[!] class-name mismatch for {variant}: missing={missing_classes} "
                  f"unexpected={unexpected_classes} -- reported per_class below may not cover "
                  f"all 6 expected classes")

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
            "diagnostic": "final_dp_holdout_test_eval",
            "note": "ONE-TIME held-out test evaluation of a manifest-locked, validation-selected "
                    "checkpoint. No model update, no optimizer, no privacy-accountant step, no "
                    "test-based model selection: the checkpoint was chosen ENTIRELY from the "
                    "validation split during training (see the source result JSON's "
                    "best_round_number), before this script ever ran.",
            "variant": variant,
            "checkpoint_path": exp["selected_checkpoint_path"],
            "checkpoint_sha256": live_sha256,
            "selected_round_number": exp["selected_round_number"],
            "source_result_json": exp["result_json_path"],
            "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
            "protocol_fingerprint": exp["protocol_fingerprint"],
            "run_fingerprint": exp["run_fingerprint"],
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
            "no_model_update": True, "no_optimizer_step": True,
            "no_privacy_accountant_step": True, "no_test_based_model_selection": True,
            "val_best_map50_for_reference": exp["best_validation_map50"],
        }
        results[variant] = record

        out_path = OUTPUTS[variant]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"Saved {out_path}")

    print("\n=== held-out test evaluation complete for both variants ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
