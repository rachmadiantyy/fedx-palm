#!/usr/bin/env python3
"""VISUAL matched-sample panels: Original+GT | B2 (Non-DP) | E1 (Full DP) |
E2 (Partial DP), for a small, deterministic subset of the SAME matched-
sample manifest scripts/49/50 already used.

Selection rule (GT-driven, no CAM/prediction involved): the first 2 samples
per class in the manifest's own stored order (manifest.samples is already
sorted by class then by (image, box_index) -- see scripts/49). This is a
plain slice, not a re-shuffle, so it is trivially deterministic and never
looks at any model output to choose which samples to visualize.

Reuses scripts/50's checkpoint resolution (MODEL_SPECS, resolve_checkpoint
-- same hash-gated lock verification) and image loading (load_image_tensor)
by importing that script as a module (no CLI side effects: scripts/50's
work only happens inside `if __name__ == "__main__"`, never triggered by
import). The CAM/Average-Drop/Focus-Retention-Rate FORMULAS themselves are
called directly from fedxpalm.xai.{gradcam,metrics} -- the same single
source of truth scripts/50 uses, not reimplemented here.

Each panel's caption reports: model label, confidence for the GT class
(pre-occlusion), Average Drop, Focus Retention Rate, and a correct-
detection flag (IoU >= 0.5 vs GT, same threshold as scripts/50). Every
model panel uses the identical CAM colormap and overlay alpha
(fedxpalm.xai.gradcam.overlay_heatmap's own default, unchanged).

For Empty Bunch specifically, the caption additionally notes that E1/E2's
held-out test recall for this class is 0.0000 (scripts/45's locked result)
-- so a visualized CAM there must not be read as a successful detection.

Usage:
    python scripts/51_visualize_xai_matched.py --device 0

Outputs (refuses to overwrite):
    results/xai_matched/visual_panels/*.jpg
    results/xai_matched/visual_manifest.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fedxpalm  # noqa: E402,F401 (applies the GroupNorm-safe `fuse()` patch)
from fedxpalm.xai.gradcam import YOLOGradCAMPlusPlus, overlay_heatmap  # noqa: E402
from fedxpalm.xai.metrics import average_drop, class_confidence, focus_retention_rate, occlude_top_region  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
N_PER_CLASS = 2
CAPTION_H = 110  # px, black strip below each panel image
PANEL_W = 380    # px, each of the 4 panels is resized to this width for a uniform grid

# EMPTY BUNCH held-out test recall for E1/E2 (scripts/45's already-locked result) --
# a fixed, cited fact, not recomputed here
EMPTY_BUNCH_DP_TEST_RECALL = {"E1_full": 0.0000, "E2_partial_P2": 0.0000}

OUT_DIR = REPO_ROOT / "results/xai_matched/visual_panels"
OUT_MANIFEST = REPO_ROOT / "results/xai_matched/visual_manifest.json"

_spec = importlib.util.spec_from_file_location(
    "_xai_matched_quant", REPO_ROOT / "scripts" / "50_evaluate_xai_matched.py")
_m50 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m50)  # defines functions/constants only -- main() is never called


def draw_gt_panel(image_bgr: np.ndarray, gt_xyxy, class_name: str) -> np.ndarray:
    panel = image_bgr.copy()
    x1, y1, x2, y2 = (int(round(v)) for v in gt_xyxy)
    cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(panel, f"GT: {class_name}", (x1, max(0, y1 - 8)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return panel


def make_caption_strip(width: int, lines: list[str]) -> np.ndarray:
    strip = np.zeros((CAPTION_H, width, 3), dtype=np.uint8)
    y = 18
    for line in lines:
        cv2.putText(strip, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        y += 16
    return strip


def resize_keep_square(img: np.ndarray, width: int) -> np.ndarray:
    return cv2.resize(img, (width, width), interpolation=cv2.INTER_LINEAR)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--manifest", default=str(REPO_ROOT / "results/xai_matched/sample_manifest.json"))
    args = parser.parse_args()

    if OUT_MANIFEST.exists() or OUT_DIR.exists():
        print(f"FAIL: {OUT_MANIFEST} or {OUT_DIR} already exists -- refusing to overwrite")
        return 1

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"FAIL: {manifest_path} not found -- run scripts/49 first")
        return 1
    with open(manifest_path) as f:
        manifest = json.load(f)
    samples = manifest["samples"]

    # deterministic, GT-only selection: first N_PER_CLASS per class, in the
    # manifest's own stored (already class-sorted) order
    selected = []
    seen_per_class: dict[str, int] = {}
    for s in samples:
        c = s["class_name"]
        if seen_per_class.get(c, 0) < N_PER_CLASS:
            selected.append(s)
            seen_per_class[c] = seen_per_class.get(c, 0) + 1

    device = args.device if args.device in ("cpu",) else (
        f"cuda:{args.device}" if str(args.device).isdigit() else args.device)

    resolved = {}
    for label, spec in _m50.MODEL_SPECS.items():
        try:
            resolved[label] = _m50.resolve_checkpoint(label, spec)
        except (FileNotFoundError, ValueError) as e:
            print(f"FAIL: {e}")
            return 1

    from ultralytics import YOLO
    models = {label: YOLO(str(r["path"])).model.to(device) for label, r in resolved.items()}
    model_order = ["B2", "E1_full", "E2_partial_P2"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    visual_records = []

    for sample in selected:
        image_path = REPO_ROOT / sample["image_path"]
        img_bgr = cv2.imread(str(image_path))
        img_bgr = cv2.resize(img_bgr, (_m50.IMGSZ, _m50.IMGSZ), interpolation=cv2.INTER_LINEAR)
        gt_xyxy = tuple(sample["gt_box_xyxy_imgsz"])
        class_id, class_name = sample["class_id"], sample["class_name"]

        gt_panel = draw_gt_panel(img_bgr, gt_xyxy, class_name)
        gt_strip = make_caption_strip(_m50.IMGSZ, [f"Original + GT", f"class={class_name}",
                                                   f"sample_id={sample['sample_id']}"])
        panels = [np.vstack([resize_keep_square(gt_panel, PANEL_W),
                             resize_keep_square(gt_strip, PANEL_W)])]

        per_model_records = {}
        for label in model_order:
            detection_model = models[label]
            tensor, _ = _m50.load_image_tensor(image_path, _m50.IMGSZ, device)
            cam_engine = YOLOGradCAMPlusPlus(detection_model, target_layer_idx=_m50.TARGET_LAYER_IDX)
            try:
                cam, raw_score, anchor_idx, decoded_box_xywh = cam_engine.generate(
                    tensor.clone(), class_id, output_size=(_m50.IMGSZ, _m50.IMGSZ))
                y_c = float(torch.sigmoid(torch.tensor(raw_score)).item())
                occluded = occlude_top_region(tensor, cam, top_fraction=_m50.TOP_FRACTION)
                o_c = class_confidence(detection_model, occluded, class_id, anchor_idx)
                ad = average_drop(y_c, o_c)
                frr = focus_retention_rate(cam, gt_xyxy)
                decoded_xyxy = _m50.xywh_to_xyxy(decoded_box_xywh)
                iou = _m50.iou_xyxy(decoded_xyxy, gt_xyxy)
                correct_detection = iou >= _m50.IOU_CORRECT_THRESHOLD
            finally:
                cam_engine.remove_hooks()

            overlay = overlay_heatmap(img_bgr, cam)  # same default alpha for every model
            caption_lines = [
                f"{label}",
                f"score(GT class)={y_c:.3f}",
                f"AD={ad:.3f}  FRR={frr:.3f}",
                f"correct_detection={'YES' if correct_detection else 'NO'} (IoU={iou:.2f})",
            ]
            if class_name == "Empty Bunch" and label in EMPTY_BUNCH_DP_TEST_RECALL:
                caption_lines.append(
                    f"NOTE: {label} held-out test recall for Empty Bunch = "
                    f"{EMPTY_BUNCH_DP_TEST_RECALL[label]:.4f} -- CAM shows internal "
                    f"response only, NOT a successful detection")
            strip = make_caption_strip(_m50.IMGSZ, caption_lines)
            panel = np.vstack([resize_keep_square(overlay, PANEL_W), resize_keep_square(strip, PANEL_W)])
            panels.append(panel)

            per_model_records[label] = {
                "checkpoint_path": str(resolved[label]["path"].relative_to(REPO_ROOT)),
                "checkpoint_sha256": resolved[label]["sha256"],
                "anchor_idx": int(anchor_idx), "raw_score": float(raw_score),
                "confidence_before_occlusion": y_c, "confidence_after_occlusion": float(o_c),
                "average_drop": float(ad), "focus_retention_rate": float(frr),
                "iou_vs_gt": iou, "correct_detection": bool(correct_detection),
            }

        composite = np.hstack(panels)
        out_path = OUT_DIR / f"{sample['sample_id']}.jpg"
        cv2.imwrite(str(out_path), composite)

        visual_records.append({
            "sample_id": sample["sample_id"], "image_path": sample["image_path"],
            "class_id": class_id, "class_name": class_name,
            "gt_box_xyxy_imgsz": sample["gt_box_xyxy_imgsz"],
            "panel_path": str(out_path.relative_to(REPO_ROOT)),
            "per_model": per_model_records,
            "empty_bunch_dp_test_recall_note": (
                "E1_full and E2_partial_P2 both have held-out test recall=0.0000 for this "
                "class (scripts/45's locked result) -- these panels show internal model "
                "response only, never a successful detection" if class_name == "Empty Bunch" else None),
        })
        print(f"{sample['sample_id']}: saved {out_path.relative_to(REPO_ROOT)}")

    manifest_out = {
        "manifest_kind": "xai_matched_visual_manifest",
        "note": "Deterministic subset (first 2 samples per class, in the sample manifest's own "
                "stored order) -- selection never used any model prediction, confidence, or CAM "
                "value.",
        "selection_rule": f"first {N_PER_CLASS} samples per class in sample_manifest.json's own order",
        "source_sample_manifest": str(manifest_path.relative_to(REPO_ROOT))
            if manifest_path.is_absolute() else str(manifest_path),
        "target_layer_idx": _m50.TARGET_LAYER_IDX, "imgsz": _m50.IMGSZ,
        "top_fraction": _m50.TOP_FRACTION, "iou_correct_detection_threshold": _m50.IOU_CORRECT_THRESHOLD,
        "git_commit": _m50.git_commit(), "git_dirty": _m50.git_dirty(),
        "n_panels": len(visual_records),
        "no_model_update": True, "no_optimizer_step": True, "no_checkpoint_selection": True,
        "panels": visual_records,
    }
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest_out, f, indent=2)
    print(f"\nSaved {len(visual_records)} panels to {OUT_DIR}")
    print(f"Saved {OUT_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
