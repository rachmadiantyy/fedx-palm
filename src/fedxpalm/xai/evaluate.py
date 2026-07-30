"""Dataset-level Grad-CAM++ faithfulness evaluation: per-class Average Drop and
Focus Retention Rate over a YOLO-format image+label directory (mirrors thesis
Table 4.9 "Metrik faithfulness XAI per-kelas").

For every ground-truth box in the set, explains the model's own best-scoring
anchor for that box's class (not necessarily a "correct" detection -- this
measures whether the explanation is faithful to whatever the model actually
computed, which is the point of a faithfulness metric).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from fedxpalm.xai.gradcam import YOLOGradCAMPlusPlus, overlay_heatmap
from fedxpalm.xai.metrics import average_drop, class_confidence, focus_retention_rate, occlude_top_region


def _load_image_and_labels(img_path: Path, lbl_path: Path, imgsz: int):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None, None, None
    orig_h, orig_w = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    boxes = []  # (class_id, x1, y1, x2, y2) in resized-image pixel coords
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            cls_id, cx, cy, w, h = line.split()[:5]
            cls_id = int(cls_id)
            cx, cy, w, h = (float(v) for v in (cx, cy, w, h))
            x1 = (cx - w / 2) * imgsz
            y1 = (cy - h / 2) * imgsz
            x2 = (cx + w / 2) * imgsz
            y2 = (cy + h / 2) * imgsz
            boxes.append((cls_id, x1, y1, x2, y2))
    return resized, tensor, boxes


def evaluate_faithfulness(
    detection_model: torch.nn.Module,
    images_dir: str,
    labels_dir: str,
    class_names: list[str],
    imgsz: int = 640,
    target_layer_idx: int = 22,
    top_fraction: float = 0.2,
    max_images: int | None = None,
    save_overlays_dir: str | None = None,
) -> dict:
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    image_paths = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if max_images is not None:
        image_paths = image_paths[:max_images]

    cam_engine = YOLOGradCAMPlusPlus(detection_model, target_layer_idx=target_layer_idx)
    per_class = defaultdict(lambda: {"ad": [], "frr": []})

    if save_overlays_dir:
        Path(save_overlays_dir).mkdir(parents=True, exist_ok=True)

    try:
        for img_path in image_paths:
            lbl_path = labels_dir / (img_path.stem + ".txt")
            img_bgr, tensor, boxes = _load_image_and_labels(img_path, lbl_path, imgsz)
            if img_bgr is None or not boxes:
                continue

            for box_idx, (class_id, x1, y1, x2, y2) in enumerate(boxes):
                cam, raw_score, anchor_idx, _decoded_box = cam_engine.generate(
                    tensor.clone(), class_id, output_size=(imgsz, imgsz))
                y_c = torch.sigmoid(torch.tensor(raw_score)).item()

                occluded = occlude_top_region(tensor, cam, top_fraction=top_fraction)
                o_c = class_confidence(detection_model, occluded, class_id, anchor_idx)

                ad = average_drop(y_c, o_c)
                frr = focus_retention_rate(cam, (x1, y1, x2, y2))
                per_class[class_id]["ad"].append(ad)
                per_class[class_id]["frr"].append(frr)

                if save_overlays_dir and box_idx == 0:
                    overlay = overlay_heatmap(img_bgr, cam)
                    out_name = f"{img_path.stem}_c{class_id}.jpg"
                    cv2.imwrite(str(Path(save_overlays_dir) / out_name), overlay)
    finally:
        cam_engine.remove_hooks()

    results = {}
    all_ad, all_frr = [], []
    for class_id, vals in sorted(per_class.items()):
        name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        results[name] = {
            "n": len(vals["ad"]),
            "average_drop": float(np.mean(vals["ad"])),
            "focus_retention_rate": float(np.mean(vals["frr"])),
        }
        all_ad.extend(vals["ad"])
        all_frr.extend(vals["frr"])

    results["__global__"] = {
        "n": len(all_ad),
        "average_drop": float(np.mean(all_ad)) if all_ad else 0.0,
        "focus_retention_rate": float(np.mean(all_frr)) if all_frr else 0.0,
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="path to a .pt checkpoint")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--target-layer-idx", type=int, default=22)
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--save-overlays-dir", default=None)
    parser.add_argument("--out-json", default="results/xai_faithfulness.json")
    args = parser.parse_args()

    import yaml
    from ultralytics import YOLO

    with open(args.dataset_config) as f:
        names = yaml.safe_load(f)["names"]

    yolo = YOLO(args.weights)
    results = evaluate_faithfulness(
        yolo.model, args.images_dir, args.labels_dir, names,
        imgsz=args.imgsz, target_layer_idx=args.target_layer_idx,
        top_fraction=args.top_fraction, max_images=args.max_images,
        save_overlays_dir=args.save_overlays_dir,
    )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    for name, m in results.items():
        print(f"{name:>15}: n={m['n']:4d}  AD={m['average_drop']:.3f}  FRR={m['focus_retention_rate']:.3f}")
