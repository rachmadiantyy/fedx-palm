"""Thin wrapper around Ultralytics' own validator: mAP@0.5, mAP@0.5:0.95,
precision, recall, F1 -- global and per-class (mirrors thesis metric set in
Bab 3.11.1 / Bab 4 result tables).
"""
from __future__ import annotations

from ultralytics import YOLO


def evaluate_detector(weights_path: str, data_yaml: str, split: str = "test", imgsz: int = 640, device: str = "0") -> dict:
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml, split=split, imgsz=imgsz, device=device, plots=False, verbose=False)

    names = metrics.names  # {class_id: name}
    precision, recall, f1 = metrics.box.p, metrics.box.r, metrics.box.f1
    ap50, ap50_95 = metrics.box.ap50, metrics.box.ap

    per_class = {}
    for i, class_id in enumerate(metrics.box.ap_class_index):
        name = names[int(class_id)]
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "ap50": float(ap50[i]),
            "ap50_95": float(ap50_95[i]),
        }

    return {
        "split": split,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class": per_class,
    }
