"""Thin wrapper around Ultralytics' own validator: mAP@0.5, mAP@0.5:0.95,
precision, recall, F1 -- global and per-class (mirrors thesis metric set in
Bab 3.11.1 / Bab 4 result tables).
"""
from __future__ import annotations

from ultralytics import YOLO


def evaluate_detector(weights_path: str, data_yaml: str, split: str = "test", imgsz: int = 640, device: str = "0",
                       plots: bool = False, project: str | None = None, name: str | None = None) -> dict:
    """`plots=True` additionally makes Ultralytics save its own confusion_matrix.png,
    PR_curve.png, P_curve.png, R_curve.png, F1_curve.png, val_batch*_pred.jpg, etc. to
    `project/name/` (default `runs/detect/val*`) -- real evaluation artifacts straight
    from this exact run, not a hand-drawn chart. See scripts/42_generate_b1_test_plots.py."""
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml, split=split, imgsz=imgsz, device=device, plots=plots, verbose=False,
                        project=project, name=name)

    names = metrics.names  # {class_id: name}
    precision, recall, f1 = metrics.box.p, metrics.box.r, metrics.box.f1
    ap50, ap50_95 = metrics.box.ap50, metrics.box.ap

    nt_per_class = getattr(metrics, "nt_per_class", None)
    per_class = {}
    for i, class_id in enumerate(metrics.box.ap_class_index):
        name = names[int(class_id)]
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "ap50": float(ap50[i]),
            "ap50_95": float(ap50_95[i]),
            "n_instances": int(nt_per_class[int(class_id)]) if nt_per_class is not None else None,
        }

    return {
        "split": split,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class": per_class,
        # total ground-truth box instances in this split, summed across classes
        # (Ultralytics' own nt_per_class -- not derived/re-counted here)
        "n_boxes_total": int(nt_per_class.sum()) if nt_per_class is not None else None,
    }
