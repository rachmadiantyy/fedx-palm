"""Thin wrapper around Ultralytics' own validator: mAP@0.5, mAP@0.5:0.95,
precision, recall, F1 -- global and per-class (mirrors thesis metric set in
Bab 3.11.1 / Bab 4 result tables).
"""
from __future__ import annotations

import yaml
from ultralytics import YOLO


def _load_class_names(data_yaml: str) -> dict[int, str]:
    """Ground-truth class-id -> name mapping, read directly from data_yaml
    (this project's single, immutable 6-class dataset) rather than trusting
    `metrics.names`. Checkpoints produced by the federated server's manual
    save/reload cycle (torch.load -> load_state_dict -> torch.save, every
    round -- see fedxpalm.federated.server.run_federated_training, used by
    B2/E1/E2) have been observed to lose their real class names somewhere in
    that cycle and report back numeric fallback names ("0".."5") instead --
    a real bug (found via scripts/45's held-out E1 test run), NOT a dataset
    problem. data_yaml's own `names` is authoritative and never touched by
    that cycle, so reading it directly here sidesteps the bug entirely
    regardless of what a given checkpoint's own metadata says."""
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return dict(enumerate(names))


def evaluate_detector(weights_path: str, data_yaml: str, split: str = "test", imgsz: int = 640, device: str = "0",
                       plots: bool = False, project: str | None = None, name: str | None = None) -> dict:
    """`plots=True` additionally makes Ultralytics save its own confusion_matrix.png,
    PR_curve.png, P_curve.png, R_curve.png, F1_curve.png, val_batch*_pred.jpg, etc. to
    `project/name/` (default `runs/detect/val*`) -- real evaluation artifacts straight
    from this exact run, not a hand-drawn chart. See scripts/42_generate_b1_test_plots.py."""
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml, split=split, imgsz=imgsz, device=device, plots=plots, verbose=False,
                        project=project, name=name)

    names = _load_class_names(data_yaml)  # {class_id: name} -- see _load_class_names' docstring
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
