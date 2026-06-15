"""
Per-class XAI evaluation: Grad-CAM++ faithfulness on the test set.

Computes, PER ripeness class and globally:
  - Average Drop (AD %): lower = explanation highlights regions the model
    actually relies on.
  - Focus Retention Rate (FRR, 0-1): fraction of heatmap intensity that
    falls inside the fruit bounding boxes — doubles as the agronomic
    "does the model look at the fruit, not the background" check.

Reuses the existing Grad-CAM++ implementation in xai/explainer.py.

Works with either:
  - a federated checkpoint ({'state':..., 'config':...} from fl_dp_loop), or
  - a centralized Ultralytics checkpoint (B1 best.pt).

Outputs:
  thesis_rebuild/tables/xai_per_class.csv   (one row per class + GLOBAL)

Use:
  python thesis_rebuild/scripts/evaluate_xai.py \
      --weights thesis_rebuild/runs/b2_fl_K4_seed42/best.pt \
      --data data/global_test.yaml --n-images 120
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402

from thesis_rebuild.scripts.utils.fl_dp_loop import build_gn_yolo  # noqa: E402
from xai.explainer import (  # noqa: E402
    GradCAMPlusPlus,
    AverageDrop,
    FocusRetentionRate,
)

CLASS_NAMES = ["Abnormal", "Empty Bunch", "Overripe", "Ripe", "Underripe", "Unripe"]


def load_model(weights: str, device: str) -> YOLO:
    """Rebuild a GN YOLO from a federated checkpoint, or load B1 directly."""
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state" in ckpt:
        cfg = ckpt.get("config", {})
        base = cfg.get("weights", "yolo11n.pt")
        groups = cfg.get("num_groups", 8)
        model = build_gn_yolo(base, groups, freeze_backbone=False)
        model.load_state_dict(ckpt["state"], strict=False)
        yolo = YOLO(base)
        yolo.model = model
        print(f"[model] rebuilt GN YOLO from federated ckpt "
              f"(K={cfg.get('K')}, sigma={cfg.get('noise_multiplier')})")
    else:
        yolo = YOLO(weights)
        print(f"[model] loaded Ultralytics ckpt {weights}")
    yolo.model.to(device).eval()
    return yolo


def list_test_images(data_yaml: str) -> list[Path]:
    info = check_det_dataset(data_yaml)
    src = info.get("test") or info.get("val")
    src = Path(src)
    if src.is_dir():
        imgs = sorted(
            p for p in src.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
    else:  # a .txt list
        imgs = [Path(line.strip()) for line in src.read_text().splitlines() if line.strip()]
    return imgs


def label_path_for(img: Path) -> Path:
    parts = list(img.parts)
    if "images" in parts:
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(img: Path, w: int, h: int) -> list[tuple[int, list[int]]]:
    """Return [(class_id, [x1,y1,x2,y2] px), ...] from a YOLO label file."""
    lp = label_path_for(img)
    if not lp.exists():
        return []
    out = []
    for line in lp.read_text().splitlines():
        f = line.split()
        if len(f) < 5:
            continue
        cls = int(float(f[0]))
        cx, cy, bw, bh = (float(x) for x in f[1:5])
        x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
        out.append((cls, [x1, y1, x2, y2]))
    return out


def overlay_heatmap(img_bgr: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Blend a Grad-CAM++ heatmap (0-1, any size) onto a BGR image."""
    hm = cv2.resize(heatmap.astype(np.float32), (img_bgr.shape[1], img_bgr.shape[0]))
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-8)
    hm_u8 = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 0.55, hm_color, 0.45, 0)


def save_grid(per_cls_repr: dict, fig_out: str, cell: int = 320) -> None:
    """Save a 2x3 montage of representative per-class Grad-CAM++ overlays."""
    rows, cols = 2, 3
    canvas = np.full((rows * cell, cols * cell, 3), 255, np.uint8)
    for cls in range(len(CLASS_NAMES)):
        r, c = divmod(cls, cols)
        y0, x0 = r * cell, c * cell
        if cls in per_cls_repr:
            img, hm = per_cls_repr[cls]
            ov = overlay_heatmap(img, hm)
            ov = cv2.resize(ov, (cell, cell - 24))
        else:
            ov = np.full((cell - 24, cell, 3), 230, np.uint8)
        canvas[y0:y0 + cell - 24, x0:x0 + cell] = ov
        cv2.putText(canvas, f"({chr(97 + cls)}) {CLASS_NAMES[cls]}",
                    (x0 + 6, y0 + cell - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
    out = Path(fig_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"[fig] Grad-CAM++ grid -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-class Grad-CAM++ XAI evaluation")
    ap.add_argument("--weights", required=True, help="best.pt (federated or B1)")
    ap.add_argument("--data", default="data/global_test.yaml")
    ap.add_argument("--n-images", type=int, default=120, help="max images to score")
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="thesis_rebuild/tables/xai_per_class.csv")
    ap.add_argument("--fig-out",
                    default="paper/paper_b_juti_fl_xai/figures/xai_per_class.png",
                    help="output path for the 2x3 per-class heatmap grid (Fig. 1)")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    yolo = load_model(args.weights, device)

    cam = GradCAMPlusPlus(yolo)
    ad_metric = AverageDrop(yolo)
    frr_metric = FocusRetentionRate()

    # Shuffle BEFORE truncating: the test set is grouped by bunch, so the
    # first N images all belong to the same 1-2 bunches (same class). A seeded
    # shuffle spreads the sample across all classes.
    all_imgs_list = list_test_images(args.data)
    import random
    random.Random(42).shuffle(all_imgs_list)
    images = all_imgs_list[: args.n_images]
    print(f"[data] scoring {len(images)} of {len(all_imgs_list)} test images (shuffled)")

    # Accumulate per class: images, heatmaps, bbox lists
    per_cls_imgs: dict[int, list[np.ndarray]] = {}
    per_cls_heat: dict[int, list[np.ndarray]] = {}
    per_cls_boxes: dict[int, list[list[list[int]]]] = {}
    per_cls_repr: dict[int, tuple] = {}  # one (img, heatmap) per class for Fig. 1
    dbg = 0

    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        labels = read_labels(img_path, w, h)
        if not labels:
            continue
        classes_here = sorted({c for c, _ in labels})
        for cls in classes_here:
            boxes = [b for c, b in labels if c == cls]
            try:
                expl = cam.generate(img, target_class=cls)
            except Exception as e:  # keep going; XAI on detectors is fragile
                print(f"  warn: gradcam failed on {img_path.name} cls{cls}: {e}")
                continue
            if expl.heatmap is None:
                continue
            # Diagnostic for the first few: shows whether heatmaps are real
            # (max>0) or degenerate (max==0 -> near-zero class score/gradient).
            if dbg < 8:
                conf = ad_metric._get_confidence(img, cls)
                print(f"  dbg cls={CLASS_NAMES[cls]:<11} "
                      f"heatmap_max={float(expl.heatmap.max()):.4f} "
                      f"model_conf={conf:.4f}")
                dbg += 1
            per_cls_imgs.setdefault(cls, []).append(img)
            per_cls_heat.setdefault(cls, []).append(expl.heatmap)
            per_cls_boxes.setdefault(cls, []).append(boxes)
            # Keep one representative (image, heatmap) per class for Fig. 1.
            per_cls_repr.setdefault(cls, (img, expl.heatmap))
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(images)}")

    cam.cleanup()

    rows = []
    all_imgs, all_heat, all_boxes = [], [], []
    for cls in range(len(CLASS_NAMES)):
        imgs_c = per_cls_imgs.get(cls, [])
        if not imgs_c:
            rows.append({"class_id": cls, "class": CLASS_NAMES[cls],
                         "n": 0, "average_drop": "", "frr": ""})
            continue
        heat_c = per_cls_heat[cls]
        boxes_c = per_cls_boxes[cls]
        ad = ad_metric.compute(imgs_c, heat_c, target_class=cls,
                               mask_threshold=args.mask_threshold)
        frr = frr_metric.compute_batch(heat_c, boxes_c)
        rows.append({
            "class_id": cls, "class": CLASS_NAMES[cls], "n": len(imgs_c),
            "average_drop": round(ad.get("average_drop", float("nan")), 2),
            "frr": round(frr.get("mean_frr", float("nan")), 4),
        })
        all_imgs += imgs_c; all_heat += heat_c; all_boxes += boxes_c

    # Global row (target_class=None -> top prediction per image)
    if all_imgs:
        ad_g = ad_metric.compute(all_imgs, all_heat, target_class=None,
                                 mask_threshold=args.mask_threshold)
        frr_g = frr_metric.compute_batch(all_heat, all_boxes)
        rows.append({
            "class_id": -1, "class": "GLOBAL", "n": len(all_imgs),
            "average_drop": round(ad_g.get("average_drop", float("nan")), 2),
            "frr": round(frr_g.get("mean_frr", float("nan")), 4),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class_id", "class", "n",
                                          "average_drop", "frr"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Fig. 1: representative per-class Grad-CAM++ grid for the paper.
    if per_cls_repr:
        save_grid(per_cls_repr, args.fig_out)

    print(f"\n[xai] per-class results -> {out}")
    print(f"{'class':<12} {'n':>4} {'AD%':>8} {'FRR':>7}")
    for r in rows:
        print(f"{r['class']:<12} {r['n']:>4} {str(r['average_drop']):>8} "
              f"{str(r['frr']):>7}")


if __name__ == "__main__":
    main()
