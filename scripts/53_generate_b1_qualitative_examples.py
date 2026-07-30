#!/usr/bin/env python3
"""Generates B1's qualitative detection examples for the manuscript/Bab 4
Figure 4.2 -- REAL model.predict() output (bounding boxes, class labels,
confidence scores drawn by Ultralytics itself), never a hand-drawn or
diagram-tool illustration.

Image selection is entirely GT-label-driven and deterministic (first
occurrence, sorted by filename, within each requested class's label set)
-- never chosen by looking at model predictions, confidence, or "which
one looks good". Default selects one image per class in CLASS_ORDER
(4 images: Ripe first per the requirement to include that class, then
three others for visual variety), matching this project's established
"never cherry-pick from model output" selection discipline (see
scripts/49's sample manifest for the same principle).

Usage:
    python scripts/53_generate_b1_qualitative_examples.py --device 0

Output (Ultralytics' own predict() layout; refuses to overwrite):
    runs/b1_qualitative_examples/predict/*.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

import fedxpalm  # noqa: E402,F401 (applies the GroupNorm-safe `fuse()` patch)

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_ORDER = ["Ripe", "Abnormal", "Unripe", "Overripe"]  # Ripe first (required); rest for variety
N_TOTAL = 4
OUT_DIR = REPO_ROOT / "runs/b1_qualitative_examples"
DEFAULT_WEIGHTS = "runs/b1_centralized_leakagefree/train/weights/best.pt"


def first_image_per_class(images_dir: Path, labels_dir: Path, class_names: list[str]) -> dict[str, Path]:
    """Deterministic, GT-only selection: for each requested class name, the
    first image (sorted by filename) whose label file contains at least one
    box of that class. Never looks at any model output."""
    name_to_id = {n: i for i, n in enumerate(class_names)}
    image_paths = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    found: dict[str, Path] = {}
    remaining = set(CLASS_ORDER)
    for img_path in image_paths:
        if not remaining:
            break
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        classes_in_image = set()
        for line in lbl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            classes_in_image.add(int(line.split()[0]))
        for cname in list(remaining):
            if name_to_id.get(cname) in classes_in_image:
                found[cname] = img_path
                remaining.discard(cname)
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()

    if (OUT_DIR / "predict").exists():
        print(f"FAIL: {OUT_DIR / 'predict'} already exists -- refusing to overwrite. "
              f"Move/rename it first if you intend to regenerate these examples")
        return 1

    weights_path = REPO_ROOT / args.weights
    if not weights_path.exists():
        print(f"FAIL: {weights_path} not found")
        return 1

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    class_names = ds_cfg["names"]
    splits_dir = Path(ds_cfg["output_dir"]).resolve()
    images_dir = splits_dir / "test" / "images"
    labels_dir = splits_dir / "test" / "labels"

    found = first_image_per_class(images_dir, labels_dir, class_names)
    missing = [c for c in CLASS_ORDER if c not in found]
    if missing:
        print(f"FAIL: could not find a held-out test image containing class(es) {missing}")
        return 1

    selected = [found[c] for c in CLASS_ORDER][:N_TOTAL]
    print("Selected images (deterministic, GT-label-driven, one per class):")
    for cname, p in zip(CLASS_ORDER, selected):
        print(f"  {cname}: {p.relative_to(REPO_ROOT)}")

    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    model.predict(
        source=[str(p) for p in selected],
        imgsz=args.imgsz, conf=args.conf, device=args.device,
        save=True, project=str(OUT_DIR), name="predict", exist_ok=False,
    )
    print(f"\nSaved qualitative examples to {OUT_DIR / 'predict'}")
    print("Each output image already has Ultralytics' own drawn bounding boxes, class labels, "
          "and confidence scores -- use these directly for Figure 4.2, not a redrawn version.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
