#!/usr/bin/env python3
"""Builds the MATCHED-SAMPLE manifest used by the B2/E1/E2 XAI comparison --
one row per (image, GT-box-within-that-image), selected ENTIRELY from the
held-out test split's own images/labels. No model (B2, E1, or E2) is ever
loaded or run here: selection cannot depend on any model's predictions,
confidence, correctness, or CAM quality by construction, not just by policy.

Default (no --quota-per-class): uses EVERY GT box in the test split -- the
same population Ultralytics' own validator counts (3821 boxes across 1051
images, per the B1/B2/E1/E2 held-out test runs already completed). Exact
duplicate label ROWS within a single image's .txt file are dropped (keeping
the first occurrence), mirroring Ultralytics' own de-duplication (which is
why B2's held-out test run above printed "1 duplicate labels removed") --
so this manifest's total box count matches the same 3821 already reported,
not a silently different number.

--quota-per-class N caps the population to N GT boxes per class, chosen by
a fixed, seed-42 deterministic shuffle (random.Random(42).shuffle on the
sorted (image, box_index) list for that class, then truncated) -- for
compute-limited runs. Still entirely GT-driven: shuffling is seeded and
uses only the box's identity (image name + box index), never anything
about a model's output.

Usage:
    python scripts/49_build_xai_matched_manifest.py
    python scripts/49_build_xai_matched_manifest.py --quota-per-class 40

Output (refuses to overwrite):
    results/xai_matched/sample_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_MANIFEST = REPO_ROOT / "results/xai_matched/sample_manifest.json"
SEED = 42
EXPECTED_TEST_IMAGES = 1051
EXPECTED_TOTAL_BOXES = 3821  # cross-check vs the already-completed B1/B2/E1/E2 held-out test runs


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_label_file(lbl_path: Path) -> tuple[list[tuple[int, float, float, float, float]], int]:
    """Returns (deduped rows in file order, n_duplicates_removed)."""
    seen = set()
    rows = []
    n_dupes = 0
    for line in lbl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls_id = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:5])
        key = (cls_id, round(cx, 6), round(cy, 6), round(w, 6), round(h, 6))
        if key in seen:
            n_dupes += 1
            continue
        seen.add(key)
        rows.append((cls_id, cx, cy, w, h))
    return rows, n_dupes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quota-per-class", type=int, default=None,
                        help="cap the population to N GT boxes per class (deterministic, "
                             "seed=42 shuffle of GT box identities only). Default: use every "
                             "GT box in the test split")
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()

    if OUT_MANIFEST.exists():
        print(f"FAIL: {OUT_MANIFEST} already exists -- refusing to overwrite. Move/rename it "
              f"first if you intend to rebuild the manifest")
        return 1

    with open("configs/dataset.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    class_names = ds_cfg["names"]
    splits_dir = Path(ds_cfg["output_dir"])
    images_dir = splits_dir / "test" / "images"
    labels_dir = splits_dir / "test" / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        print(f"FAIL: {images_dir} or {labels_dir} not found")
        return 1

    image_paths = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(image_paths) != EXPECTED_TEST_IMAGES:
        print(f"FAIL: found {len(image_paths)} test images, expected {EXPECTED_TEST_IMAGES}")
        return 1

    image_sha256_cache: dict[str, str] = {}
    samples_by_class: dict[int, list[dict]] = {c: [] for c in range(len(class_names))}
    total_dupes = 0

    for img_path in image_paths:
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue  # background image, no GT boxes -- not a sample source
        rows, n_dupes = parse_label_file(lbl_path)
        total_dupes += n_dupes
        if not rows:
            continue
        lbl_sha256 = sha256_of(lbl_path)
        for box_idx, (cls_id, cx, cy, w, h) in enumerate(rows):
            if cls_id not in samples_by_class:
                print(f"FAIL: {lbl_path} has out-of-range class_id={cls_id} "
                      f"(nc={len(class_names)})")
                return 1
            x1 = (cx - w / 2) * args.imgsz
            y1 = (cy - h / 2) * args.imgsz
            x2 = (cx + w / 2) * args.imgsz
            y2 = (cy + h / 2) * args.imgsz
            rel_img = str(img_path.resolve().relative_to(REPO_ROOT))
            if rel_img not in image_sha256_cache:
                image_sha256_cache[rel_img] = sha256_of(img_path)
            samples_by_class[cls_id].append({
                "image_path": rel_img,
                "image_sha256": image_sha256_cache[rel_img],
                "label_path": str(lbl_path.resolve().relative_to(REPO_ROOT)),
                "label_sha256": lbl_sha256,
                "box_index": box_idx,
                "class_id": cls_id,
                "class_name": class_names[cls_id],
                "gt_box_yolo_norm": [cx, cy, w, h],
                "gt_box_xyxy_imgsz": [x1, y1, x2, y2],
            })

    total_boxes_found = sum(len(v) for v in samples_by_class.values())
    if args.quota_per_class is None and total_boxes_found != EXPECTED_TOTAL_BOXES:
        print(f"FAIL: total GT boxes found (post-dedup) = {total_boxes_found}, expected "
              f"{EXPECTED_TOTAL_BOXES} (the count already reported by evaluate_detector for "
              f"this exact test split in the B1/B2/E1/E2 held-out test runs) -- investigate "
              f"before building a manifest that silently disagrees with the locked test results")
        return 1

    selection_rule = ("every GT box in the held-out test split (post exact-duplicate-row removal "
                      "within each label file), file order, sorted by image filename")
    if args.quota_per_class is not None:
        rng = random.Random(SEED)
        capped = {}
        for cls_id, rows in samples_by_class.items():
            ordered = sorted(rows, key=lambda r: (r["image_path"], r["box_index"]))
            rng.shuffle(ordered)
            capped[cls_id] = ordered[:args.quota_per_class]
        samples_by_class = capped
        selection_rule = (f"seed={SEED} deterministic shuffle of each class's (image, box_index) "
                          f"identities, truncated to {args.quota_per_class} per class -- shuffle "
                          f"key is GT identity only, never a model prediction/confidence/CAM value")

    samples = []
    for cls_id in sorted(samples_by_class):
        for row in samples_by_class[cls_id]:
            sample_id = f"{Path(row['image_path']).stem}_box{row['box_index']}_c{cls_id}"
            samples.append({"sample_id": sample_id, **row})

    n_per_class = {class_names[c]: len(v) for c, v in samples_by_class.items()}

    manifest = {
        "manifest_kind": "xai_matched_sample_manifest",
        "note": "Derived exclusively from test-split images + GT labels -- no model was loaded "
                "or run to build this manifest, so selection cannot depend on any model's "
                "predictions, confidence, correctness, or CAM quality.",
        "seed": SEED,
        "selection_rule": selection_rule,
        "quota_per_class": args.quota_per_class,
        "imgsz": args.imgsz,
        "test_split_identity": {
            "images_dir": str(images_dir.resolve().relative_to(REPO_ROOT)),
            "labels_dir": str(labels_dir.resolve().relative_to(REPO_ROOT)),
            "n_images_on_disk": len(image_paths),
            "n_images_expected": EXPECTED_TEST_IMAGES,
            "n_duplicate_label_rows_removed": total_dupes,
            "n_total_boxes_reference": EXPECTED_TOTAL_BOXES,
        },
        "n_samples_total": len(samples),
        "n_samples_per_class": n_per_class,
        "samples": samples,
    }

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"n_samples_total={len(samples)}  n_duplicate_label_rows_removed={total_dupes}")
    print("n_samples_per_class:")
    for name, n in n_per_class.items():
        print(f"  {name:>14}: {n}")
    print(f"\nSaved {OUT_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
