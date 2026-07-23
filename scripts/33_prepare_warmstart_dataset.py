#!/usr/bin/env python3
"""Prepares a PUBLIC auxiliary detection dataset (e.g. the Nature Scientific
Data oil-palm FFB ripeness dataset, https://doi.org/10.57760/sciencedb.j00001.00713)
into an Ultralytics-standard layout, for the NON-PRIVATE warm-start track
(see conversation: P2's per-sample signal is small partly because COCO-init
gradients across samples partially cancel -- a public, in-domain warm-start
before federated DP training may raise per-sample signal without touching
noise or epsilon at all).

CRITICAL SCOPE BOUNDARY: this script and scripts/34 touch ONLY public data.
Nothing here reads any K=4 client file, any federated_partitions/ path, or
any DP/Opacus code. The output is a plain checkpoint file with the exact
same architecture as models/base_groupnorm.pt -- a drop-in --init-weights
for the EXISTING E2 scripts, no new architecture, no new privacy mechanism.

Input: one or more extracted split folders (e.g. the unzipped train_rev2/,
valid_rev2/, test_rev2/). Structure inside is NOT assumed -- files are
matched purely by basename (stem) via a recursive walk, so this is robust
to flat layouts (images and .txt sitting in the same folder, as observed:
`frame2--33-_png.rf.<hash>.jpg` + `frame2--33-_png.rf.<hash>.txt`) or nested
images/labels layouts.

Auto-detects the number of classes (nc) actually present across ALL label
files (max class index + 1) -- NOT assumed from the paper's stated 6
categories, since box-level labels may only mark presence (e.g. a single
generic "bunch" class) rather than per-box ripeness. Hard-fails if any
class index >= 6 is found: models/base_groupnorm.pt's head has exactly 6
output channels (the locked P2/B1/B2 class count), and reusing it AS-IS
(no checkpoint surgery) for warm-start requires aux label indices to be a
SUBSET of that space -- this is what makes scripts/34's approach need zero
head-reshaping surgery. If nc_aux < 6, only that many output channels get
warm-started; the rest keep their original values until real federated
training exercises them -- expected and harmless (the class head is always
going to be fully retrained on the real 6-class labels regardless).

Malformed label lines (wrong token count, unparseable floats) are counted
and reported, not silently included. Images with no matching label file
are counted and EXCLUDED by default (not copied as unlabeled background) --
report their count so nothing is silently dropped.

  python scripts/33_prepare_warmstart_dataset.py --train-dir train_rev2 --valid-dir valid_rev2 --test-dir test_rev2 --out-dir data/warmstart_ffb

Output:
  data/warmstart_ffb/{train,valid,test}/images/*  (copied)
  data/warmstart_ffb/{train,valid,test}/labels/*  (copied)
  data/warmstart_ffb/data.yaml
  results/warmstart_dataset_prep_audit.json
"""
import argparse
import json
import shutil
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png")


def find_pairs(root: Path):
    """Returns (pairs, unmatched_images, orphan_labels) -- (jpg_path, txt_path)
    tuples matched purely by stem, via a recursive walk (layout-agnostic)."""
    txt_by_stem = {}
    for p in root.rglob("*.txt"):
        txt_by_stem.setdefault(p.stem, p)  # first match wins if duplicate stems
    imgs = [p for ext in IMG_EXTS for p in root.rglob(f"*{ext}")]
    pairs, unmatched = [], []
    used_stems = set()
    for img in imgs:
        if img.stem in txt_by_stem and img.stem not in used_stems:
            pairs.append((img, txt_by_stem[img.stem]))
            used_stems.add(img.stem)
        else:
            unmatched.append(img)
    orphan_labels = [p for stem, p in txt_by_stem.items() if stem not in used_stems]
    return pairs, unmatched, orphan_labels


def validate_and_copy(pairs, out_images: Path, out_labels: Path):
    """Copies each pair, validating every label line is `class x y w h` with
    parseable numbers. Returns (class_counts: dict[int,int], malformed: list)."""
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    class_counts: dict = {}
    malformed = []
    for img_path, txt_path in pairs:
        lines = txt_path.read_text().strip().splitlines()
        for line in lines:
            tokens = line.split()
            if len(tokens) != 5:
                malformed.append(f"{txt_path.name}: {line!r} (expected 5 tokens, got {len(tokens)})")
                continue
            try:
                cls = int(tokens[0])
                coords = [float(t) for t in tokens[1:]]
            except ValueError:
                malformed.append(f"{txt_path.name}: {line!r} (unparseable)")
                continue
            if cls < 0 or any(not (0.0 <= c <= 1.0) for c in coords):
                malformed.append(f"{txt_path.name}: {line!r} (out-of-range class/coords)")
                continue
            class_counts[cls] = class_counts.get(cls, 0) + 1
        shutil.copy2(img_path, out_images / img_path.name)
        shutil.copy2(txt_path, out_labels / f"{img_path.stem}.txt")
    return class_counts, malformed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True, help="extracted train_rev2 folder")
    parser.add_argument("--valid-dir", required=True, help="extracted valid_rev2 folder")
    parser.add_argument("--test-dir", default=None, help="extracted test_rev2 folder (optional)")
    parser.add_argument("--out-dir", default="data/warmstart_ffb")
    parser.add_argument("--max-nc", type=int, default=6,
                        help="hard cap on detected classes (default 6, matching the locked "
                             "B1/B2/E1/E2 class head) -- refuses to proceed above this")
    args = parser.parse_args()

    splits = {"train": Path(args.train_dir), "valid": Path(args.valid_dir)}
    if args.test_dir:
        splits["test"] = Path(args.test_dir)
    for name, p in splits.items():
        if not p.exists():
            print(f"FAIL: --{name}-dir {p} does not exist"); return 1

    out_dir = Path(args.out_dir)
    all_class_counts: dict = {}
    all_malformed = []
    split_stats = {}
    for split, root in splits.items():
        pairs, unmatched, orphans = find_pairs(root)
        print(f"[{split}] found {len(pairs)} image/label pairs, "
              f"{len(unmatched)} images with no label (excluded), "
              f"{len(orphans)} orphan label files (no matching image)")
        if not pairs:
            print(f"FAIL: no matched pairs found under {root} -- check the folder was extracted correctly")
            return 1
        class_counts, malformed = validate_and_copy(
            pairs, out_dir / split / "images", out_dir / split / "labels")
        for c, n in class_counts.items():
            all_class_counts[c] = all_class_counts.get(c, 0) + n
        all_malformed.extend(malformed)
        split_stats[split] = {
            "n_pairs": len(pairs), "n_unmatched_images": len(unmatched),
            "n_orphan_labels": len(orphans), "class_counts": class_counts,
            "n_malformed_lines": len(malformed),
        }

    if all_malformed:
        print(f"[!] {len(all_malformed)} malformed label lines skipped (not counted toward nc or copied):")
        for m in all_malformed[:10]:
            print(f"    {m}")
        if len(all_malformed) > 10:
            print(f"    ...(+{len(all_malformed) - 10} more)")

    if not all_class_counts:
        print("FAIL: zero valid label lines found across all splits"); return 1
    nc = max(all_class_counts) + 1
    print(f"\ndetected classes: {sorted(all_class_counts)}  (nc = max_index+1 = {nc})")
    print(f"class distribution: {dict(sorted(all_class_counts.items()))}")
    if nc > args.max_nc:
        print(f"FAIL: detected nc={nc} exceeds --max-nc={args.max_nc} -- models/base_groupnorm.pt's "
              f"head has exactly {args.max_nc} output channels; a class index >= {args.max_nc} would "
              f"crash or corrupt Ultralytics' loss computation if trained as-is. Aborting rather than "
              f"proceeding with an unsafe assumption.")
        return 1

    names = [f"class_{i}" for i in range(nc)]
    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "nc": nc,
        "names": names,
        "note": ("PLACEHOLDER class names -- this auxiliary dataset's label semantics are NOT assumed "
                "to match the real 6-class ripeness scheme; used only for non-private warm-start "
                "feature adaptation, discarded once real federated training on the true labels begins"),
    }
    if "test" in splits:
        data_yaml["test"] = "test/images"
    import yaml
    with open(out_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    print(f"\nWrote {out_dir / 'data.yaml'}")

    record = {
        "note": "PUBLIC auxiliary dataset prep for the non-private warm-start track -- touches no "
                "client/federated_partitions data whatsoever.",
        "source_dirs": {k: str(v) for k, v in splits.items()},
        "out_dir": str(out_dir), "detected_nc": nc, "max_nc_allowed": args.max_nc,
        "class_counts": all_class_counts, "split_stats": split_stats,
        "n_malformed_lines_total": len(all_malformed),
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/warmstart_dataset_prep_audit.json", "w") as f:
        json.dump(record, f, indent=2)
    print("Saved results/warmstart_dataset_prep_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
