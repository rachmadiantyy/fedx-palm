"""
[01] Anti-leakage re-split by bunch_id (stratified group split).

Step 1 of the rebuild pipeline. Roboflow splits train/valid/test
randomly PER FRAME, but the source is video: many frames come from the
SAME palm bunch. A bunch appearing in both train and valid means the
model memorizes bunches instead of generalizing (SOFT leakage = 100%
in the original split — see REVISI_THESIS.md F+.1).

Fix: regroup ALL images by bunch_id and split 80/10/10 so each bunch
lands in exactly one split. Stratify by each bunch's dominant class so
rare classes (e.g. Empty Bunch) stay represented in valid/test.

  bunch_id: framesawit27-7-_png.rf.<hash>.jpg -> "framesawit27"

Deterministic: seed=42.

Usage:
    python thesis_rebuild/scripts/01_resplit_bunch_id.py

Input:  data/raw/{train,valid,test}/{images,labels}
Output: data/resplit/{train,valid,test}/{images,labels}
        data/resplit/data.yaml
"""
import argparse
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

NC = 6
CLASS_NAMES = {
    0: "Abnormal", 1: "Empty Bunch", 2: "Overripe",
    3: "Ripe", 4: "Underripe", 5: "Unripe",
}
RATIO = {"train": 0.80, "valid": 0.10, "test": 0.10}
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def strip_rf(name: str) -> str:
    """framesawit27-7-_png.rf.abc123.jpg -> framesawit27-7-_png.jpg-ish stem."""
    return re.split(r"\.rf\.", os.path.basename(name))[0]


def bunch_id(path: Path) -> str:
    base = strip_rf(path.name)
    m = re.match(r"(frame[a-z]*\d+)", base, re.I)
    return m.group(1) if m else base


def primary_class(lbl_path: Path) -> int:
    if not lbl_path.exists():
        return -1
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                return int(parts[0])
    return -1


def audit_leakage(resplit_dir: Path) -> bool:
    """Verify no bunch_id appears in more than one split. Returns clean?"""
    split_bunches = {}
    for split in ("train", "valid", "test"):
        img_dir = resplit_dir / split / "images"
        bset = set()
        if img_dir.exists():
            for img in img_dir.iterdir():
                if img.suffix.lower() in IMG_EXTS:
                    bset.add(bunch_id(img))
        split_bunches[split] = bset

    clean = True
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = split_bunches[a] & split_bunches[b]
        status = "BERSIH" if not overlap else f"BOCOR ({len(overlap)})"
        print(f"  {a:6} <-> {b:6}: {status}")
        if overlap:
            clean = False
    return clean


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="[01] Anti-leakage bunch_id re-split")
    p.add_argument("--raw", type=str, default=str(REPO_ROOT / "data" / "raw"))
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "data" / "resplit"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw)
    out_dir = Path(args.out)

    if not raw_dir.exists():
        sys.exit(f"ERROR: {raw_dir} not found. Run 00_download_dataset.py first.")

    import random
    rng = random.Random(args.seed)

    print("=" * 70)
    print("[01] ANTI-LEAKAGE RE-SPLIT (bunch_id stratified group split)")
    print("=" * 70)

    # 1. Pool all (image, label) pairs from the three original splits
    pairs = []
    for split in ("train", "valid", "test"):
        img_dir = raw_dir / split / "images"
        lbl_dir = raw_dir / split / "labels"
        if not img_dir.exists():
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() in IMG_EXTS:
                pairs.append((img, lbl_dir / (img.stem + ".txt")))
    print(f"  Pooled images:  {len(pairs)}")

    # 2. Group by bunch_id
    bunches = defaultdict(list)
    for img, lbl in pairs:
        bunches[bunch_id(img)].append((img, lbl))
    print(f"  Unique bunches: {len(bunches)}")

    # 3. Dominant class per bunch (for stratification)
    bunch_cls = {
        bid: Counter(primary_class(l) for _, l in items).most_common(1)[0][0]
        for bid, items in bunches.items()
    }

    # 4. Stratified group split: per class, deterministic shuffle, 80/10/10
    by_cls = defaultdict(list)
    for bid, c in bunch_cls.items():
        by_cls[c].append(bid)

    split_assign = {}
    for c, bids in by_cls.items():
        bids = sorted(bids)
        rng.shuffle(bids)
        n = len(bids)
        n_train = int(round(n * RATIO["train"]))
        n_valid = int(round(n * RATIO["valid"]))
        for i, bid in enumerate(bids):
            if i < n_train:
                split_assign[bid] = "train"
            elif i < n_train + n_valid:
                split_assign[bid] = "valid"
            else:
                split_assign[bid] = "test"

    # 5. Materialize: copy files into clean split dirs
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "valid", "test"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    cnt = Counter()
    cls_dist = {s: Counter() for s in ("train", "valid", "test")}
    for bid, items in bunches.items():
        split = split_assign[bid]
        for img, lbl in items:
            shutil.copy2(img, out_dir / split / "images" / img.name)
            if lbl.exists():
                shutil.copy2(lbl, out_dir / split / "labels" / lbl.name)
            cnt[split] += 1
            cls_dist[split][primary_class(lbl)] += 1

    # 6. Write data.yaml
    with open(out_dir / "data.yaml", "w") as f:
        yaml.safe_dump({
            "path": str(out_dir),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": NC,
            "names": CLASS_NAMES,
        }, f, sort_keys=False)

    # 7. Report
    print("\n  Re-split result (per bunch, stratified):")
    for s in ("train", "valid", "test"):
        n_b = sum(1 for sp in split_assign.values() if sp == s)
        pretty = {CLASS_NAMES[c]: cls_dist[s][c]
                  for c in sorted(cls_dist[s]) if 0 <= c < NC}
        print(f"    {s:6}: {cnt[s]:5d} img | {n_b:4d} bunches | {pretty}")

    print("\n  Leakage audit:")
    clean = audit_leakage(out_dir)
    if not clean:
        sys.exit("\nERROR: leakage detected after re-split — aborting.")

    print(f"\n  data.yaml -> {out_dir / 'data.yaml'}")
    print("\nNEXT: python thesis_rebuild/scripts/02_dirichlet_partition.py")


if __name__ == "__main__":
    main()
