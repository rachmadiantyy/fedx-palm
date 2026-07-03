"""Leakage-free train/val/test split, grouped by bunch (FFB) identity and
stratified by each bunch's dominant class.

Roboflow's own train/valid/test split is per-frame and random, so multiple
photos of the *same physical bunch* can end up in both train and test --
the model then partly "memorizes" a bunch it was tested on. This module
re-pools every image across Roboflow's splits, groups them by bunch_id
parsed from the filename, and re-splits at the bunch level so no bunch_id
appears in more than one split.

Roboflow's exported filenames look like
`frame1-10-_png_jpg.rf.3f155c329fc5b6dc42143afe332116da.jpg`,
`framesawit27-7-_png.rf.<hash>.jpg`, or `frame9kombinasi-322-_png_jpg.rf.
<hash>.jpg`: a `<hash>` suffix unique per augmented copy, appended after
`.rf.`. Naively regexing the raw filename (matching e.g. a trailing
`_<digits>` right before the extension) fails silently -- the `.rf.<hash>`
segment doesn't match that pattern, so every image falls back to being
treated as its own unique "bunch", which defeats the whole point of this
module without raising an error (confirmed empirically: 10814 "bunches"
for 10814 images on a first, buggy version of this code). An earlier fix
attempt matched `frame[a-z]*\\d+` (letters then digits), but that breaks
on names like "frame9kombinasi" where a digit comes *before* the letters.
Extraction here strips the `.rf.<hash>` suffix first, then takes everything
after "frame" up to the first `-` as the bunch identity -- confirmed
against this dataset's real filenames, all of which follow
`frame<anything>-<frame position digits>-...`.
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

_RF_HASH_SUFFIX = re.compile(r"\.rf\.", re.IGNORECASE)
_FRAME_BUNCH_ID = re.compile(r"^(frame[^-]+)-", re.IGNORECASE)
_FRAME_BUNCH_ID_NO_HYPHEN = re.compile(r"^(frame[a-z0-9]+)", re.IGNORECASE)


def _parse_bunch_id(image_path: Path) -> str:
    stem_before_hash = _RF_HASH_SUFFIX.split(image_path.name, maxsplit=1)[0]
    m = _FRAME_BUNCH_ID.match(stem_before_hash)
    if m:
        return m.group(1)
    m = _FRAME_BUNCH_ID_NO_HYPHEN.match(stem_before_hash)
    if m:
        return m.group(1)
    return stem_before_hash  # fallback for any filename that isn't "frame...": treat as its own bunch


def _collect_pairs(dataset_dir: Path) -> list[tuple[Path, Path | None]]:
    """Find every (image, label) pair across Roboflow's train/valid/test dirs."""
    pairs = []
    for split_name in ("train", "valid", "test"):
        img_dir = dataset_dir / split_name / "images"
        lbl_dir = dataset_dir / split_name / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            pairs.append((img_path, lbl_path if lbl_path.exists() else None))
    return pairs


def _primary_class(label_path: Path | None) -> int:
    """Most frequent class among a label file's boxes, or -1 if none/background."""
    if label_path is None or not label_path.exists():
        return -1
    counts: Counter = Counter()
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if line:
            counts[int(line.split()[0])] += 1
    return counts.most_common(1)[0][0] if counts else -1


def audit_class_distribution(output_dir: Path, class_names: list[str], splits: tuple[str, ...] = ("train", "val", "test")) -> dict[str, dict[int, int]]:
    """Counts label instances per class per split and warns about classes
    that end up with zero (or very few) instances in any split.

    A prior rebuild of this exact dataset hit "Empty Bunch" at 0 instances
    in the val split after a naive (non-stratified) split -- Empty Bunch is
    the minority class dataset-wide (~20% of the largest class) -- so this
    is worth checking every time the split ratio or seed changes, even with
    stratified-by-dominant-class splitting below.
    """
    counts: dict[str, dict[int, int]] = {}
    for split_name in splits:
        labels_dir = output_dir / split_name / "labels"
        counter: Counter = Counter()
        if labels_dir.exists():
            for lbl_path in labels_dir.glob("*.txt"):
                for line in lbl_path.read_text().splitlines():
                    line = line.strip()
                    if line:
                        counter[int(line.split()[0])] += 1
        counts[split_name] = dict(counter)

    header = f"{'Class':<16}" + "".join(f"{s:>10}" for s in splits) + f"{'TOTAL':>10}"
    print("\nClass distribution per split:")
    print(header)
    print("-" * len(header))
    zero_warnings = []
    for class_id, name in enumerate(class_names):
        row = [counts[s].get(class_id, 0) for s in splits]
        total = sum(row)
        flag = ""
        if any(v == 0 for v in row):
            flag = "  <-- ZERO instances in a split!"
            zero_warnings.append(name)
        print(f"{name:<16}" + "".join(f"{v:>10}" for v in row) + f"{total:>10}{flag}")
    print("-" * len(header))
    if zero_warnings:
        print(f"WARNING: {', '.join(zero_warnings)} has/have zero instances in at least one "
              f"split -- consider a different split_seed, or check if these classes are just "
              f"genuinely rare dataset-wide before proceeding to Dirichlet partitioning.")
    return counts


def leakage_free_split(cfg_path: str = "configs/dataset.yaml") -> Path:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    dataset_dir = Path(cfg["download_dir"])
    output_dir = Path(cfg["output_dir"])
    ratios = cfg["split_ratios"]
    rng = random.Random(cfg["split_seed"])

    pairs = _collect_pairs(dataset_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No images found under {dataset_dir}/{{train,valid,test}}/images. "
            "Run scripts/01_download_dataset.py first."
        )

    bunches: dict[str, list[tuple[Path, Path | None]]] = defaultdict(list)
    for img_path, lbl_path in pairs:
        bunches[_parse_bunch_id(img_path)].append((img_path, lbl_path))

    # Stratify by each bunch's dominant class: split bunches of each class
    # separately by the target ratios, so a minority class (e.g. Empty
    # Bunch) doesn't get unluckily concentrated into one split by a plain
    # image-count-deficit greedy assignment.
    bunch_dominant_class: dict[str, int] = {
        bunch_id: Counter(_primary_class(lbl) for _, lbl in items).most_common(1)[0][0]
        for bunch_id, items in bunches.items()
    }
    bunches_by_class: dict[int, list[str]] = defaultdict(list)
    for bunch_id, cls in bunch_dominant_class.items():
        bunches_by_class[cls].append(bunch_id)

    split_names = list(ratios.keys())
    split_of_bunch: dict[str, str] = {}
    for cls, bunch_ids in bunches_by_class.items():
        bunch_ids = sorted(bunch_ids)  # deterministic order before shuffling
        rng.shuffle(bunch_ids)
        n = len(bunch_ids)
        cursor = 0
        remaining = dict(ratios)
        for i, split_name in enumerate(split_names):
            is_last = i == len(split_names) - 1
            count = n - cursor if is_last else round(n * remaining[split_name])
            for bunch_id in bunch_ids[cursor:cursor + count]:
                split_of_bunch[bunch_id] = split_name
            cursor += count

    # Wipe any previous split output first -- otherwise stale files from an
    # earlier run (e.g. before a bunch_id extraction fix, or a different
    # split_ratios/seed) linger alongside the new copies, and the bunch that
    # owns them appears to "leak" across splits when it's really just old +
    # new files coexisting under the same split_name directories.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split_name in ratios:
        (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)

    assigned = {name: 0 for name in ratios}
    for bunch_id, items in bunches.items():
        split_name = split_of_bunch[bunch_id]
        for img_path, lbl_path in items:
            dst_img = output_dir / split_name / "images" / img_path.name
            shutil.copy2(img_path, dst_img)
            dst_lbl = output_dir / split_name / "labels" / (img_path.stem + ".txt")
            if lbl_path is not None:
                shutil.copy2(lbl_path, dst_lbl)
            else:
                dst_lbl.write_text("")  # background image, no boxes
            assigned[split_name] += 1

    # leakage audit: assert no bunch_id appears in more than one split
    # (trivially true here since split_of_bunch assigns exactly one split
    # per bunch_id, but re-derive it from the written files to catch any
    # regression in the assignment/materialization logic above)
    seen = defaultdict(set)
    for split_name in ratios:
        for img_path in (output_dir / split_name / "images").glob("*"):
            seen[_parse_bunch_id(img_path)].add(split_name)
    leaked = {b: s for b, s in seen.items() if len(s) > 1}
    assert not leaked, f"Leakage detected for bunch_ids: {leaked}"

    total_images = sum(assigned.values())
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": cfg["nc"],
        "names": cfg["names"],
    }
    with open(output_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"{len(bunches)} bunches / {total_images} images split as:")
    for name in ratios:
        print(f"  {name}: {assigned[name]} images ({assigned[name] / total_images:.1%})")
    print(f"Leakage audit: 0 bunch_ids span multiple splits. Wrote {output_dir}/data.yaml")

    audit_class_distribution(output_dir, cfg["names"], tuple(ratios.keys()))
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--audit-only", action="store_true",
                         help="skip re-splitting, just re-run the class-distribution audit "
                              "on an existing data/splits/ (e.g. after manually editing files)")
    args = parser.parse_args()

    if args.audit_only:
        with open(args.config) as f:
            _cfg = yaml.safe_load(f)
        audit_class_distribution(Path(_cfg["output_dir"]), _cfg["names"], tuple(_cfg["split_ratios"].keys()))
    else:
        leakage_free_split(args.config)
