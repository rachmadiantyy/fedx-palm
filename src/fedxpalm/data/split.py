"""Leakage-free train/val/test split, grouped by bunch (FFB) identity.

Roboflow's own train/valid/test split is per-frame and random, so multiple
photos of the *same physical bunch* can end up in both train and test --
the model then partly "memorizes" a bunch it was tested on. This module
re-pools every image across Roboflow's splits, groups them by bunch_id
parsed from the filename, and re-splits at the bunch level so no bunch_id
appears in more than one split.
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def _parse_bunch_id(image_path: Path, pattern: re.Pattern) -> str:
    m = pattern.match(image_path.name)
    if m and "bunch_id" in m.groupdict():
        return m.group("bunch_id")
    return image_path.stem  # fallback: treat every image as its own bunch


def _collect_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    """Find every (image, label) pair across Roboflow's train/valid/test dirs."""
    pairs = []
    for split_name in ("train", "valid", "test", "val"):
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


def audit_class_distribution(output_dir: Path, class_names: list[str], splits: tuple[str, ...] = ("train", "val", "test")) -> dict[str, dict[int, int]]:
    """Counts label instances per class per split and warns about classes
    that end up with zero (or very few) instances in any split.

    A prior rebuild of this exact dataset hit "Empty Bunch" at 0 instances
    in the val split after a naive split -- Empty Bunch is the minority
    class dataset-wide (~20% of the largest class), so this is a real risk
    worth checking every time the split ratio or seed changes, not just
    trusting the bunch-level greedy assignment to balance classes too.
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
    pattern = re.compile(cfg["bunch_id_regex"], re.IGNORECASE)
    rng = random.Random(cfg["split_seed"])

    pairs = _collect_pairs(dataset_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No images found under {dataset_dir}/{{train,valid,test}}/images. "
            "Run scripts/01_download_dataset.py first."
        )

    bunches: dict[str, list[tuple[Path, Path | None]]] = defaultdict(list)
    for img_path, lbl_path in pairs:
        bunch_id = _parse_bunch_id(img_path, pattern)
        bunches[bunch_id].append((img_path, lbl_path))

    bunch_ids = list(bunches.keys())
    rng.shuffle(bunch_ids)

    total_images = len(pairs)
    targets = {name: ratio * total_images for name, ratio in ratios.items()}
    assigned = {name: 0 for name in ratios}
    split_of_bunch: dict[str, str] = {}

    for bunch_id in bunch_ids:
        n = len(bunches[bunch_id])
        # assign to whichever split is furthest below its target share
        deficit = {name: targets[name] - assigned[name] for name in ratios}
        best_split = max(deficit, key=deficit.get)
        split_of_bunch[bunch_id] = best_split
        assigned[best_split] += n

    for split_name in ratios:
        (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)

    audit_rows = []
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
        audit_rows.append((bunch_id, split_name, len(items)))

    # leakage audit: assert no bunch_id appears in more than one split
    seen = defaultdict(set)
    for bunch_id, split_name, _ in audit_rows:
        seen[bunch_id].add(split_name)
    leaked = {b: s for b, s in seen.items() if len(s) > 1}
    assert not leaked, f"Leakage detected for bunch_ids: {leaked}"

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

    print(f"{len(bunch_ids)} bunches / {total_images} images split as:")
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
