"""Use Roboflow's own train/valid/test split as-is -- no bunch-level
re-splitting.

An earlier version of this module re-pooled every image across Roboflow's
splits and re-split them at a `bunch_id` (physical FFB identity) level to
guard against near-duplicate video frames straddling train/test. That is
methodologically more rigorous, but this dataset's related work (and this
project's own earlier benchmark run) reports results computed directly on
Roboflow's own exported split, so this version matches that: copy
`train/valid/test` from the Roboflow download as-is into `output_dir`,
renaming `valid` -> `val` for consistency with the rest of this repo's
scripts, and skip any bunch-identity grouping.
"""
from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import yaml


def _copy_split(src_dir: Path, dst_dir: Path) -> int:
    img_src, lbl_src = src_dir / "images", src_dir / "labels"
    img_dst, lbl_dst = dst_dir / "images", dst_dir / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    n = 0
    for img_path in sorted(img_src.glob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        shutil.copy2(img_path, img_dst / img_path.name)
        lbl_path = lbl_src / (img_path.stem + ".txt")
        dst_lbl = lbl_dst / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.copy2(lbl_path, dst_lbl)
        else:
            dst_lbl.write_text("")  # background image, no boxes
        n += 1
    return n


def audit_class_distribution(output_dir: Path, class_names: list[str], splits: tuple[str, ...] = ("train", "val", "test")) -> dict[str, dict[int, int]]:
    """Counts label instances per class per split and warns about classes
    that end up with zero (or very few) instances in any split.

    A prior rebuild of this exact dataset hit "Empty Bunch" at 0 instances
    in the val split -- Empty Bunch is the minority class dataset-wide
    (~20% of the largest class) -- so this is worth checking whenever the
    dataset source or version changes, even though the split itself is now
    just Roboflow's own (not something this repo controls).
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
              f"split -- check if these classes are just genuinely rare in Roboflow's split "
              f"before proceeding to Dirichlet partitioning.")
    return counts


def use_roboflow_split(cfg_path: str = "configs/dataset.yaml") -> Path:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    dataset_dir = Path(cfg["download_dir"])
    output_dir = Path(cfg["output_dir"])

    if not (dataset_dir / "train" / "images").exists():
        raise FileNotFoundError(
            f"No images found under {dataset_dir}/train/images. "
            "Run scripts/01_download_dataset.py first."
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)

    roboflow_to_internal = {"train": "train", "valid": "val", "test": "test"}
    counts = {}
    for rf_name, internal_name in roboflow_to_internal.items():
        counts[internal_name] = _copy_split(dataset_dir / rf_name, output_dir / internal_name)

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

    total_images = sum(counts.values())
    print(f"{total_images} images copied from Roboflow's own split as-is:")
    for name, n in counts.items():
        print(f"  {name}: {n} images ({n / total_images:.1%})")
    print(f"Wrote {output_dir}/data.yaml")

    audit_class_distribution(output_dir, cfg["names"])
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--audit-only", action="store_true",
                         help="skip re-copying, just re-run the class-distribution audit "
                              "on an existing data/splits/ (e.g. after manually editing files)")
    args = parser.parse_args()

    if args.audit_only:
        with open(args.config) as f:
            _cfg = yaml.safe_load(f)
        audit_class_distribution(Path(_cfg["output_dir"]), _cfg["names"])
    else:
        use_roboflow_split(args.config)
