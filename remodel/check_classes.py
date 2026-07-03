"""
Cek distribusi kelas per split (train/valid/test) dari file label YOLO.

Berguna untuk memastikan tidak ada kelas yang hilang di suatu split
(mis. 'Empty Bunch' yang kemarin 0 instance di val).

    python remodel/check_classes.py --data data/resplit/data.yaml
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

import yaml

CLASS_NAMES = ["Abnormal", "Empty Bunch", "Overripe", "Ripe", "Underripe", "Unripe"]


def labels_dir_from_images(images_dir: str) -> str:
    # Roboflow: .../<split>/images  -> .../<split>/labels
    return images_dir.replace(os.sep + "images", os.sep + "labels")


def count_split(images_dir: str) -> Counter:
    counter: Counter = Counter()
    labels_dir = labels_dir_from_images(images_dir)
    if not os.path.isdir(labels_dir):
        print(f"  [!] folder label tidak ada: {labels_dir}")
        return counter
    for fn in os.listdir(labels_dir):
        if not fn.endswith(".txt"):
            continue
        with open(os.path.join(labels_dir, fn)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls_id = int(line.split()[0])
                counter[cls_id] += 1
    return counter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path data.yaml")
    args = ap.parse_args()

    with open(args.data) as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", CLASS_NAMES)
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    splits = {}
    for key in ("train", "val", "test"):
        d = cfg.get(key)
        if d:
            splits[key] = count_split(d)

    print(f"\n{'Kelas':<14}" + "".join(f"{k:>10}" for k in splits) + f"{'TOTAL':>10}")
    print("-" * (14 + 10 * (len(splits) + 1)))
    for i, cn in enumerate(names):
        row = [splits[k].get(i, 0) for k in splits]
        total = sum(row)
        flag = "   <-- KOSONG di suatu split!" if any(v == 0 for v in row) else ""
        print(f"{cn:<14}" + "".join(f"{v:>10}" for v in row) + f"{total:>10}{flag}")
    print("-" * (14 + 10 * (len(splits) + 1)))


if __name__ == "__main__":
    main()
