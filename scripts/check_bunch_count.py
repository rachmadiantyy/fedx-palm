#!/usr/bin/env python3
"""One-off diagnostic: verify the real bunch count on data/raw using the
FIXED bunch_id extraction (src/fedxpalm/data/split.py), before committing
to the full leakage_free_split() run. Not part of the numbered pipeline --
delete or ignore once the dataset is frozen."""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedxpalm.data.split import _parse_bunch_id  # noqa: E402

if __name__ == "__main__":
    raw_dir = Path("data/raw")
    images = []
    for split in ("train", "valid", "test"):
        images.extend((raw_dir / split / "images").glob("*"))

    groups: dict[str, list[Path]] = defaultdict(list)
    split_of_image: dict[Path, str] = {}
    for split in ("train", "valid", "test"):
        for img in (raw_dir / split / "images").glob("*"):
            split_of_image[img] = split

    for img in images:
        groups[_parse_bunch_id(img)].append(img)

    sizes = sorted(len(v) for v in groups.values())
    print(f"total images:  {len(images)}")
    print(f"true bunches:  {len(groups)}")
    print(f"bunch size min/median/max: {sizes[0]} / {sizes[len(sizes)//2]} / {sizes[-1]}")
    print(f"avg images/bunch: {len(images) / len(groups):.1f}")

    # leakage check on Roboflow's OWN original train/valid/test (expected to leak --
    # this is exactly the problem leakage_free_split() fixes)
    bunch_splits: dict[str, set] = defaultdict(set)
    for img, split in split_of_image.items():
        bunch_splits[_parse_bunch_id(img)].add(split)
    leaked = {b: s for b, s in bunch_splits.items() if len(s) > 1}
    print(f"\nbunches spanning >1 of Roboflow's OWN original splits: {len(leaked)} "
          f"(out of {len(groups)}) -- this is the leakage scripts/02 fixes, "
          f"a non-zero number here is EXPECTED and not a bug.")
