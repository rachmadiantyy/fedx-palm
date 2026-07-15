#!/usr/bin/env python3
"""Split-leakage & class-distribution audit for the materialized splits.

Produces results/split_audit.json + results/split_audit.md with, per split:
  - image count and bounding-box (annotation) count
  - unique bunch_id count
  - class instance distribution (all 6 classes)
and across splits:
  - bunch_ids appearing in more than one split (MUST be zero)
  - exact-duplicate files across splits (MD5 content hash; catches the same
    frame or an identical augmented copy straddling two splits even when the
    filename differs)

Exit code is non-zero if any leakage is found, so this can gate a training
run in a shell script: `python scripts/11_audit_split.py && python scripts/06_...`
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

from fedxpalm.data.split import _parse_bunch_id  # noqa: E402


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--out", default="results/split_audit")
    parser.add_argument("--skip-hash", action="store_true",
                        help="skip the MD5 duplicate scan (faster; bunch audit still runs)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    splits_dir = Path(cfg["output_dir"])
    names = cfg["names"]
    splits = ("train", "val", "test")

    report = {"splits_dir": str(splits_dir), "splits": {}, "leakage": {}}
    bunch_to_splits: dict[str, set] = defaultdict(set)
    hash_to_locations: dict[str, list[str]] = defaultdict(list)

    for split in splits:
        img_dir = splits_dir / split / "images"
        lbl_dir = splits_dir / split / "labels"
        if not img_dir.is_dir():
            print(f"[!] missing split dir: {img_dir}")
            continue

        images = [p for p in sorted(img_dir.glob("*"))
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        class_counts: Counter = Counter()
        n_boxes = 0
        for img in images:
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    line = line.strip()
                    if line:
                        class_counts[int(line.split()[0])] += 1
                        n_boxes += 1
            bunch_to_splits[_parse_bunch_id(img)].add(split)
            if not args.skip_hash:
                hash_to_locations[md5_of(img)].append(f"{split}/{img.name}")

        bunches_here = {b for b, s in bunch_to_splits.items() if split in s}
        report["splits"][split] = {
            "images": len(images),
            "boxes": n_boxes,
            "unique_bunches": len(bunches_here),
            "class_boxes": {names[c]: class_counts.get(c, 0) for c in range(len(names))},
            "classes_with_zero": [names[c] for c in range(len(names)) if class_counts.get(c, 0) == 0],
        }

    cross_bunches = {b: sorted(s) for b, s in bunch_to_splits.items() if len(s) > 1}
    dup_files = ({h: locs for h, locs in hash_to_locations.items()
                  if len({loc.split("/", 1)[0] for loc in locs}) > 1}
                 if not args.skip_hash else None)
    report["leakage"] = {
        "bunches_in_multiple_splits": cross_bunches,
        "duplicate_files_across_splits": dup_files,
        "passed": not cross_bunches and not dup_files,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(report, f, indent=2)

    lines = ["# Split audit\n", f"Splits dir: `{splits_dir}`\n",
             "| Split | Images | Boxes | Unique bunches |" + "".join(f" {n} |" for n in names),
             "|---|---|---|---|" + "---|" * len(names)]
    for split, s in report["splits"].items():
        lines.append(f"| {split} | {s['images']} | {s['boxes']} | {s['unique_bunches']} |"
                     + "".join(f" {s['class_boxes'][n]} |" for n in names))
    lines.append("")
    lines.append(f"- Bunch_ids in >1 split: **{len(cross_bunches)}** (must be 0)")
    if dup_files is not None:
        lines.append(f"- Exact-duplicate files across splits: **{len(dup_files)}** (must be 0)")
    lines.append(f"- **Leakage audit: {'PASSED' if report['leakage']['passed'] else 'FAILED'}**")
    for split, s in report["splits"].items():
        if s["classes_with_zero"]:
            lines.append(f"- WARNING: {split} has ZERO instances of: {', '.join(s['classes_with_zero'])}")
    with open(out.with_suffix(".md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nSaved {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0 if report["leakage"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
