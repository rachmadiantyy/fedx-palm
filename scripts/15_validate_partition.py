#!/usr/bin/env python3
"""Validate a K-client Dirichlet partition BEFORE any federated training.

Checks (all recomputed from disk, not trusted from the manifest):
  1. every client's stems exist under the train split
  2. no image stem appears in more than one client
  3. the union of all clients == the ENTIRE train split (count + set equality)
  4. no client is empty / suspiciously small
  5. per-client: images, unique bunches (alias-aware), boxes per class,
     missing classes (warned; fails only if a client has zero boxes at all)

Exit code non-zero on any failure, so this can gate training:
    python scripts/15_validate_partition.py --k 4 --seed 42 && python scripts/06_...
"""
import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import yaml

_split_path = Path(__file__).resolve().parent.parent / "src" / "fedxpalm" / "data" / "split.py"
_spec = importlib.util.spec_from_file_location("_fedx_split", _split_path)
_split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_split)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42, help="partition seed (root resolution matches scripts/06)")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--fl-config", default="configs/fl_config.yaml")
    parser.add_argument("--min-client-frac", type=float, default=0.02,
                        help="fail if any client holds under this fraction of train images")
    args = parser.parse_args()

    with open(args.config) as f:
        ds_cfg = yaml.safe_load(f)
    with open(args.fl_config) as f:
        fl_cfg = yaml.safe_load(f)

    splits_dir = Path(ds_cfg["output_dir"])
    names = ds_cfg["names"]
    nc = ds_cfg["nc"]
    cfg_seed = fl_cfg["clients"]["partition_seed"]
    root = "federated_partitions" if args.seed == cfg_seed else f"federated_partitions_seed{args.seed}"
    part_json = splits_dir / root / f"k{args.k}.json"
    if not part_json.exists():
        print(f"FAIL: {part_json} not found -- run scripts/03_partition_clients.py "
              f"--k {args.k}" + ("" if args.seed == cfg_seed else f" --seed {args.seed}"))
        return 1

    with open(part_json) as f:
        assignment: dict[str, list[str]] = json.load(f)

    train_images = splits_dir / "train" / "images"
    train_labels = splits_dir / "train" / "labels"
    train_stems = {p.stem for p in train_images.glob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    aliases = _split.load_source_aliases()

    failures, warnings = [], []

    # 2) overlap between clients
    seen: dict[str, str] = {}
    overlaps = []
    for cid, stems in assignment.items():
        for s in stems:
            if s in seen:
                overlaps.append((s, seen[s], cid))
            seen[s] = cid
    if overlaps:
        failures.append(f"{len(overlaps)} stem(s) assigned to more than one client "
                        f"(e.g. {overlaps[0][0]} -> clients {overlaps[0][1]} & {overlaps[0][2]})")

    # 1+3) coverage vs train split
    all_stems = set(seen)
    missing_on_disk = all_stems - train_stems
    uncovered = train_stems - all_stems
    if missing_on_disk:
        failures.append(f"{len(missing_on_disk)} assigned stem(s) have no file under {train_images} "
                        f"(e.g. {sorted(missing_on_disk)[:3]}) -- partition predates the current split? "
                        f"Regenerate with scripts/03.")
    if uncovered:
        failures.append(f"{len(uncovered)} train image(s) belong to NO client "
                        f"(e.g. {sorted(uncovered)[:3]})")
    total_assigned = sum(len(s) for s in assignment.values())
    print(f"Train split: {len(train_stems)} images | assigned to {args.k} clients: {total_assigned}")
    if total_assigned != len(train_stems):
        failures.append(f"total assigned ({total_assigned}) != train images ({len(train_stems)})")

    # 4+5) per-client stats, recomputed from label files
    print(f"\n{'client':<8}{'images':>8}{'bunches':>9}{'groups':>8}{'boxes':>8}  "
          + "".join(f"{n[:9]:>10}" for n in names) + "  missing")
    min_images = max(1, int(args.min_client_frac * len(train_stems)))
    for cid in sorted(assignment, key=lambda c: int(c) if str(c).isdigit() else c):
        stems = assignment[cid]
        class_counts: Counter = Counter()
        bunches, groups = set(), set()
        for s in stems:
            bunch = _split._parse_bunch_id(Path(s + ".jpg"))
            bunches.add(bunch)
            groups.add(aliases.get(bunch, bunch))
            lbl = train_labels / f"{s}.txt"
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    line = line.strip()
                    if line:
                        class_counts[int(line.split()[0])] += 1
        missing = [names[c] for c in range(nc) if class_counts.get(c, 0) == 0]
        n_boxes = sum(class_counts.values())
        print(f"{cid:<8}{len(stems):>8}{len(bunches):>9}{len(groups):>8}{n_boxes:>8}  "
              + "".join(f"{class_counts.get(c, 0):>10}" for c in range(nc))
              + ("  " + ",".join(missing) if missing else "  -"))
        if not stems:
            failures.append(f"client {cid} is EMPTY")
        elif len(stems) < min_images:
            warnings.append(f"client {cid} holds only {len(stems)} images "
                            f"(<{args.min_client_frac:.0%} of train) -- alpha may be too extreme")
        if n_boxes == 0 and stems:
            failures.append(f"client {cid} has images but ZERO boxes")
        if missing:
            warnings.append(f"client {cid} missing class(es): {', '.join(missing)}")

    print()
    for w in warnings:
        print(f"WARNING: {w}")
    for msg in failures:
        print(f"FAIL: {msg}")
    if failures:
        print("\nPartition INVALID -- do NOT train on this.")
        return 1
    print(f"\nPartition VALID: {total_assigned}/{len(train_stems)} images covered, "
          f"no overlap, no empty client. ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
