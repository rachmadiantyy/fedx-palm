#!/usr/bin/env python3
"""Split-leakage & class-distribution audit for the materialized splits.

Produces results/split_audit.json + results/split_audit.md with, per split:
  - image count and bounding-box (annotation) count
  - unique bunch_id count + images-per-bunch stats
  - class instance distribution (all 6 classes)
and across splits:
  - bunch_ids appearing in more than one split (MUST be zero)
  - exact-duplicate files across splits (MD5 content hash)
  - NEAR-duplicate images across splits (perceptual dHash, Hamming <= N):
    catches re-encoded/resized/re-augmented copies of the same frame that
    MD5 misses. Candidate pairs come from 16-bit LSH bands (pigeonhole: any
    pair within Hamming distance 3 of a 64-bit hash shares at least one of
    four 16-bit bands), then get verified by exact Hamming distance.
  - sample filename -> bunch_id mappings, to eyeball-verify the parser

Exit code is non-zero if any leakage is found, so this can gate a training
run in a shell script: `python scripts/11_audit_split.py && python scripts/06_...`
"""
import argparse
import hashlib
import importlib.util
import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import yaml

# Load split.py directly (not via `import fedxpalm...`): the package __init__
# imports torch for the GroupNorm fuse patch, which a pure data audit doesn't
# need -- this keeps the audit runnable on machines without torch installed.
_split_path = Path(__file__).resolve().parent.parent / "src" / "fedxpalm" / "data" / "split.py"
_spec = importlib.util.spec_from_file_location("_fedx_split", _split_path)
_split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_split)
_parse_bunch_id = _split._parse_bunch_id
_parse_source_frame = _split._parse_source_frame
load_source_aliases = _split.load_source_aliases


def load_review_decisions(path: Path = Path("data/source_alias_review.csv")) -> dict[tuple[str, str], str]:
    """(source_a, source_b) -> decision, from the human review file."""
    import csv
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {(r["source_a"].strip(), r["source_b"].strip()): (r.get("decision") or "uncertain").strip()
                for r in csv.DictReader(f)}


def classify_candidates(near_dupes: list[dict], decisions: dict) -> dict:
    """Split raw dHash candidates into confirmed / reviewed-false-positive /
    uncertain, by the review decision of their source-prefix pair. Raw
    candidates whose pair was reviewed as different_source do NOT gate the
    audit; anything unreviewed does."""
    confirmed, false_pos, uncertain = [], [], []
    for c in near_dupes:
        src_a, _ = _parse_source_frame(c["a"].split("/", 1)[1])
        src_b, _ = _parse_source_frame(c["b"].split("/", 1)[1])
        key = (src_a, src_b) if (src_a, src_b) in decisions else (src_b, src_a)
        decision = decisions.get(key, "unreviewed")
        if decision == "confirmed_alias":
            confirmed.append(c)
        elif decision == "different_source":
            false_pos.append(c)
        else:
            uncertain.append({**c, "decision": decision})
    return {"confirmed_cross_split_duplicates": confirmed,
            "reviewed_false_positives": false_pos,
            "uncertain_or_unreviewed": uncertain}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash64(path: Path):
    """64-bit difference hash (row-wise gradient sign on a 9x8 grayscale
    thumbnail). Robust to re-encoding/resizing; returns None if unreadable."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((9, 8), Image.LANCZOS)
            px = im.tobytes()  # 72 bytes, one per pixel ("L" mode); indexing yields ints
    except Exception:
        return None
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (1 if px[row * 9 + col] > px[row * 9 + col + 1] else 0)
    return bits


def near_duplicates(phash_index: list[tuple[int, str, str]], max_hamming: int) -> list[dict]:
    """phash_index: [(hash, split, filename)]. Returns cross-split pairs with
    Hamming(hash_a, hash_b) <= max_hamming. LSH banding keeps this O(n) in
    practice; exact for max_hamming <= 3 (4 bands of 16 bits, pigeonhole)."""
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, (h, _split, _name) in enumerate(phash_index):
        for band in range(4):
            buckets[(band, (h >> (16 * band)) & 0xFFFF)].append(idx)

    seen_pairs, out = set(), []
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if (a, b) in seen_pairs:
                    continue
                seen_pairs.add((a, b))
                ha, sa, na = phash_index[a]
                hb, sb, nb = phash_index[b]
                if sa == sb:
                    continue  # same split: not leakage
                dist = bin(ha ^ hb).count("1")
                if dist <= max_hamming:
                    out.append({"a": f"{sa}/{na}", "b": f"{sb}/{nb}", "hamming": dist})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--out", default="results/split_audit")
    parser.add_argument("--hamming", type=int, default=3,
                        help="near-duplicate threshold on 64-bit dHash (exact detection for <= 3)")
    parser.add_argument("--skip-hash", action="store_true",
                        help="skip the MD5 duplicate scan")
    parser.add_argument("--skip-phash", action="store_true",
                        help="skip the perceptual near-duplicate scan")
    parser.add_argument("--mapping-samples", type=int, default=12,
                        help="how many filename->bunch_id examples to show per split")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    splits_dir = Path(cfg["output_dir"])
    names = cfg["names"]
    splits = ("train", "val", "test")

    try:
        import PIL  # noqa: F401
        pil_ok = True
    except ImportError:
        pil_ok = False
        if not args.skip_phash:
            print("[!] Pillow not installed -- skipping perceptual near-duplicate scan "
                  "(pip install pillow to enable)")

    aliases = load_source_aliases()
    decisions = load_review_decisions()
    if aliases:
        print(f"[aliases] {len(aliases)} source->group mapping(s) loaded "
              f"({len(set(aliases.values()))} groups)")
    if decisions:
        print(f"[review] {len(decisions)} reviewed source pair(s) loaded")

    report = {"splits_dir": str(splits_dir), "splits": {}, "leakage": {}, "bunch_id_mapping_samples": {}}
    bunch_to_splits: dict[str, set] = defaultdict(set)
    group_to_splits: dict[str, set] = defaultdict(set)
    hash_to_locations: dict[str, list[str]] = defaultdict(list)
    phash_index: list[tuple[int, str, str]] = []
    rng = random.Random(0)

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
        bunch_sizes: Counter = Counter()
        for img in images:
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    line = line.strip()
                    if line:
                        class_counts[int(line.split()[0])] += 1
                        n_boxes += 1
            bunch = _parse_bunch_id(img)
            bunch_to_splits[bunch].add(split)
            group_to_splits[aliases.get(bunch, bunch)].add(split)
            bunch_sizes[bunch] += 1
            if not args.skip_hash:
                hash_to_locations[md5_of(img)].append(f"{split}/{img.name}")
            if pil_ok and not args.skip_phash:
                h = dhash64(img)
                if h is not None:
                    phash_index.append((h, split, img.name))

        per_bunch = sorted(bunch_sizes.values())
        report["splits"][split] = {
            "images": len(images),
            "boxes": n_boxes,
            "unique_bunches": len(bunch_sizes),
            "images_per_bunch": ({"min": per_bunch[0], "median": st.median(per_bunch),
                                  "max": per_bunch[-1]} if per_bunch else {}),
            "class_boxes": {names[c]: class_counts.get(c, 0) for c in range(len(names))},
            "classes_with_zero": [names[c] for c in range(len(names)) if class_counts.get(c, 0) == 0],
        }
        # eyeball-verification samples for the bunch_id parser
        sample = rng.sample(images, min(args.mapping_samples, len(images)))
        report["bunch_id_mapping_samples"][split] = {p.name: _parse_bunch_id(p) for p in sample}

    cross_bunches = {b: sorted(s) for b, s in bunch_to_splits.items() if len(s) > 1}
    cross_groups = {g: sorted(s) for g, s in group_to_splits.items() if len(s) > 1}
    dup_files = ({h: locs for h, locs in hash_to_locations.items()
                  if len({loc.split("/", 1)[0] for loc in locs}) > 1}
                 if not args.skip_hash else None)
    near_dupes = (near_duplicates(phash_index, args.hamming)
                  if pil_ok and not args.skip_phash else None)

    # Raw dHash candidates are NOT a pass/fail gate by themselves: each
    # candidate is classified by the human review decision of its source
    # pair. Reviewed false positives (different_source) don't gate; anything
    # confirmed as an alias that still straddles splits, or not yet
    # reviewed, does.
    classified = classify_candidates(near_dupes or [], decisions)

    passed = (not cross_groups
              and not cross_bunches
              and not dup_files
              and not classified["confirmed_cross_split_duplicates"]
              and not classified["uncertain_or_unreviewed"])
    report["leakage"] = {
        "bunches_in_multiple_splits": cross_bunches,
        "source_groups_in_multiple_splits": cross_groups,
        "duplicate_files_across_splits": dup_files,
        "raw_dhash_candidates": near_dupes,
        "raw_dhash_candidate_count": len(near_dupes) if near_dupes is not None else None,
        "confirmed_cross_split_duplicates": classified["confirmed_cross_split_duplicates"],
        "reviewed_false_positives_count": len(classified["reviewed_false_positives"]),
        "uncertain_or_unreviewed": classified["uncertain_or_unreviewed"],
        "phash_hamming_threshold": args.hamming if near_dupes is not None else None,
        "passed": passed,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(report, f, indent=2)

    lines = ["# Split audit\n", f"Splits dir: `{splits_dir}`\n",
             "| Split | Images | Boxes | Unique bunches | Img/bunch (min/med/max) |"
             + "".join(f" {n} |" for n in names),
             "|---|---|---|---|---|" + "---|" * len(names)]
    for split, s in report["splits"].items():
        ipb = s["images_per_bunch"]
        ipb_str = f"{ipb.get('min','-')}/{ipb.get('median','-')}/{ipb.get('max','-')}"
        lines.append(f"| {split} | {s['images']} | {s['boxes']} | {s['unique_bunches']} | {ipb_str} |"
                     + "".join(f" {s['class_boxes'][n]} |" for n in names))
    lines.append("")
    lines.append(f"- Bunch_ids in >1 split: **{len(cross_bunches)}** (must be 0)")
    lines.append(f"- Source groups (alias-aware) in >1 split: **{len(cross_groups)}** (must be 0)")
    for g, s in list(cross_groups.items())[:10]:
        lines.append(f"    - group `{g}` spans {s}")
    if dup_files is not None:
        lines.append(f"- Exact-duplicate files across splits (MD5): **{len(dup_files)}** (must be 0)")
    if near_dupes is not None:
        conf = classified["confirmed_cross_split_duplicates"]
        unrev = classified["uncertain_or_unreviewed"]
        lines.append(f"- Raw dHash candidates across splits (Hamming<={args.hamming}): "
                     f"**{len(near_dupes)}** (informational -- gated via review below)")
        lines.append(f"    - confirmed cross-split duplicates: **{len(conf)}** (must be 0)")
        lines.append(f"    - reviewed false positives (different_source): "
                     f"{len(classified['reviewed_false_positives'])} (do not gate)")
        lines.append(f"    - uncertain / unreviewed: **{len(unrev)}** (must be 0 -- "
                     f"review via scripts/13 + data/source_alias_review.csv)")
        for pair in (conf + unrev)[:20]:
            lines.append(f"        - {pair['a']}  <->  {pair['b']}  (hamming {pair['hamming']})")
    lines.append(f"- **Leakage audit: {'PASSED' if report['leakage']['passed'] else 'FAILED'}**")
    for split, s in report["splits"].items():
        if s["classes_with_zero"]:
            lines.append(f"- WARNING: {split} has ZERO instances of: {', '.join(s['classes_with_zero'])}")
    lines.append("\n## Sample filename -> bunch_id mappings (parser sanity check)\n")
    for split, mapping in report["bunch_id_mapping_samples"].items():
        lines.append(f"**{split}:**")
        for fn, bid in mapping.items():
            lines.append(f"- `{fn}` -> `{bid}`")
        lines.append("")
    with open(out.with_suffix(".md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nSaved {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0 if report["leakage"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
