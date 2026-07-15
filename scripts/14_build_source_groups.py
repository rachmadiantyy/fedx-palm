#!/usr/bin/env python3
"""Build the source-group alias mapping from HUMAN-CONFIRMED review decisions.

Reads data/source_alias_review.csv (decision column: confirmed_alias /
different_source / uncertain), union-finds the confirmed_alias pairs into
connected components (A~B, B~C => {A,B,C} one source group), and writes
data/source_group_aliases.json:

    {
      "source_to_group": {"framesawit39": "frame1", ...},   # member -> representative
      "groups": {"frame1": ["frame1", "framesawit39"], ...},
      "stats": {...}
    }

Only confirmed_alias rows contribute -- uncertain and different_source rows
never merge anything. The representative is the lexicographically smallest
member, so the mapping is deterministic across machines.

--estimate-split then previews the regenerated split sizes (dry run, no file
touched) with the mapping applied.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import importlib.util as _ilu

_split_path = Path(__file__).resolve().parent.parent / "src" / "fedxpalm" / "data" / "split.py"
_spec = _ilu.spec_from_file_location("_fedx_split", _split_path)
_split = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_split)

REVIEW_CSV = Path("data/source_alias_review.csv")
ALIASES_JSON = Path("data/source_group_aliases.json")


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic: smaller string becomes the root
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", default=str(REVIEW_CSV))
    parser.add_argument("--out", default=str(ALIASES_JSON))
    parser.add_argument("--estimate-split", action="store_true",
                        help="also dry-run the regenerated split with the new mapping (no files written)")
    parser.add_argument("--config", default="configs/dataset.yaml")
    args = parser.parse_args()

    review_path = Path(args.review)
    if not review_path.exists():
        raise SystemExit(f"{review_path} not found -- run scripts/13_group_near_duplicates.py first")

    with open(review_path, newline="") as f:
        rows = list(csv.DictReader(f))

    counts = defaultdict(int)
    uf = UnionFind()
    confirmed_pairs = []
    for r in rows:
        decision = (r.get("decision") or "uncertain").strip()
        counts[decision] += 1
        if decision == "confirmed_alias":
            uf.union(r["source_a"].strip(), r["source_b"].strip())
            confirmed_pairs.append((r["source_a"].strip(), r["source_b"].strip()))
        elif decision not in ("different_source", "uncertain"):
            print(f"[!] unknown decision '{decision}' for ({r['source_a']}, {r['source_b']}) "
                  f"-- treated as uncertain")

    groups: dict[str, list[str]] = defaultdict(list)
    for member in list(uf.parent):
        groups[uf.find(member)].append(member)
    groups = {rep: sorted(members) for rep, members in groups.items()}
    source_to_group = {m: rep for rep, members in groups.items() for m in members if m != rep}

    print(f"Review rows: {sum(counts.values())} "
          f"({counts['confirmed_alias']} confirmed_alias, "
          f"{counts['different_source']} different_source, "
          f"{counts['uncertain']} uncertain)")
    if counts["uncertain"]:
        print(f"[!] {counts['uncertain']} pair(s) still uncertain -- the final audit "
              f"(scripts/11) will FAIL until every pair is decided.")
    if not confirmed_pairs:
        print("No confirmed_alias pairs -- no mapping written (nothing to merge).")
        if args.estimate_split:
            _split.leakage_free_split(args.config, dry_run=True)
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_to_group": source_to_group,
        "groups": groups,
        "stats": {
            "confirmed_pairs": len(confirmed_pairs),
            "groups": len(groups),
            "largest_group": max(len(m) for m in groups.values()),
            "review_file": str(review_path),
        },
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"\nSource groups ({len(groups)}):")
    for rep, members in sorted(groups.items()):
        print(f"  {rep}: {', '.join(members)}")
    print(f"Saved {out} ({len(source_to_group)} alias mapping(s))")

    if args.estimate_split:
        print("\n--- estimated regenerated split (dry run, nothing written) ---")
        _split.leakage_free_split(args.config, dry_run=True, aliases_path=out)


if __name__ == "__main__":
    main()
