#!/usr/bin/env python3
"""Step 2: leakage-free train/val/test split, grouped by source_group.

The grouping key is the bunch_id run through the confirmed source-alias
mapping (data/source_group_aliases.json, built by scripts/14 from the
human-reviewed data/source_alias_review.csv) when that file exists --
so two filename prefixes confirmed to be the same physical source can
never straddle two splits. Without the mapping file this is the plain
bunch_id-grouped split (prior behavior).

--dry-run previews the resulting split sizes + class distribution without
touching any file. Regenerating for real REPLACES data/splits_v2/ -- run
the dry run and scripts/11_audit_split.py's review workflow first."""
import argparse
import importlib.util
from pathlib import Path

# Load split.py directly (not via `import fedxpalm...`): the package __init__
# imports torch for the GroupNorm fuse patch, which pure data prep doesn't
# need -- keeps this runnable on machines without torch installed.
_split_path = Path(__file__).resolve().parent.parent / "src" / "fedxpalm" / "data" / "split.py"
_spec = importlib.util.spec_from_file_location("_fedx_split", _split_path)
_split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_split)
leakage_free_split = _split.leakage_free_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="preview assignment sizes/class distribution; write nothing")
    args = parser.parse_args()
    leakage_free_split(args.config, dry_run=args.dry_run)
