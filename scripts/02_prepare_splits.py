#!/usr/bin/env python3
"""Step 2: leakage-free (bunch_id-grouped) train/val/test split."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedxpalm.data.split import leakage_free_split  # noqa: E402

if __name__ == "__main__":
    leakage_free_split("configs/dataset.yaml")
