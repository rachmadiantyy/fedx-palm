#!/usr/bin/env python3
"""Step 2: copy Roboflow's own train/valid/test split as-is (no bunch-level re-split)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedxpalm.data.split import use_roboflow_split  # noqa: E402

if __name__ == "__main__":
    use_roboflow_split("configs/dataset.yaml")
