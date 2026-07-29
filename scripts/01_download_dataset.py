#!/usr/bin/env python3
"""Step 1: download the Roboflow dataset. Run this on a machine with network
access to api.roboflow.com (this repo's own dev sandbox may not have it)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedxpalm.data.download import download_dataset  # noqa: E402

if __name__ == "__main__":
    download_dataset("configs/dataset.yaml")
