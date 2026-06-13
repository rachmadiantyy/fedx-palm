"""
[00] Download Palm Fruit Ripeness dataset from Roboflow — FRESH.

Step 0 of the from-scratch rebuild pipeline. Downloads the raw dataset
exactly as Roboflow ships it (random per-frame train/valid/test split).
The leakage fix happens in the NEXT step (01_resplit_bunch_id.py).

Source: dydy-worker/palm-fruit-ripeness-detection-f6sac-ccb2z, version 2,
        YOLOv11 format.

Usage:
    # API key via flag
    python thesis_rebuild/scripts/00_download_dataset.py --api-key XXXX

    # or via env var
    set ROBOFLOW_API_KEY=XXXX        # Windows
    export ROBOFLOW_API_KEY=XXXX     # Linux/Mac
    python thesis_rebuild/scripts/00_download_dataset.py

Output:
    data/raw/{train,valid,test}/{images,labels}
    data/raw/data.yaml
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

WORKSPACE = "dydy-worker"
PROJECT = "palm-fruit-ripeness-detection-f6sac-ccb2z"
VERSION = 2
FORMAT = "yolov11"

CLASS_NAMES = {
    0: "Abnormal", 1: "Empty Bunch", 2: "Overripe",
    3: "Ripe", 4: "Underripe", 5: "Unripe",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="[00] Download dataset from Roboflow")
    p.add_argument("--api-key", type=str, default=os.environ.get("ROBOFLOW_API_KEY"),
                   help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "data" / "raw"),
                   help="Output directory for raw dataset")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if output dir already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key:
        sys.exit(
            "ERROR: no Roboflow API key.\n"
            "  Get it from https://app.roboflow.com -> Settings -> API\n"
            "  Then: --api-key XXXX  OR  set ROBOFLOW_API_KEY=XXXX"
        )

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        print(f"Output {out_dir} already exists and is non-empty.")
        print("Use --force to re-download. Skipping.")
        return

    print("=" * 70)
    print("[00] DOWNLOAD DATASET (fresh from Roboflow)")
    print("=" * 70)
    print(f"  Workspace: {WORKSPACE}")
    print(f"  Project:   {PROJECT}")
    print(f"  Version:   {VERSION}  Format: {FORMAT}")
    print(f"  Output:    {out_dir}")
    print()

    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("ERROR: roboflow not installed. Run: pip install roboflow")

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset = version.download(FORMAT, location=str(out_dir), overwrite=args.force)

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"  Location: {dataset.location}")
    for split in ("train", "valid", "test"):
        img_dir = out_dir / split / "images"
        n = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        print(f"  {split:6}: {n} images")
    print()
    print("NEXT: python thesis_rebuild/scripts/01_resplit_bunch_id.py")
    print("  (fixes per-frame leakage via bunch_id stratified group split)")


if __name__ == "__main__":
    main()
