"""
STEP 1 — Setup dataset & tulis data.yaml (6 kelas kematangan TBS).

Colab:
    !pip install ultralytics grad-cam -q
    from google.colab import drive; drive.mount('/content/drive')
    python benchmark/01_setup_dataset.py \
        --zip /content/drive/MyDrive/dataset_merged.zip \
        --out /content/dataset_merged

Lokal:
    python benchmark/01_setup_dataset.py --zip dataset_merged.zip --out ./dataset_merged
"""
from __future__ import annotations

import argparse
import os
import zipfile

import yaml

CLASS_NAMES = ["Abnormal", "Empty Bunch", "Overripe", "Ripe", "Underripe", "Unripe"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="path ke dataset_merged.zip")
    ap.add_argument("--out", default="./dataset_merged", help="folder tujuan ekstrak")
    args = ap.parse_args()

    # 1) Ekstrak kalau belum ada
    if not os.path.exists(args.out):
        with zipfile.ZipFile(args.zip, "r") as zf:
            zf.extractall(args.out)
        print(f"[OK] Dataset diekstrak ke {args.out}")
    else:
        print(f"[SKIP] {args.out} sudah ada")

    # 2) Tulis data.yaml (Roboflow: train/ valid/ test/ berisi images/ + labels/)
    data_yaml_path = os.path.join(args.out, "data.yaml")
    data_yaml = {
        "train": os.path.join(args.out, "train", "images"),
        "val":   os.path.join(args.out, "valid", "images"),
        "test":  os.path.join(args.out, "test", "images"),
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"[OK] data.yaml ditulis -> {data_yaml_path}")

    # 3) Verifikasi jumlah gambar per split
    for split in ("train", "valid", "test"):
        d = os.path.join(args.out, split, "images")
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        print(f"   {split:<6}: {n} gambar" + ("" if n else "  <-- CEK PATH!"))


if __name__ == "__main__":
    main()
