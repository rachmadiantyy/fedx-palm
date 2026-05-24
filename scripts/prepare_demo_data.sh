#!/usr/bin/env bash
# Prepare data/client_1 dan data/client_2 untuk FAST DEMO Docker.
# Download dataset Roboflow (kecil), split 50/50 untuk 2 client,
# bikin data.yaml YOLO format di masing-masing client folder.
set -euo pipefail

ROBOFLOW_API_KEY="${ROBOFLOW_API_KEY:-Ej0bSMpeSri3ky0IYkOU}"
WORKSPACE="palm-fruit-ripeness-detection-f6sac-ccb2z"
PROJECT="palm-fruit-ripeness-detection-f6sac-ccb2z"
VERSION=2

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${ROOT}/data"
RAW_DIR="${DATA_DIR}/_raw"
CLASSES='["Abnormal","Empty Bunch","Overripe","Ripe","Underripe","Unripe"]'

echo "[1/4] Install roboflow SDK (pip --user)"
pip install --quiet --user roboflow==1.1.45 pyyaml >/dev/null

echo "[2/4] Download dataset YOLOv11 format -> ${RAW_DIR}"
mkdir -p "${RAW_DIR}"
cd "${RAW_DIR}"
python <<PY
from roboflow import Roboflow
rf = Roboflow(api_key="${ROBOFLOW_API_KEY}")
ds = rf.workspace("${WORKSPACE}").project("${PROJECT}").version(${VERSION}).download("yolov11")
print("DATASET_DIR=" + ds.location)
PY

# Find downloaded folder (Roboflow nests under a versioned name)
RAW_SUB="$(find "${RAW_DIR}" -maxdepth 2 -name 'data.yaml' -printf '%h\n' | head -1)"
if [[ -z "${RAW_SUB}" ]]; then
  echo "ERROR: dataset folder tidak ketemu" >&2; exit 1
fi
echo "Dataset di: ${RAW_SUB}"

echo "[3/4] Split train images 50/50 ke client_1 & client_2"
python <<PY
import os, shutil, random, pathlib, yaml
raw = pathlib.Path("${RAW_SUB}")
out = pathlib.Path("${DATA_DIR}")
classes = ["Abnormal","Empty Bunch","Overripe","Ripe","Underripe","Unripe"]

train_imgs = sorted((raw/"train"/"images").glob("*.*"))
val_imgs   = sorted((raw/"valid"/"images").glob("*.*"))
random.seed(42); random.shuffle(train_imgs)
half = len(train_imgs) // 2
splits = {"client_1": train_imgs[:half], "client_2": train_imgs[half:]}

for cid, imgs in splits.items():
    base = out / cid
    for sub in ["images/train","images/val","labels/train","labels/val"]:
        (base/sub).mkdir(parents=True, exist_ok=True)
    # train images + labels
    for img in imgs:
        shutil.copy(img, base/"images/train"/img.name)
        lbl = raw/"train"/"labels"/(img.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, base/"labels/train"/lbl.name)
    # shared val set (small)
    for img in val_imgs[:20]:
        shutil.copy(img, base/"images/val"/img.name)
        lbl = raw/"valid"/"labels"/(img.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, base/"labels/val"/lbl.name)
    with open(base/"data.yaml","w") as f:
        yaml.safe_dump({
            "path": "/app/data",
            "train": "images/train",
            "val":   "images/val",
            "nc": len(classes),
            "names": classes,
        }, f, sort_keys=False)
    print(f"  {cid}: {len(imgs)} train + 20 val")
PY

echo "[4/4] Selesai. Struktur:"
ls "${DATA_DIR}"/client_1 "${DATA_DIR}"/client_2
echo
echo "Siap run: docker compose -f docker-compose.fast-demo.yml up --build"
