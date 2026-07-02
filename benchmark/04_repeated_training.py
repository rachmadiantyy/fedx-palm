"""
STEP 4 — Repeated training multi-seed untuk analisis ketangguhan statistik.

Melatih model yang sama pada beberapa seed, lalu melaporkan mAP/P/R di test set
(disimpan sebagai JSON per-run). Berguna untuk melaporkan rerata +/- simpangan
baku antar-seed.

    python benchmark/04_repeated_training.py \
        --data /content/dataset_merged/data.yaml \
        --project /content/drive/MyDrive/Tesis_XAI/repeated_training
"""
from __future__ import annotations

import argparse
import json
import os

from ultralytics import YOLO

# (model_file, model_key) — pasangan bobot awal & nama run.
# CATATAN: 'yolo26m.pt' TIDAK ada di Ultralytics. Ganti dengan varian nyata:
#   yolov8m.pt, yolo11m.pt, yolo12m.pt (kalau tersedia di versi Ultralytics-mu).
DEFAULT_MODELS = [
    ("yolov8m.pt", "yolov8m"),
    ("yolo11m.pt", "yolo11m"),
    # ("yolo12m.pt", "yolo12m"),   # aktifkan kalau paket mendukung
]
DEFAULT_SEEDS = [42, 123]


def train_and_eval(model_file: str, model_key: str, seed: int,
                   data: str, results_dir: str) -> dict:
    run_name = f"{model_key}_seed{seed}"
    result_file = os.path.join(results_dir, f"{run_name}_metrics.json")

    if os.path.exists(result_file):
        print(f"[SKIP] {run_name} sudah ada.")
        with open(result_file) as f:
            return json.load(f)

    print(f"\n{'='*55}\n  Training: {run_name}\n{'='*55}")
    model = YOLO(model_file)
    model.train(
        data=data,
        epochs=100, batch=16, imgsz=640,
        optimizer="AdamW", lr0=0.001,
        seed=seed, deterministic=True, pretrained=True,
        exist_ok=True, verbose=True, close_mosaic=10,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=15.0, translate=0.1, scale=0.5,
        flipud=0.1, fliplr=0.5, mosaic=1.0, mixup=0.1,
        project=results_dir, name=run_name,
    )

    print(f"\nEvaluasi {run_name} di test set ...")
    metrics = model.val(data=data, split="test")
    output = {
        "model": model_key,
        "seed": seed,
        "mAP50":     round(float(metrics.box.map50), 4),
        "mAP50_95":  round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall":    round(float(metrics.box.mr), 4),
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[OK] {run_name}: mAP50={output['mAP50']} mAP50-95={output['mAP50_95']}")
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path data.yaml")
    ap.add_argument("--project", default="./results/repeated", help="folder output")
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = ap.parse_args()

    os.makedirs(args.project, exist_ok=True)
    all_results = []
    for model_file, model_key in DEFAULT_MODELS:
        for seed in args.seeds:
            all_results.append(
                train_and_eval(model_file, model_key, seed, args.data, args.project))

    # Ringkasan rerata +/- SD per model
    print("\n" + "=" * 55)
    print("RINGKASAN (rerata antar-seed):")
    import statistics as st
    by_model: dict[str, list] = {}
    for r in all_results:
        by_model.setdefault(r["model"], []).append(r)
    for model_key, runs in by_model.items():
        maps = [r["mAP50_95"] for r in runs]
        mean = sum(maps) / len(maps)
        sd = st.pstdev(maps) if len(maps) > 1 else 0.0
        print(f"  {model_key:<10} mAP50-95 = {mean:.4f} +/- {sd:.4f}  (n={len(maps)} seed)")
    print("=" * 55)


if __name__ == "__main__":
    main()
