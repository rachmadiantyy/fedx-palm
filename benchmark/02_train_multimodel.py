"""
STEP 2 — Latih beberapa varian YOLOv11 (n/s/m/l/x) secara tersentral.

Konfigurasi identik untuk semua model agar perbandingan adil (AdamW, 100 epoch,
augmentasi standar, cls=1.5 untuk membantu kelas minoritas).

Colab / lokal:
    python benchmark/02_train_multimodel.py \
        --data /content/dataset_merged/data.yaml \
        --project /content/drive/MyDrive/results
"""
from __future__ import annotations

import argparse

import torch
from ultralytics import YOLO

DEFAULT_MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]


def build_config(data: str, project: str, epochs: int, batch: int, device: str) -> dict:
    return {
        "data": data,
        "epochs": epochs,
        "imgsz": 640,
        "batch": batch,
        "workers": 4,
        "device": device,
        # Optimizer
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        # Augmentasi
        "mosaic": 1.0,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 15.0,
        "fliplr": 0.5, "flipud": 0.1,
        "mixup": 0.1,
        "copy_paste": 0.0, "erasing": 0.0,
        # Class-loss weight (bantu kelas minoritas: Empty Bunch)
        "cls": 1.5,
        # Saving
        "save": True, "save_period": 10,
        "project": project,
        "exist_ok": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path data.yaml")
    ap.add_argument("--project", default="./results", help="folder output")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="daftar bobot awal (default n/s/m/l/x)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="0 untuk GPU, cpu untuk CPU")
    args = ap.parse_args()

    cfg = build_config(args.data, args.project, args.epochs, args.batch, args.device)
    print(f"Konfigurasi siap | models={args.models} | epochs={args.epochs} | -> {args.project}")

    summary = {}
    for model_name in args.models:
        print(f"\n{'='*60}\nTraining: {model_name}\n{'='*60}")
        model = YOLO(model_name)
        result = model.train(**cfg, name=model_name.replace(".pt", ""))

        rd = result.results_dict
        summary[model_name] = {
            "mAP50":     rd.get("metrics/mAP50(B)", 0),
            "mAP50-95":  rd.get("metrics/mAP50-95(B)", 0),
            "precision": rd.get("metrics/precision(B)", 0),
            "recall":    rd.get("metrics/recall(B)", 0),
        }
        m = summary[model_name]
        print(f"[OK] {model_name}: mAP50={m['mAP50']:.4f} mAP50-95={m['mAP50-95']:.4f} "
              f"P={m['precision']:.4f} R={m['recall']:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nSemua model selesai ditraining.")


if __name__ == "__main__":
    main()
