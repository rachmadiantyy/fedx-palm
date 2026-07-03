"""
PEMODELAN ULANG — YOLOv11 model BESAR (tanpa nano), kejar mAP maksimal.

Fokus ke model besar (l & x) karena nano/small hasilnya rendah.
Setelan diarahkan untuk akurasi tinggi, bukan kecepatan:
  - imgsz 800 (objek kecil lebih terbaca)
  - 200 epoch + cosine LR + early stopping
  - copy_paste + cls weight (bantu kelas minoritas)
  - TANPA differential privacy (sentral penuh)

Setiap model otomatis dievaluasi di test set dan hasilnya diringkas.

Contoh (server IIUM, RTX 4080):
    python remodel/train_large.py \
        --data data/resplit/data.yaml \
        --project runs/remodel \
        --models yolo11l.pt yolo11x.pt \
        --imgsz 800 --epochs 200

Kalau VRAM kurang (OOM), turunkan --batch 8 atau --imgsz 640.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from ultralytics import YOLO

CLASS_NAMES = ["Abnormal", "Empty Bunch", "Overripe", "Ripe", "Underripe", "Unripe"]

# Sengaja TIDAK menyertakan yolo11n / yolo11s (hasil kecil).
DEFAULT_MODELS = ["yolo11l.pt", "yolo11x.pt"]


def train_config(data, project, name, args) -> dict:
    return {
        "data": data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        # Optimizer
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 5,
        "cos_lr": True,
        "patience": args.patience,
        # Augmentasi
        "mosaic": 1.0,
        "close_mosaic": 15,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 15.0, "translate": 0.1, "scale": 0.5,
        "fliplr": 0.5, "flipud": 0.1,
        "mixup": 0.1, "copy_paste": 0.1,
        "cls": args.cls,
        # Saving
        "save": True, "save_period": 20,
        "project": project, "name": name, "exist_ok": True,
        "verbose": True, "seed": args.seed, "deterministic": True,
        "pretrained": True,
    }


def evaluate(best_path, data, imgsz, device):
    model = YOLO(best_path)
    try:
        m = model.val(data=data, split="test", imgsz=imgsz, device=device, verbose=False)
    except Exception:
        print("[info] split 'test' tidak ada -> pakai 'val'")
        m = model.val(data=data, split="val", imgsz=imgsz, device=device, verbose=False)
    rd = m.results_dict
    p = float(rd.get("metrics/precision(B)", 0.0))
    r = float(rd.get("metrics/recall(B)", 0.0))
    f1 = 2 * p * r / (p + r + 1e-10)
    out = {
        "mAP50":    round(float(rd.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50_95": round(float(rd.get("metrics/mAP50-95(B)", 0.0)), 4),
        "precision": round(p, 4),
        "recall":    round(r, 4),
        "f1":        round(f1, 4),
    }
    per_class = {}
    try:
        for i, cn in enumerate(CLASS_NAMES):
            if i < len(m.box.ap):
                per_class[cn] = round(float(m.box.ap[i]), 4)
    except Exception:
        pass
    out["AP_per_class"] = per_class
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path data.yaml (resplit anti-kebocoran)")
    ap.add_argument("--project", default="runs/remodel", help="folder output")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="daftar bobot besar (default: yolo11l.pt yolo11x.pt)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--cls", type=float, default=1.5)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    os.makedirs(args.project, exist_ok=True)
    print("=" * 64)
    print("  PEMODELAN ULANG — model besar (tanpa nano)")
    print(f"  models={args.models}  imgsz={args.imgsz}  epochs={args.epochs}")
    print(f"  data={args.data}  optimizer={args.optimizer}")
    print("=" * 64)

    summary = {}
    for model_file in args.models:
        name = model_file.replace(".pt", "") + f"_img{args.imgsz}"
        print(f"\n{'='*64}\n  Training: {name}\n{'='*64}")
        model = YOLO(model_file)
        model.train(**train_config(args.data, args.project, name, args))

        best = os.path.join(args.project, name, "weights", "best.pt")
        print(f"\n  Evaluasi {name} ...")
        res = evaluate(best, args.data, args.imgsz, args.device)
        res["best_weights"] = best
        summary[name] = res

        print(f"  -> mAP50={res['mAP50']}  mAP50-95={res['mAP50_95']}  "
              f"P={res['precision']}  R={res['recall']}  F1={res['f1']}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Ringkasan akhir
    print("\n" + "=" * 78)
    print(f"  {'Model':<22}{'mAP50':>9}{'mAP50-95':>11}{'Prec':>8}{'Recall':>8}{'F1':>8}")
    print("-" * 78)
    for name, m in summary.items():
        print(f"  {name:<22}{m['mAP50']:>9}{m['mAP50_95']:>11}"
              f"{m['precision']:>8}{m['recall']:>8}{m['f1']:>8}")
    print("=" * 78)

    out = os.path.join(args.project, "remodel_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Ringkasan disimpan: {out}")


if __name__ == "__main__":
    main()
