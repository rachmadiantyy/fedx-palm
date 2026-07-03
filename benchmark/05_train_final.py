"""
STEP 5 — Pemodelan ULANG (sentral, kejar akurasi maksimal).

Beda dari 02_train_multimodel.py (yang menyapu n/s/m/l/x untuk perbandingan),
skrip ini FOKUS melatih satu model besar dengan setelan yang mengejar mAP
setinggi mungkin — untuk memenuhi target akurasi pembimbing.

Lever yang dipakai untuk menaikkan akurasi:
  1. Model lebih besar        -> default yolo11l.pt (bisa yolo11x.pt)
  2. TANPA differential privacy -> tidak ada noise/clipping yang menurunkan utility
  3. Resolusi lebih tinggi     -> imgsz 800 (deteksi objek kecil lebih baik)
  4. Training lebih panjang    -> 200 epoch + early stopping (patience)
  5. cls weight               -> bantu kelas minoritas (Empty Bunch)
  6. Split anti-kebocoran      -> pakai data.yaml hasil resplit by bunch_id

Setelah training, otomatis evaluasi di test set dan cetak metrik lengkap
(mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1, AP per-kelas).

Contoh (server GPU):
    python benchmark/05_train_final.py \
        --data resplit/data.yaml \
        --project ./results_final \
        --model yolo11l.pt --imgsz 800 --epochs 200

Kalau GPU terbatas, turunkan --imgsz 640 dan/atau --model yolo11m.pt.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from ultralytics import YOLO

CLASS_NAMES = ["Abnormal", "Empty Bunch", "Overripe", "Ripe", "Underripe", "Unripe"]


def build_config(args) -> dict:
    return {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        # Optimizer (AdamW = setelan sentral ala Quenta; kejar konvergensi cepat)
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 5,
        "cos_lr": True,            # cosine LR schedule -> konvergensi lebih halus
        "patience": args.patience, # early stopping kalau tidak membaik
        # Augmentasi (kuat tapi aman untuk deteksi buah)
        "mosaic": 1.0,
        "close_mosaic": 15,        # matikan mosaic di 15 epoch terakhir -> stabil
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 15.0,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5, "flipud": 0.1,
        "mixup": 0.1,
        "copy_paste": 0.1,         # bantu kelas minoritas
        # Class-loss weight (bantu Empty Bunch dkk.)
        "cls": args.cls,
        # Saving
        "save": True, "save_period": 20,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "verbose": True,
        "seed": args.seed,
        "deterministic": True,
        "pretrained": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path data.yaml (hasil resplit by bunch_id)")
    ap.add_argument("--project", default="./results_final", help="folder output")
    ap.add_argument("--model", default="yolo11l.pt",
                    help="bobot awal (yolo11m.pt / yolo11l.pt / yolo11x.pt)")
    ap.add_argument("--name", default=None, help="nama run (default: dari --model)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--cls", type=float, default=1.5)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="0", help="0 untuk GPU, cpu untuk CPU")
    args = ap.parse_args()

    if args.name is None:
        args.name = args.model.replace(".pt", "") + f"_final_imgsz{args.imgsz}"

    print("=" * 64)
    print(f"  PEMODELAN ULANG (sentral) — kejar akurasi maksimal")
    print(f"  model={args.model}  imgsz={args.imgsz}  epochs={args.epochs}")
    print(f"  optimizer={args.optimizer}  lr0={args.lr0}  cls={args.cls}")
    print(f"  data={args.data}")
    print("=" * 64)

    cfg = build_config(args)
    model = YOLO(args.model)
    model.train(**cfg)

    # ---- Evaluasi otomatis di test set ----
    print("\nEvaluasi di test set ...")
    best = os.path.join(args.project, args.name, "weights", "best.pt")
    if os.path.exists(best):
        model = YOLO(best)
    # split 'test' kalau tersedia di data.yaml, kalau tidak fallback ke 'val'
    try:
        metrics = model.val(data=args.data, split="test", imgsz=args.imgsz,
                            device=args.device, verbose=False)
    except Exception:
        print("[info] split 'test' tidak ada -> pakai 'val'")
        metrics = model.val(data=args.data, split="val", imgsz=args.imgsz,
                            device=args.device, verbose=False)

    rd = metrics.results_dict
    p = rd.get("metrics/precision(B)", 0.0)
    r = rd.get("metrics/recall(B)", 0.0)
    f1 = 2 * p * r / (p + r + 1e-10)
    summary = {
        "model": args.model,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "mAP50":    round(float(rd.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50_95": round(float(rd.get("metrics/mAP50-95(B)", 0.0)), 4),
        "precision": round(float(p), 4),
        "recall":    round(float(r), 4),
        "f1":        round(float(f1), 4),
        "best_weights": best,
    }
    # AP per-kelas
    per_class = {}
    try:
        for i, cn in enumerate(CLASS_NAMES):
            if i < len(metrics.box.ap):
                per_class[cn] = round(float(metrics.box.ap[i]), 4)
    except Exception:
        pass
    summary["AP_per_class"] = per_class

    print("\n" + "=" * 64)
    print("  HASIL AKHIR")
    print("=" * 64)
    print(f"  mAP@0.5      : {summary['mAP50']}")
    print(f"  mAP@0.5:0.95 : {summary['mAP50_95']}")
    print(f"  Precision    : {summary['precision']}")
    print(f"  Recall       : {summary['recall']}")
    print(f"  F1-score     : {summary['f1']}")
    if per_class:
        print("  AP per-kelas :")
        for cn, ap in per_class.items():
            print(f"     {cn:<12}: {ap}")
    print("=" * 64)

    out = os.path.join(args.project, args.name, "final_metrics.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Ringkasan disimpan: {out}")
    print(f"[OK] Bobot terbaik     : {best}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
