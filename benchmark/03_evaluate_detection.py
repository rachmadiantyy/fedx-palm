"""
STEP 3 — Evaluasi detection pada test set untuk semua model terlatih.

Menghasilkan tabel + CSV: mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1, FPS,
ukuran model, dan AP per-kelas.

    python benchmark/03_evaluate_detection.py \
        --data /content/dataset_merged/data.yaml \
        --project /content/drive/MyDrive/results
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd
import torch
from ultralytics import YOLO

CLASS_NAMES = ["Abnormal", "Empty Bunch", "Overripe", "Ripe", "Underripe", "Unripe"]
DEFAULT_MODELS = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path data.yaml")
    ap.add_argument("--project", default="./results", help="folder hasil training")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--device", default="0")
    ap.add_argument("--fps-sample", type=int, default=100, help="jumlah citra untuk ukur FPS")
    args = ap.parse_args()

    # Ambil folder test images dari data.yaml
    import yaml
    with open(args.data) as f:
        test_dir = yaml.safe_load(f).get("test")

    summary = {}
    for model_name in args.models:
        model_path = os.path.join(args.project, model_name, "weights", "best.pt")
        if not os.path.exists(model_path):
            print(f"[SKIP] {model_name} tidak ditemukan di {model_path}")
            continue

        print(f"\nEvaluasi: {model_name}")
        model = YOLO(model_path)

        # mAP/P/R pada test set
        metrics = model.val(data=args.data, split="test", imgsz=640,
                            device=args.device, verbose=False)

        # FPS (rata-rata atas beberapa citra)
        fps = float("nan")
        if test_dir and os.path.isdir(test_dir):
            imgs = [os.path.join(test_dir, x) for x in os.listdir(test_dir)][: args.fps_sample]
            t0 = time.time()
            for im in imgs:
                model.predict(source=im, imgsz=640, device=args.device, verbose=False)
            dt = time.time() - t0
            fps = len(imgs) / dt if dt > 0 else float("nan")

        size_mb = os.path.getsize(model_path) / (1024 ** 2)
        rd = metrics.results_dict
        p = rd.get("metrics/precision(B)", 0)
        r = rd.get("metrics/recall(B)", 0)
        row = {
            "mAP50":     rd.get("metrics/mAP50(B)", 0),
            "mAP50-95":  rd.get("metrics/mAP50-95(B)", 0),
            "precision": p,
            "recall":    r,
            "f1":        2 * p * r / (p + r + 1e-10),
            "fps":       fps,
            "size_mb":   size_mb,
        }
        # AP per-kelas
        try:
            for i, cn in enumerate(CLASS_NAMES):
                if i < len(metrics.box.ap):
                    row[f"AP_{cn}"] = float(metrics.box.ap[i])
        except Exception:
            pass
        summary[model_name] = row
        print(f"   mAP50={row['mAP50']:.4f}  mAP50-95={row['mAP50-95']:.4f}  "
              f"P={p:.4f}  R={r:.4f}  F1={row['f1']:.4f}  FPS={fps:.2f}  {size_mb:.1f}MB")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not summary:
        print("Tidak ada model yang dievaluasi.")
        return

    # Tabel + CSV
    print("\n" + "=" * 85)
    print(f"{'Model':<12}{'mAP50':>8}{'mAP50-95':>10}{'Prec':>8}{'Recall':>8}{'F1':>8}{'FPS':>8}{'Size(MB)':>10}")
    print("-" * 85)
    for name, m in summary.items():
        print(f"{name:<12}{m['mAP50']:>8.4f}{m['mAP50-95']:>10.4f}{m['precision']:>8.4f}"
              f"{m['recall']:>8.4f}{m['f1']:>8.4f}{m['fps']:>8.2f}{m['size_mb']:>10.2f}")
    print("=" * 85)

    out_csv = os.path.join(args.project, "detection_metrics.csv")
    df = pd.DataFrame(summary).T
    df.index.name = "Model"
    df.to_csv(out_csv)
    print(f"\n[OK] Tersimpan: {out_csv}")


if __name__ == "__main__":
    main()
