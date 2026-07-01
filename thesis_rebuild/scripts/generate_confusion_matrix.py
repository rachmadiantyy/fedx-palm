"""
Generate a per-class confusion matrix for a FedX-Palm checkpoint.

WHY
---
The federated training loop only logs GLOBAL aggregate detection metrics per
round (see thesis Section 3.11.4), so per-class AP / confusion is NOT available
from the training logs. But a confusion matrix can still be produced POST-HOC:
just run Ultralytics validation on the saved checkpoint over the held-out test
set. Ultralytics then auto-plots `confusion_matrix.png` and
`confusion_matrix_normalized.png` in the run's save directory.

This script reuses the SAME proven loader as evaluate_xai.py (rebuild the
GroupNorm YOLO, load the federated state), so it works on the custom federated
best.pt ({'state','config',...}) as well as a plain B1 checkpoint.

RUN (on the machine where evaluate_xai.py already works — your laptop):
    python thesis_rebuild/scripts/generate_confusion_matrix.py \
        --weights thesis_rebuild/runs/b2_fl_K4_seed42/best.pt \
        --data    configs/global_test.yaml \
        --name    cm_b2_k4

Output (look for the printed path at the end):
    runs/detect/cm_b2_k4/confusion_matrix_normalized.png   <-- pakai ini untuk tesis
    runs/detect/cm_b2_k4/confusion_matrix.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Reuse the exact loader that already works for the XAI evaluation.
from thesis_rebuild.scripts.evaluate_xai import load_model, CLASS_NAMES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Confusion matrix (post-hoc) for a FedX-Palm checkpoint")
    ap.add_argument("--weights", required=True,
                    help="federated best.pt ({'state',...}) atau B1 .pt biasa")
    ap.add_argument("--data", required=True,
                    help="dataset YAML yang punya split 'test' (mis. global_test.yaml)")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="ambang confidence")
    ap.add_argument("--iou", type=float, default=0.5, help="ambang IoU untuk pencocokan")
    ap.add_argument("--device", default="cpu", help="cpu atau 0 (GPU)")
    ap.add_argument("--name", default="confusion_matrix", help="nama folder output")
    args = ap.parse_args()

    # 1) Rebuild the GN YOLO and load the federated state (proven path).
    #    load_model() juga sudah menonaktifkan in-place SiLU + fusi Conv+BN
    #    (GroupNorm tak punya running stats), jadi .val() aman.
    yolo = load_model(args.weights, args.device)

    # 2) Beri label 6 kelas sawit agar sumbu confusion matrix benar.
    #    (Kepala deteksi 80-lebar warisan COCO; hanya indeks 0-5 yang dilatih.)
    yolo.model.names = {i: n for i, n in enumerate(CLASS_NAMES)}
    if hasattr(yolo, "names"):
        yolo.names = yolo.model.names

    # 3) Jalankan validasi -> Ultralytics otomatis membuat confusion matrix.
    print(f"[val] menjalankan validasi pada split '{args.split}' dari {args.data} ...")
    metrics = yolo.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=True,        # <- ini yang menghasilkan confusion_matrix(_normalized).png
        save_json=False,
        name=args.name,
        exist_ok=True,
    )

    save_dir = getattr(metrics, "save_dir", None) or f"runs/detect/{args.name}"
    print("\n==================================================")
    print(f"[OK] Confusion matrix tersimpan di: {save_dir}")
    print("     -> confusion_matrix_normalized.png   (PAKAI INI untuk tesis)")
    print("     -> confusion_matrix.png              (versi nilai mentah)")
    print("--------------------------------------------------")
    try:
        print("mAP@0.5 per-kelas:")
        for i, n in enumerate(CLASS_NAMES):
            ap50 = metrics.box.maps[i] if hasattr(metrics, "box") else None
            print(f"   {n:<12} {ap50:.3f}" if ap50 is not None else f"   {n:<12} -")
    except Exception as e:  # noqa: BLE001
        print(f"   (ringkasan per-kelas tidak tersedia: {e})")
    print("==================================================")


if __name__ == "__main__":
    main()
