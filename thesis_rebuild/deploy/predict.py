"""FedX-Palm — Inference deployment (batch CLI + opsi HTTP endpoint).

Menyajikan model deteksi kematangan TBS hasil pelatihan (best.pt). Skrip ini
adalah bagian dari blueprint deployment (lihat Dockerfile pada folder yang
sama) dan TIDAK melakukan pelatihan; pelatihan dilakukan via simulasi FL
sequential berbeda (Bab 3).

Dua mode:
  1. Batch CLI : python predict.py --source path/ke/gambar_atau_folder
  2. HTTP API  : python predict.py --serve   (POST gambar ke /predict)

Variabel lingkungan:
  MODEL_PATH  (default: weights/best.pt)
  CONF_THRES  (default: 0.25)
  PORT        (default: 8080)
"""
import argparse
import io
import os
from pathlib import Path

from ultralytics import YOLO

MODEL_PATH = os.getenv("MODEL_PATH", "weights/best.pt")
CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))
PORT = int(os.getenv("PORT", "8080"))

# 6 kelas kematangan TBS (urutan alfabet, konsisten dengan data.yaml).
CLASS_NAMES = ["Abnormal", "Empty Bunch", "Overripe", "Ripe", "Underripe", "Unripe"]


def load_model(model_path: str = MODEL_PATH) -> YOLO:
    """Memuat model YOLOv11 dari checkpoint .pt."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan di '{model_path}'. "
            "Mount weights/ sebagai volume atau set MODEL_PATH."
        )
    return YOLO(model_path)


def predict_batch(model: YOLO, source: str, out_dir: str = "runs/predict") -> None:
    """Inferensi batch pada file/folder gambar, simpan hasil ber-anotasi."""
    results = model.predict(
        source=source, conf=CONF_THRES, save=True, project=out_dir, name="infer"
    )
    for r in results:
        counts: dict[str, int] = {}
        for c in r.boxes.cls.tolist():
            name = CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else str(int(c))
            counts[name] = counts.get(name, 0) + 1
        print(f"{Path(r.path).name}: {counts or 'tidak ada deteksi'}")
    print(f"\nHasil ber-anotasi tersimpan di: {out_dir}/infer")


def serve(model: YOLO) -> None:
    """Menjalankan HTTP endpoint sederhana: POST /predict (field 'image')."""
    from flask import Flask, jsonify, request

    app = Flask("fedx-palm-infer")

    @app.get("/health")
    def health():
        return jsonify(status="ok", model=MODEL_PATH, classes=CLASS_NAMES)

    @app.post("/predict")
    def predict():
        if "image" not in request.files:
            return jsonify(error="kirim file pada field 'image'"), 400
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
        results = model.predict(source=np.array(img), conf=CONF_THRES, verbose=False)
        r = results[0]
        detections = [
            {
                "class": CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else int(c),
                "confidence": round(float(p), 4),
                "bbox_xyxy": [round(float(v), 1) for v in box],
            }
            for c, p, box in zip(
                r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()
            )
        ]
        return jsonify(detections=detections, count=len(detections))

    print(f"FedX-Palm inference server di http://0.0.0.0:{PORT}  (model: {MODEL_PATH})")
    app.run(host="0.0.0.0", port=PORT)


def main() -> None:
    ap = argparse.ArgumentParser(description="FedX-Palm inference deployment")
    ap.add_argument("--source", help="file/folder gambar untuk inferensi batch")
    ap.add_argument("--serve", action="store_true", help="jalankan HTTP endpoint")
    ap.add_argument("--model", default=MODEL_PATH, help="path checkpoint .pt")
    args = ap.parse_args()

    model = load_model(args.model)

    if args.serve:
        serve(model)
    elif args.source:
        predict_batch(model, args.source)
    else:
        ap.error("pilih salah satu: --source <path> (batch) atau --serve (HTTP)")


if __name__ == "__main__":
    main()
