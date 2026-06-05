"""FedX-Palm — Model Serving Demo (inference + Grad-CAM++).

Memuat best.pt hasil pelatihan Colab (federated/centralized baseline) dan
menyajikan endpoint web: unggah foto buah sawit -> deteksi YOLOv11 + heatmap
Grad-CAM++ secara langsung. Dipakai untuk demo sidang di VPS (CPU-only).

Angka metrik agregat (mAP, Average Drop, FRR) berasal dari simulasi Colab,
lihat Bab 4 thesis. App ini menunjukkan model nyata bekerja pada citra baru.
"""
import base64
import io
import os
import logging

import cv2
import numpy as np
from flask import Flask, request, render_template_string
from ultralytics import YOLO

from xai.explainer import GradCAMPlusPlus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fedx-serve")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/weights/best.pt")
CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))
PORT = int(os.getenv("PORT", "8080"))

# Metrik tervalidasi dari simulasi Colab (Bab 4 thesis) — ditampilkan sebagai
# konteks, BUKAN dihitung ulang per-request.
THESIS_METRICS = {
    "map50": "0,9945",
    "map5095": "0,8973",
    "avg_drop": "95,1%",
    "frr": "0,962",
}

app = Flask(__name__)
_model = None
_cam_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Bobot model tidak ditemukan di {MODEL_PATH}. "
                f"Salin best.pt dari Colab ke ./weights/best.pt (mount volume)."
            )
        logger.info(f"Loading model from {MODEL_PATH}")
        _model = YOLO(MODEL_PATH)
    return _model


def get_cam_model():
    """Instance YOLO terpisah khusus Grad-CAM++.

    Model untuk predict() berjalan di bawah inference_mode sehingga tensornya
    ter-"taint" dan tidak bisa di-backward. Grad-CAM butuh model yang belum
    pernah lewat predict(), jadi dipakai instance sendiri.
    """
    global _cam_model
    if _cam_model is None:
        logger.info(f"Loading separate CAM model from {MODEL_PATH}")
        _cam_model = YOLO(MODEL_PATH)
    return _cam_model


def encode_png(image_bgr):
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


PAGE = """
<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FedX-Palm — Demo Deteksi & Grad-CAM++</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }
    header { padding: 24px; background: #1e293b; border-bottom: 1px solid #334155; }
    h1 { margin: 0; font-size: 20px; }
    .sub { color: #94a3b8; font-size: 13px; margin-top: 4px; }
    main { max-width: 1100px; margin: 0 auto; padding: 24px; }
    .metrics { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px 18px; }
    .card .v { font-size: 22px; font-weight: 700; color: #38bdf8; }
    .card .l { font-size: 12px; color: #94a3b8; }
    form { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 20px; }
    input[type=file] { color: #e2e8f0; }
    button { background: #38bdf8; color: #0f172a; border: 0; padding: 10px 18px; border-radius: 8px; font-weight: 700; cursor: pointer; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; margin-top: 24px; }
    .grid figure { margin: 0; background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 12px; }
    .grid img { width: 100%; border-radius: 6px; }
    figcaption { font-size: 13px; color: #94a3b8; margin-top: 8px; }
    table { width: 100%; border-collapse: collapse; margin-top: 18px; background: #1e293b; border-radius: 10px; overflow: hidden; }
    th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid #334155; font-size: 14px; }
    th { background: #334155; }
    .err { background: #7f1d1d; border: 1px solid #b91c1c; padding: 14px; border-radius: 8px; margin-top: 18px; }
    .note { font-size: 12px; color: #64748b; margin-top: 24px; }
  </style>
</head>
<body>
  <header>
    <h1>FedX-Palm — Deteksi Kematangan Sawit + Grad-CAM++</h1>
    <div class="sub">Model: YOLOv11 (federated baseline, ε=∞) hasil pelatihan Colab · inference live</div>
  </header>
  <main>
    <div class="metrics">
      <div class="card"><div class="v">{{ m.map50 }}</div><div class="l">mAP@0.5 (baseline)</div></div>
      <div class="card"><div class="v">{{ m.map5095 }}</div><div class="l">mAP@0.5:0.95</div></div>
      <div class="card"><div class="v">{{ m.avg_drop }}</div><div class="l">Average Drop (XAI)</div></div>
      <div class="card"><div class="v">{{ m.frr }}</div><div class="l">Focus Retention Rate</div></div>
    </div>

    <form method="post" action="/predict" enctype="multipart/form-data">
      <p>Unggah foto buah kelapa sawit (.jpg/.png):</p>
      <input type="file" name="image" accept="image/*" required>
      <button type="submit">Deteksi + Heatmap</button>
    </form>

    {% if error %}<div class="err">{{ error }}</div>{% endif %}

    {% if detections is not none %}
    <div class="grid">
      <figure><img src="data:image/png;base64,{{ img_det }}"><figcaption>Hasil deteksi (bounding box + kelas)</figcaption></figure>
      {% if img_cam %}<figure><img src="data:image/png;base64,{{ img_cam }}"><figcaption>Grad-CAM++ (area fokus model)</figcaption></figure>{% endif %}
    </div>
    <table>
      <tr><th>#</th><th>Kelas</th><th>Confidence</th></tr>
      {% for d in detections %}
      <tr><td>{{ loop.index }}</td><td>{{ d.name }}</td><td>{{ '%.3f'|format(d.conf) }}</td></tr>
      {% else %}
      <tr><td colspan="3">Tidak ada objek terdeteksi (coba foto lain / turunkan threshold).</td></tr>
      {% endfor %}
    </table>
    {% endif %}

    <p class="note">Catatan: metrik agregat di atas berasal dari simulasi Colab (Bab 4 thesis).
    Halaman ini menjalankan model nyata pada citra yang Anda unggah untuk membuktikan
    model terdeploy berfungsi. Data uji tidak disimpan di server.</p>
  </main>
</body>
</html>
"""


@app.route("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH),
            "loaded": _model is not None}


@app.route("/")
def index():
    return render_template_string(PAGE, m=THESIS_METRICS, detections=None,
                                  error=None, img_det=None, img_cam=None)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template_string(PAGE, m=THESIS_METRICS, detections=None,
                                      error="Tidak ada berkas yang diunggah.",
                                      img_det=None, img_cam=None)

    try:
        model = get_model()
    except FileNotFoundError as e:
        return render_template_string(PAGE, m=THESIS_METRICS, detections=None,
                                      error=str(e), img_det=None, img_cam=None)

    data = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return render_template_string(PAGE, m=THESIS_METRICS, detections=None,
                                      error="Berkas bukan gambar yang valid.",
                                      img_det=None, img_cam=None)

    results = model.predict(source=image, conf=CONF_THRES, verbose=False)
    res = results[0]
    annotated = res.plot()  # BGR dengan box + label

    detections = []
    top_class = None
    top_conf = -1.0
    names = res.names
    if res.boxes is not None and len(res.boxes) > 0:
        for cls_id, conf in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            detections.append({"name": names[int(cls_id)], "conf": float(conf)})
            if conf > top_conf:
                top_conf = conf
                top_class = int(cls_id)

    # Grad-CAM++ pada kelas dengan confidence tertinggi.
    img_cam = None
    if top_class is not None:
        cam = GradCAMPlusPlus(get_cam_model())
        try:
            exp = cam.generate(image, target_class=top_class)
            if exp.overlay is not None:
                img_cam = encode_png(exp.overlay)
        except Exception as e:
            logger.warning(f"Grad-CAM++ gagal: {e}")
        finally:
            cam.cleanup()

    return render_template_string(
        PAGE, m=THESIS_METRICS, detections=detections,
        error=None, img_det=encode_png(annotated), img_cam=img_cam,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False, threaded=True)
