"""
FedX-Palm — Inference + Grad-CAM++ web service for VM deployment.

Endpoints
---------
GET  /          : Upload UI (HTML).
GET  /healthz   : Liveness probe (used by Docker healthcheck).
POST /predict   : Multipart form `image=<file>`; returns JSON with
                  detections (boxes/classes/conf) and a Grad-CAM++ overlay
                  as base64 PNG.

Environment variables
---------------------
MODEL_PATH   path to best.pt mounted into the container (default
             /app/weights/best.pt)
CONF_THRES   minimum detection confidence (default 0.25)
PORT         HTTP port inside the container (default 8080)
"""
from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image
from ultralytics import YOLO

# Vendored Grad-CAM++ implementation; the repo's xai/ module is copied into
# the image at /app/xai/ so we can import it directly here.
sys.path.insert(0, "/app")
from xai.explainer import GradCAMPlusPlus  # noqa: E402

# The deployed checkpoint is a GroupNorm YOLOv11 (BatchNorm was replaced for
# DP-SGD compatibility). GroupNorm has no running statistics, so Ultralytics'
# Conv+BN fusion crashes with "'GroupNorm' object has no attribute 'running_var'".
# Fusion is a BN-only inference speed-up that is numerically identical for a GN
# model, so neutralize it to a no-op. The fuse() method is defined on BaseModel
# (DetectionModel inherits it) AND AutoBackend calls model.fuse() at predict
# time, so we patch BaseModel.fuse at the class level BEFORE any model is loaded
# to cover every code path (load, val, and predict via AutoBackend). is_fused()
# is also forced True so AutoBackend skips fusion entirely.
from ultralytics.nn.tasks import BaseModel, DetectionModel  # noqa: E402
BaseModel.fuse = lambda self, verbose=True: self
DetectionModel.fuse = lambda self, verbose=True: self
BaseModel.is_fused = lambda self, thresh=10: True

# -----------------------------------------------------------------------------
# Constants & configuration
# -----------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "/app/weights/best.pt")
CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))
PORT = int(os.getenv("PORT", "8080"))

CLASS_NAMES = ["Abnormal", "Empty Bunch", "Overripe", "Ripe", "Underripe", "Unripe"]
# Distinct BGR colours for each class' bounding box (OpenCV uses BGR).
CLASS_COLOURS = [
    (60, 60, 220),   # Abnormal     — red
    (200, 200, 200), # Empty Bunch  — light grey
    (50, 100, 255),  # Overripe     — orange
    (50, 50, 200),   # Ripe         — dark red
    (50, 180, 220),  # Underripe    — yellow-ish
    (80, 200, 80),   # Unripe       — green
]

# -----------------------------------------------------------------------------
# Model loading (eager at startup so /predict latency is just inference)
# -----------------------------------------------------------------------------
def _pick_target_layer(detection_model) -> Optional[str]:
    """Last shared Conv2d before YOLOv11's decoupled detection head.

    Hooking the absolute-last Conv2d (which sits inside the box/DFL branch of
    the Detect head) yields an all-zero gradient for the class score, so the
    heatmap collapses to zero. Picking the last shared neck Conv2d restores a
    meaningful gradient path.
    """
    seq = detection_model.model
    head_prefix = f"model.{len(seq) - 1}"
    last_name = None
    for name, module in detection_model.named_modules():
        if isinstance(module, nn.Conv2d) and not name.startswith(head_prefix):
            last_name = name
    return last_name


def _load_model() -> tuple[YOLO, GradCAMPlusPlus]:
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at '{MODEL_PATH}'. Mount the host "
            "weights directory or set MODEL_PATH."
        )
    yolo = YOLO(MODEL_PATH)
    # Grad-CAM hooks need gradients; YOLO's predict context disables them, so
    # we keep the model in eval mode and let the explainer enable grad inside.
    yolo.model.to("cpu").eval()
    target = _pick_target_layer(yolo.model)
    cam = GradCAMPlusPlus(yolo, target_layer=target)
    print(f"[fedx-palm] Model loaded from {MODEL_PATH} (Grad-CAM++ target: {target})",
          flush=True)
    return yolo, cam


# -----------------------------------------------------------------------------
# Image utilities
# -----------------------------------------------------------------------------
def _overlay_heatmap(img_bgr: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Blend a normalised heatmap (any size) onto a BGR image."""
    if heatmap is None:
        return img_bgr
    hm = cv2.resize(heatmap.astype(np.float32),
                    (img_bgr.shape[1], img_bgr.shape[0]))
    hm -= hm.min()
    hm /= (hm.max() + 1e-8)
    hm_u8 = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 0.55, hm_color, 0.45, 0)


def _draw_boxes(img_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and class labels on top of a BGR image."""
    out = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox_xyxy"])
        cls_id = det["class_id"]
        colour = CLASS_COLOURS[cls_id] if 0 <= cls_id < len(CLASS_COLOURS) else (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        label = f"{det['class']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _encode_png_b64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to PNG-encode image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------
app = Flask("fedx-palm", template_folder="/app/templates",
            static_folder="/app/static")
yolo_model, gradcam = _load_model()


@app.get("/")
def index():
    return render_template("index.html",
                           classes=CLASS_NAMES,
                           conf_thres=CONF_THRES)


@app.get("/healthz")
def healthz():
    return jsonify(status="ok", model=MODEL_PATH, classes=CLASS_NAMES,
                   conf_thres=CONF_THRES)


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify(error="Send the image as a form field named 'image'."), 400

    t0 = time.time()
    pil = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
    img_rgb = np.array(pil)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # 1) Detection
    results = yolo_model.predict(source=img_rgb, conf=CONF_THRES,
                                 verbose=False, device="cpu")
    r = results[0]
    detections = []
    for i, (cls, conf, box) in enumerate(zip(
            r.boxes.cls.tolist(), r.boxes.conf.tolist(),
            r.boxes.xyxy.tolist())):
        cls_id = int(cls)
        detections.append({
            "id": i,
            "class_id": cls_id,
            "class": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
            "confidence": round(float(conf), 4),
            "bbox_xyxy": [round(float(v), 1) for v in box],
        })

    infer_ms = (time.time() - t0) * 1000

    # 2) Grad-CAM++ — target the most confident detection's class
    heatmap = None
    cam_ms = 0.0
    if detections:
        top_cls = max(detections, key=lambda d: d["confidence"])["class_id"]
        try:
            tcam = time.time()
            expl = gradcam.generate(img_bgr, target_class=top_cls)
            cam_ms = (time.time() - tcam) * 1000
            heatmap = expl.heatmap
        except Exception as e:  # XAI is best-effort; never break /predict
            print(f"[fedx-palm] gradcam failed: {e}", flush=True)

    # 3) Compose response images
    detection_img = _draw_boxes(img_bgr, detections)
    heatmap_img = _overlay_heatmap(img_bgr, heatmap) if heatmap is not None else None

    return jsonify(
        detections=detections,
        count=len(detections),
        timings_ms={"inference": round(infer_ms, 1),
                    "gradcam": round(cam_ms, 1)},
        images={
            "detections_b64": _encode_png_b64(detection_img),
            "heatmap_b64": _encode_png_b64(heatmap_img) if heatmap_img is not None else None,
        },
    )


if __name__ == "__main__":
    # Production-grade WSGI is not strictly required for a single-tenant demo;
    # if you need it, swap to gunicorn in the Dockerfile CMD.
    app.run(host="0.0.0.0", port=PORT)
