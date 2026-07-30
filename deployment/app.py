#!/usr/bin/env python3
"""FedX-Palm inference service -- Flask + Ultralytics (CPU), with a Grad-CAM++
heatmap overlay on every prediction so the operator can see *why* the model
called a bunch ripe/unripe/etc, not just the label (mirrors thesis Bab 4.11).
"""
import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from flask import Flask, render_template, request  # noqa: E402

import fedxpalm  # noqa: E402,F401 (applies the GroupNorm-safe `fuse()` patch)
from fedxpalm.xai.gradcam import YOLOGradCAMPlusPlus, overlay_heatmap  # noqa: E402

WEIGHTS_PATH = os.environ.get("FEDXPALM_WEIGHTS", "models/best.pt")
DEVICE = os.environ.get("FEDXPALM_DEVICE", "cpu")
IMGSZ = int(os.environ.get("FEDXPALM_IMGSZ", "640"))
CONF_THRESHOLD = float(os.environ.get("FEDXPALM_CONF_THRESHOLD", "0.25"))
TARGET_LAYER_IDX = int(os.environ.get("FEDXPALM_GRADCAM_LAYER", "22"))

app = Flask(__name__)

with open(Path(__file__).resolve().parent.parent / "configs" / "dataset.yaml") as f:
    CLASS_NAMES = yaml.safe_load(f)["names"]

from ultralytics import YOLO  # noqa: E402

_yolo = YOLO(WEIGHTS_PATH)
_yolo.model.to(DEVICE).eval()
_gradcam = YOLOGradCAMPlusPlus(_yolo.model, target_layer_idx=TARGET_LAYER_IDX)


def _preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(image_bgr, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor.to(DEVICE), resized


@torch.no_grad()
def _best_detection(tensor: torch.Tensor):
    """Returns (class_id, anchor_idx, box_xyxy, confidence) for the single
    highest-confidence anchor across the whole image, or None if nothing
    clears CONF_THRESHOLD."""
    y, _preds = _yolo.model(tensor)
    scores = y[0, 4:, :]  # [nc, num_anchors], already sigmoid'd
    boxes = y[0, :4, :]   # [4, num_anchors], decoded xyxy
    class_id, anchor_idx = np.unravel_index(scores.argmax().item(), scores.shape)
    confidence = scores[class_id, anchor_idx].item()
    if confidence < CONF_THRESHOLD:
        return None
    box = boxes[:, anchor_idx].tolist()
    return int(class_id), int(anchor_idx), box, confidence


def run_inference(image_bgr: np.ndarray) -> dict:
    tensor, resized_bgr = _preprocess(image_bgr)
    detection = _best_detection(tensor)
    if detection is None:
        return {"detected": False, "overlay_b64": _to_b64(resized_bgr)}

    class_id, anchor_idx, box, confidence = detection
    cam, _raw_score, _, _decoded_box = _gradcam.generate(tensor.clone(), class_id, anchor_idx=anchor_idx,
                                                          output_size=(IMGSZ, IMGSZ))
    overlay = overlay_heatmap(resized_bgr, cam)

    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{CLASS_NAMES[class_id]} {confidence:.2f}"
    cv2.putText(overlay, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return {
        "detected": True,
        "class_name": CLASS_NAMES[class_id],
        "confidence": round(confidence, 4),
        "overlay_b64": _to_b64(overlay),
    }


def _to_b64(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr)
    return base64.b64encode(buf).decode("ascii") if ok else ""


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("index.html", result=None, error="Pilih file citra TBS terlebih dahulu.")

    data = np.frombuffer(file.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return render_template("index.html", result=None, error="File bukan citra yang valid.")

    result = run_inference(image_bgr)
    return render_template("index.html", result=result, error=None)


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "weights": WEIGHTS_PATH, "device": DEVICE}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
