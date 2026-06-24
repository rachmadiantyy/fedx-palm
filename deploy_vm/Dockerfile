# =============================================================================
# FedX-Palm — Inference web service (Flask + YOLOv11 + Grad-CAM++)
# CPU-only image, suitable for commodity VPS / on-prem VM deployment.
#
# Build (from the repository root):
#     docker build -f deploy_vm/Dockerfile -t fedx-palm:latest .
#
# The model checkpoint (best.pt) is NOT baked into the image; mount it at
# runtime via a host volume (see deploy_vm/docker-compose.yml).
# =============================================================================
FROM python:3.11-slim AS runtime

# OpenCV (headless) and Ultralytics need libgl + glib at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Torch CPU build is much smaller than the CUDA wheel.
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

# Python deps (split from torch so layer caching is more effective).
COPY deploy_vm/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Application code: Flask app + UI template + vendored XAI module.
COPY deploy_vm/app/app.py            /app/app.py
COPY deploy_vm/app/templates         /app/templates
COPY xai                             /app/xai

# Runtime configuration (override at `docker run` time as needed).
ENV MODEL_PATH=/app/weights/best.pt \
    CONF_THRES=0.25 \
    PORT=8080 \
    PYTHONUNBUFFERED=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/healthz || exit 1

# Gunicorn with a single sync worker is enough for the demo workload (one
# model copy, one request at a time keeps CPU memory predictable).
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", \
     "--timeout", "120", "--access-logfile", "-", "app:app"]
