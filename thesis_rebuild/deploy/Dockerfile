# =============================================================================
# FedX-Palm — Inference Deployment Blueprint
# =============================================================================
# CETAK BIRU (blueprint) untuk men-deploy model deteksi kematangan TBS hasil
# pelatihan (best.pt). Image ini HANYA untuk inference/serving, BUKAN untuk
# pelatihan maupun arsitektur Federated Learning. Pelatihan dilakukan via
# simulasi FL sequential di GPU (lihat thesis_rebuild Bab 3.1).
#
# Build : docker build -f thesis_rebuild/deploy/Dockerfile -t fedx-palm:infer .
# Run   : docker run --rm -p 8080:8080 \
#             -v "$(pwd)/weights:/app/weights:ro" \
#             fedx-palm:infer
#
# Image CPU-only dipilih karena target deployment lapangan (mini-PC pabrik,
# VPS murah) umumnya tanpa GPU; YOLOv11n cukup ringan untuk inferensi CPU.
# =============================================================================
FROM python:3.10-slim

# libGL + glib dibutuhkan OpenCV (dependensi ultralytics).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch varian CPU (jauh lebih kecil dari build CUDA).
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "ultralytics>=8.3" flask

# Script inference (model di-mount sebagai volume read-only saat run).
COPY thesis_rebuild/deploy/predict.py /app/predict.py

# Path model & konfigurasi via env (override saat run bila perlu).
ENV MODEL_PATH=/app/weights/best.pt \
    CONF_THRES=0.25 \
    PORT=8080

EXPOSE 8080

# Default: jalankan HTTP server. Untuk batch CLI:
#   docker run ... fedx-palm:infer python predict.py --source /app/data/img.jpg
CMD ["python", "predict.py", "--serve"]
