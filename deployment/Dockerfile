# FedX-Palm inference service -- CPU-only, matches the original thesis's
# deployment target (a CPU-only VPS, for reproducibility/portability across
# environments without needing a GPU in production).
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY deployment/requirements.txt /app/deployment-requirements.txt
# CPU-only torch wheels are far smaller than the default (CUDA-bundled) PyPI
# build. If your build environment can't reach download.pytorch.org, drop
# --index-url to fall back to the regular (larger) PyPI wheel instead.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1 \
    && pip install --no-cache-dir -r /app/deployment-requirements.txt

COPY src/ /app/src/
COPY configs/dataset.yaml /app/configs/dataset.yaml
COPY deployment/ /app/deployment/

ENV PYTHONPATH=/app/src
ENV FEDXPALM_WEIGHTS=/app/models/best.pt
ENV FEDXPALM_DEVICE=cpu

# Mount or bake your trained checkpoint at this path, e.g.:
#   docker run -v $(pwd)/models:/app/models -p 8000:8000 fedx-palm-infer
VOLUME ["/app/models"]

EXPOSE 8000
CMD ["python", "deployment/app.py"]
