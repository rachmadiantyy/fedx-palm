# `deploy_vm/` — Production inference for FedX-Palm

Minimal Flask + YOLOv11 + Grad-CAM++ web service, packaged as a CPU-only
Docker container, designed for deployment on a commodity VM through Portainer
Stacks.

## Files

| Path | Purpose |
|------|---------|
| `Dockerfile`           | Image build (Python 3.11 slim + torch CPU + ultralytics + flask + xai) |
| `docker-compose.yml`   | Portainer Stack file (bind-mounts the `best.pt` weights) |
| `requirements.txt`     | Python deps (split from torch so layer cache works) |
| `app/app.py`           | Flask app with `/`, `/healthz`, `/predict` |
| `app/templates/index.html` | Upload UI with detections table + heatmap pane |
| `DEPLOY.md`            | Step-by-step VM + Portainer deployment guide |
| `.dockerignore`        | Keeps the build context small |

## Quickstart (TL;DR)

1. SCP `best.pt` to the VM (e.g., `/srv/fedx-palm/weights/best.pt`).
2. Clone the repo to the VM.
3. In Portainer → Stacks → Add stack → upload `docker-compose.yml`, set
   `MODEL_PATH_HOST=/srv/fedx-palm/weights`, deploy.
4. Browse `http://VM_IP:8080/`.

Full instructions, troubleshooting, and an optional HTTPS reverse-proxy
recipe are in [`DEPLOY.md`](./DEPLOY.md).

## Why this folder exists separately from `thesis_rebuild/deploy/`

`thesis_rebuild/deploy/` was the original *blueprint* (Dockerfile +
`predict.py` batch/HTTP minimum). `deploy_vm/` is the
**production-shaped** variant used for the live thesis demonstration:

- Real HTML upload UI (not just a JSON endpoint).
- Grad-CAM++ overlay rendered server-side using the repo's `xai/` module.
- Portainer-friendly compose file with a bind-mounted weights folder so the
  model can be swapped without rebuilding the image.
- Gunicorn instead of Flask dev server.
- Docker healthcheck wired to `/healthz`.

Both folders coexist intentionally — `thesis_rebuild/deploy/` documents the
*architectural design*, `deploy_vm/` is what actually runs.
