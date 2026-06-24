# FedX-Palm — VM Deployment Guide

Deploy the federated **B2 K=4 (mAP@0.5 = 0.738)** operating checkpoint as a
small Docker service on an Ubuntu/Debian VM, managed through Portainer. The
service exposes:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/`        | GET  | Upload UI (HTML) |
| `/healthz` | GET  | Liveness probe (used by Docker healthcheck) |
| `/predict` | POST | Multipart `image=<file>` → JSON with detections + Grad-CAM++ overlay (base64 PNG) |

Assumptions: Docker engine + Portainer already running on the VM. If not, see
the *Appendix — bare-metal install* at the bottom.

---

## 1. Upload the checkpoint to the VM

From your **laptop** (where `best.pt` currently lives):

```bash
# Replace USER and VM_IP. SSH key auth assumed.
ssh USER@VM_IP "sudo mkdir -p /srv/fedx-palm/weights && sudo chown -R \$USER /srv/fedx-palm"
scp thesis_rebuild/runs/b2_fl_K4_seed42/best.pt USER@VM_IP:/srv/fedx-palm/weights/best.pt
```

On the **VM**, verify:
```bash
ls -lh /srv/fedx-palm/weights/best.pt   # should be ~5 MB for YOLOv11n
```

If you put the weights somewhere else, just remember the path — you'll feed
it to Portainer as an environment variable in step 3.

## 2. Clone the repository to the VM

```bash
cd /srv
sudo git clone https://github.com/rachmadiantyy/fedx-palm.git
sudo chown -R $USER fedx-palm
cd fedx-palm
git checkout claude/thesis-rebuild-dp-sgd      # or whichever branch holds deploy_vm/
```

The image is built from the repo because the `xai/` Python module (vendored
Grad-CAM++) is copied into the container at build time.

## 3. Deploy via Portainer Stacks (the rapi way)

1. Open Portainer in your browser (usually `https://VM_IP:9443`).
2. Go to **Stacks → Add stack**.
3. Set **Name**: `fedx-palm`.
4. Under **Build method**, choose **Upload** → upload
   `deploy_vm/docker-compose.yml`. *(Or pick **Web editor** and paste the
   YAML directly.)*
5. Under **Environment variables**, add:

   | Name              | Value                          | Notes                                  |
   |-------------------|--------------------------------|----------------------------------------|
   | `MODEL_PATH_HOST` | `/srv/fedx-palm/weights`       | host folder mounted as `/app/weights`  |
   | `PORT_HOST`       | `8080`                         | external port; change if 8080 is taken |
   | `CONF_THRES`      | `0.25`                         | detection confidence threshold         |

6. Important: tell Portainer where the build context is. The compose file uses
   `context: ..` (the repo root). In Portainer's *Web editor* mode you can't
   build from a path on the host directly; instead use one of:

   - **Repository mode**: point Portainer at the GitHub repo
     `rachmadiantyy/fedx-palm`, branch `claude/thesis-rebuild-dp-sgd`,
     compose path `deploy_vm/docker-compose.yml`. Portainer will clone and
     build on the VM.
   - **Or build locally first** (recommended for the first run):
     ```bash
     cd /srv/fedx-palm
     docker build -f deploy_vm/Dockerfile -t fedx-palm:latest .
     ```
     Then in compose, comment out the `build:` block and uncomment a line
     with `image: fedx-palm:latest`. Saves Portainer from doing the build
     dance.

7. Click **Deploy the stack**. Watch the logs panel — first deployment will
   take a few minutes (torch CPU wheel ~200 MB). Subsequent restarts are
   instant.

## 4. Verify the service

```bash
curl -fsS http://VM_IP:8080/healthz
# {"classes":["Abnormal",...],"conf_thres":0.25,"model":"/app/weights/best.pt","status":"ok"}
```

Then point a browser at `http://VM_IP:8080/` and upload a test image. You
should see bounding boxes, the Grad-CAM++ overlay, a detections table, and
the per-image timings.

## 5. Optional: HTTPS / reverse proxy

The Flask service speaks plain HTTP on port 8080. For a polished demo, put
Caddy or Nginx in front:

```caddyfile
fedx.yourdomain.com {
    reverse_proxy 127.0.0.1:8080
}
```

Caddy will auto-issue a Let's Encrypt cert. No app changes needed.

---

## Updating the model later

Replace the file in place — no rebuild needed because the weights are
mounted, not baked in:

```bash
scp new_best.pt USER@VM_IP:/srv/fedx-palm/weights/best.pt
# then in Portainer: Stacks -> fedx-palm -> Restart
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Container exits immediately with `FileNotFoundError` | Volume not mounted, or `best.pt` not at the mapped path | Re-check `MODEL_PATH_HOST` env + `ls /srv/fedx-palm/weights` on the VM |
| `/predict` returns 502 / hangs | Gunicorn worker timed out on a very large image | Resize/compress upload, or raise `--timeout` in the Dockerfile |
| Heatmap is blank but boxes are correct | Grad-CAM++ target layer fell back to detect head | Check container logs — the line `Grad-CAM++ target: model.X.cv2.2` should be present at startup |
| Build fails on torch wheel | Slow / restricted network | Pre-download wheels locally and `pip install --no-index --find-links=./wheels` (advanced) |
| Healthcheck shows `unhealthy` in Portainer | Service still warming up | First-request inference takes ~10 s; healthcheck `start_period` is 30 s, raise if needed |

## Resource footprint (Ubuntu 22.04, 2 vCPU / 2 GB VM)

- Image size: ~1.6 GB (torch CPU is the bulk)
- RSS at idle: ~700 MB
- Latency per `/predict`: ~1.2 s detection + 0.8 s Grad-CAM++ ≈ 2 s total

If your VM has <2 GB RAM, drop `--workers 1` further is impossible — already
single-worker; instead consider running only detection (skip Grad-CAM++) by
deleting the gradcam block in `app.py`.

---

## Appendix — bare-metal install (only if Docker/Portainer is *not* on the VM yet)

```bash
# Docker engine
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker   # or log out/in

# Portainer CE (web UI on :9443)
docker volume create portainer_data
docker run -d -p 8000:8000 -p 9443:9443 --name portainer --restart=always \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v portainer_data:/data \
    portainer/portainer-ce:latest

# Then open https://VM_IP:9443 to finish admin setup.
```

After that, return to step 1 above.
