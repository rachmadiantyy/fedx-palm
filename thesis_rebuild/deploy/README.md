# Deployment Inference — FedX-Palm

Blueprint untuk men-deploy model deteksi kematangan TBS hasil pelatihan
(`weights/best.pt`). **Hanya untuk inference/serving**, bukan untuk pelatihan
maupun arsitektur Federated Learning (pelatihan memakai simulasi FL sequential
di GPU, lihat Bab 3.1).

## Konsep (blueprint)

```
Dockerfile  →  docker build  →  Image  →  docker run  →  Container
[cetak biru]                  [paket]                  [aktif melayani]
```

`Dockerfile` adalah **cetak biru reproducible**: menjamin model bisa dijalankan
ulang di lingkungan mana pun (mini-PC pabrik, VPS, server) tanpa konflik
dependency. Membangun & menjalankan container bersifat opsional (nilai tambah);
Dockerfile tetap valid sebagai bukti *deployment-readiness*.

## Cara pakai

### 1. Tanpa Docker (langsung di env `fedx`)

```bash
# Inferensi batch (satu gambar atau folder)
python thesis_rebuild/deploy/predict.py --source path/ke/gambar.jpg

# HTTP endpoint
python thesis_rebuild/deploy/predict.py --serve
```

### 2. Dengan Docker

```bash
# Build (jalankan dari root repo)
docker build -f thesis_rebuild/deploy/Dockerfile -t fedx-palm:infer .

# Serve (model di-mount read-only)
docker run --rm -p 8080:8080 -v "$(pwd)/weights:/app/weights:ro" fedx-palm:infer

# Inferensi batch via container
docker run --rm -v "$(pwd)/weights:/app/weights:ro" -v "$(pwd)/data:/app/data:ro" \
    fedx-palm:infer python predict.py --source /app/data/contoh.jpg
```

### Uji endpoint

```bash
curl http://localhost:8080/health
curl -F "image=@contoh.jpg" http://localhost:8080/predict
```

Respons `POST /predict`:

```json
{
  "count": 2,
  "detections": [
    {"class": "Ripe", "confidence": 0.91, "bbox_xyxy": [120.0, 88.0, 410.0, 502.0]},
    {"class": "Overripe", "confidence": 0.77, "bbox_xyxy": [430.0, 110.0, 690.0, 540.0]}
  ]
}
```

## Konfigurasi (env var)

| Variabel | Default | Keterangan |
|---|---|---|
| `MODEL_PATH` | `weights/best.pt` | path checkpoint hasil pelatihan |
| `CONF_THRES` | `0.25` | ambang confidence deteksi |
| `PORT` | `8080` | port HTTP endpoint |

## Catatan

- Image **CPU-only** (Torch CPU): target deployment lapangan umumnya tanpa GPU,
  ukuran image jauh lebih kecil, dan YOLOv11n cukup ringan untuk inferensi CPU.
- Letakkan `best.pt` di folder `weights/` sebelum build/run, atau mount sebagai
  volume seperti contoh di atas.
