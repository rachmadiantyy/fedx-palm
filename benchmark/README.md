# YOLO Multi-Model Benchmark (Centralized)

Benchmark pelatihan **tersentral** beberapa varian YOLO pada dataset kematangan
TBS sawit (6 kelas), plus **repeated training** multi-seed untuk analisis
ketangguhan. Setup ini menyamai alur Colab "Quenta".

> Catatan: ini **jalur eksperimen terpisah** dari kerangka FedX-Palm
> (Federated Learning + DP + XAI) di branch `claude/thesis-rebuild-dp-sgd`.
> Di sini murni benchmark deteksi tersentral antar-arsitektur.

## Dataset (6 kelas)
`Abnormal, Empty Bunch, Overripe, Ripe, Underripe, Unripe`
Struktur Roboflow standar: `train/ valid/ test/` masing-masing berisi
`images/` dan `labels/`.

## Alur (jalankan berurutan)
| Skrip | Fungsi |
|---|---|
| `01_setup_dataset.py` | Ekstrak `dataset_merged.zip` + tulis `data.yaml` |
| `02_train_multimodel.py` | Latih YOLOv11 n/s/m/l/x (AdamW, 100 epoch) |
| `03_evaluate_detection.py` | Evaluasi test set: mAP, P, R, F1, FPS, size, AP per-kelas → CSV |
| `04_repeated_training.py` | Repeated training multi-seed (YOLOv8m, YOLOv11m, dst.) |

## Cara pakai (Google Colab)
```python
!pip install ultralytics grad-cam -q
from google.colab import drive; drive.mount('/content/drive')
# lalu jalankan tiap skrip / sel sesuai urutan di atas
```

## Cara pakai (lokal / GPU sendiri)
```bash
pip install ultralytics grad-cam
python benchmark/01_setup_dataset.py --zip dataset_merged.zip --out ./dataset_merged
python benchmark/02_train_multimodel.py --data ./dataset_merged/data.yaml --project ./results
python benchmark/03_evaluate_detection.py --data ./dataset_merged/data.yaml --project ./results
python benchmark/04_repeated_training.py --data ./dataset_merged/data.yaml --project ./results/repeated
```

## Konfigurasi pelatihan (default, sama untuk semua model)
- optimizer AdamW, `lr0=0.001`, `lrf=0.01`, `epochs=100`, `imgsz=640`, `batch=16`
- augmentasi: mosaic 1.0, HSV, degrees 15, fliplr 0.5, flipud 0.1, mixup 0.1
- `cls=1.5` (class-loss weight — membantu kelas minoritas seperti *Empty Bunch*)

## ⚠️ Catatan penting
- **`yolo26m.pt` tidak ada** di Ultralytics. Kalau mau varian lain untuk
  perbandingan repeated-training, pakai `yolov8m.pt`, `yolo11m.pt`, atau
  `yolo12m.pt`. Sudah kutandai di `04_repeated_training.py`.
- Perhitungan `training_time` di kode Colab asli keliru (memakai ulang waktu
  inferensi). Di `03_evaluate_detection.py` sudah dibetulkan (dibaca dari
  `results.csv` bila tersedia, atau dilaporkan sebagai N/A).
