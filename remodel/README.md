# Pemodelan Ulang — YOLOv11 Model Besar (kejar mAP maksimal)

Jalur remodeling **sentral** yang fokus ke model besar (**yolo11l / yolo11x**),
bukan nano. Tujuannya menaikkan mAP setinggi mungkin untuk memenuhi target
akurasi pembimbing.

> Nano (`yolo11n`) & small (`yolo11s`) sengaja **tidak dipakai** karena hasilnya
> paling rendah. Fokus ke `l` dan `x`.

## Kenapa ini lebih tinggi dari hasil FL+DP (0.787)
| Lever | Efek |
|---|---|
| Model besar (l/x, ~25–57 M params) | Kapasitas jauh > nano (~2.6 M) |
| Tanpa Differential Privacy | Tidak ada noise/clipping yang menurunkan utility |
| `imgsz=800` | Objek kecil/jauh lebih terbaca |
| 200 epoch + cosine LR + early stopping | Konvergensi lebih matang |
| `copy_paste` + `cls=1.5` | Bantu kelas minoritas (Empty Bunch) |

## Step-by-step (server IIUM / GPU)
```bash
# 1) Masuk folder repo & ambil branch ini
cd D:\rachma\fedx-palm
git fetch origin
git checkout claude/remodel-yolo11-large

# 2) (opsional tapi disarankan) cek distribusi kelas per split
#    memastikan tidak ada kelas yang 0 instance (mis. Empty Bunch)
python remodel/check_classes.py --data data/resplit/data.yaml

# 3) Training model besar + evaluasi otomatis
python remodel/train_large.py \
    --data data/resplit/data.yaml \
    --project runs/remodel \
    --models yolo11l.pt yolo11x.pt \
    --imgsz 800 --epochs 200

# 4) Lihat ringkasan hasil
type runs\remodel\remodel_summary.json
```

Kalau VRAM RTX 4080 kurang saat `imgsz=800` (error OOM), turunkan:
```bash
python remodel/train_large.py --data data/resplit/data.yaml \
    --project runs/remodel --models yolo11l.pt yolo11x.pt \
    --imgsz 640 --batch 8 --epochs 200
```

## Output
- Bobot terbaik: `runs/remodel/<model>_img800/weights/best.pt`
- Ringkasan metrik: `runs/remodel/remodel_summary.json`
  (mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1, AP per-kelas)

## Catatan penting
- **Ini model sentral tanpa DP.** Kalau dipakai sebagai hasil utama tesis,
  posisi Federated Learning + privasi bergeser jadi eksperimen pendukung.
  Diskusikan framing ini dengan pembimbing.
- Kalau `yolo11l` sudah tembus target (mis. ≥0.85), tidak wajib lanjut `yolo11x`.
