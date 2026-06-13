# FedX-Palm Rebuild — Pipeline From Zero

Jalankan berurutan. Setiap langkah deterministik (seed=42) dan
menulis output yang dipakai langkah berikutnya. Tidak ada state
tersembunyi dari eksperimen lama — semua dibangun dari nol.

## Prasyarat (sekali saja)

```powershell
conda activate fedx
pip install opacus roboflow            # opacus sudah; roboflow untuk download
```

## Langkah

### 0. Download dataset (fresh dari Roboflow)
```powershell
python thesis_rebuild/scripts/00_download_dataset.py --api-key <ROBOFLOW_KEY>
```
- Output: `data/raw/{train,valid,test}/{images,labels}` + `data.yaml`
- Ini split asli Roboflow (acak per-frame, MASIH leaky — diperbaiki di step 1)

### 1. Anti-leakage re-split (bunch_id)
```powershell
python thesis_rebuild/scripts/01_resplit_bunch_id.py
```
- Output: `data/resplit/{train,valid,test}` + `data.yaml`
- Stratified group split 80/10/10, tiap tandan hanya di satu split
- Audit otomatis: harus print `BERSIH` untuk ketiga pasangan split

### 2. Dirichlet Non-IID partition (4 client)
```powershell
python thesis_rebuild/scripts/02_dirichlet_partition.py
```
- Output: `data/clients/client_{1..4}/` + per-client `data.yaml`
- Output: `data/global_val.yaml`, `data/global_test.yaml`
- alpha = {0.1, 0.3, 0.5, 0.7}; val/test SHARED antar client

### 3. Smoke test (2 epoch, verifikasi pipeline)
```powershell
# B1 centralized
python thesis_rebuild/scripts/train_b1_centralized.py ^
  --data data/resplit/data.yaml --epochs 2 --name b1_smoke

# E1 DP-SGD single sigma
python thesis_rebuild/scripts/train_e1_dp_sgd_full.py ^
  --data data/resplit/data.yaml --sigma 1.0 --epochs 2
```

### 4. Eksperimen penuh (setelah smoke test lolos)
```powershell
# B1 centralized baseline (upper bound)
python thesis_rebuild/scripts/train_b1_centralized.py ^
  --data data/resplit/data.yaml --epochs 50 --name b1_centralized

# E1 full DP-SGD sweep
python thesis_rebuild/scripts/train_e1_dp_sgd_full.py ^
  --data data/resplit/data.yaml --sweep --epochs 50

# E2 partial DP-SGD sweep (backbone frozen)
python thesis_rebuild/scripts/train_e2_dp_sgd_partial.py ^
  --data data/resplit/data.yaml --sweep --epochs 50
```
(B2 federated + federated DP-SGD: ditambahkan Day 3)

## Peta direktori data (setelah pipeline)

```
data/
├── raw/                  # [00] Roboflow asli (leaky)
├── resplit/              # [01] bersih, anti-leakage  <- training centralized
│   ├── train|valid|test/
│   └── data.yaml
├── clients/              # [02] partisi Non-IID
│   └── client_{1..4}/    #      <- training federated
├── global_val.yaml       # [02] evaluasi global shared
└── global_test.yaml      # [02] held-out test
```

Semua di bawah `data/` di-gitignore (lihat .gitignore) — hanya script
dan config yang masuk repo. Reproduksi = jalankan ulang 00→02.
