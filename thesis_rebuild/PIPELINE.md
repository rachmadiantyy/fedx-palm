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

### 2. Dirichlet Non-IID partition (jalankan untuk SEMUA K)
```powershell
python thesis_rebuild/scripts/02_dirichlet_partition.py --K 2
python thesis_rebuild/scripts/02_dirichlet_partition.py --K 4
python thesis_rebuild/scripts/02_dirichlet_partition.py --K 8
python thesis_rebuild/scripts/02_dirichlet_partition.py --K 16
```
- Output per K: `data/clients_K{K}/client_{1..K}/` + per-client `data.yaml`
- Shared: `data/global_val.yaml`, `data/global_test.yaml`
- Alpha rentang per K (skewed -> near-IID); val/test SHARED antar client

### 3. Smoke test (2 epoch, verifikasi pipeline)
```powershell
# B1 centralized
python thesis_rebuild/scripts/train_b1_centralized.py ^
  --data data/resplit/data.yaml --epochs 2 --name b1_smoke

# E1 DP-SGD single sigma
python thesis_rebuild/scripts/train_e1_dp_sgd_full.py ^
  --data data/resplit/data.yaml --sigma 1.0 --epochs 2
```

### 4. Eksperimen penuh (FULL GRID K x sigma)
```powershell
# Satu perintah jalankan semua fase berurutan (~200 jam):
python thesis_rebuild/scripts/run_full_grid.py

# Atau jalankan fase tertentu manual:
python thesis_rebuild/scripts/train_b1_centralized.py --epochs 50 --name b1_centralized
python thesis_rebuild/scripts/train_b2_fl.py --all-K --rounds 5
python thesis_rebuild/scripts/train_e1_fl_dp_sgd_full.py --full-grid --rounds 5
python thesis_rebuild/scripts/train_e2_fl_dp_sgd_partial.py --full-grid --rounds 5
```

**Total compute matrix:**
| Phase | Runs | Per-run (est) | Total |
|---|---|---|---|
| B1 centralized | 1 | 2h | 2h |
| B2 federated x4 K | 4 | 3h | 12h |
| E1-FL full x4 K x5 sigma | 20 | 5h | 100h |
| E2-FL partial x4 K x5 sigma | 20 | 4h | 80h |
| **TOTAL** | **45** | | **~194h** |

Plus R1 multi-seed (~24h) -> grand total ~220h ~9-14 hari di RTX 4080.

**Tips overnight:**
- Jalankan via `Start-Process` (Windows) atau `nohup`/`tmux` (Linux)
- Monitor via `dir thesis_rebuild\runs\ /s /b | findstr best.pt`
- Kalau crash di tengah, `run_full_grid.py --skip-b1 --skip-b2` skip yang sudah selesai

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
