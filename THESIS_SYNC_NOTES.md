# Catatan Sinkronisasi Kode ↔ Draft Tesis

Dokumen ini merangkum hasil audit konsistensi antara kode di repo `fedx-palm`
dan klaim teknis pada draft tesis (PDF, 75 hal., dikompilasi 18 Mei 2026).

Bagian **A** berisi perubahan kode yang sudah dirapikan agar cocok dengan tesis.
Bagian **B** berisi gap yang **tidak bisa diperbaiki dari sisi kode** dan harus
disesuaikan di sumber LaTeX (Overleaf) sebelum submit final.

---

## A. Perubahan Kode (Sudah Diaplikasikan)

| # | Klaim Tesis | Sebelum | Sesudah | File |
|---|-------------|---------|---------|------|
| 1 | Port FL = **5000** (Sec 3.2.3.1) | `8080` | `5000` | `docker-compose*.yml`, `docker/Dockerfile.{server,client}*`, `docker/{server,client}_demo.py`, `configs/{server,client}_config.yaml`, `server/{fed_server,grpc_server}.py`, `client/fed_client.py`, `README.md` |
| 2 | TensorBoard di port **6006** (Tabel 3.1) | port di-expose tapi tidak ada logging | `SummaryWriter` di `FedXPalmStrategy.__init__`, log `avg/min/max map50`, `num_clients`, `total_samples` per ronde | `server/fed_server.py` |
| 3 | PyTorch + **CUDA 11.8** (Tabel 3.1) | base image `cuda12.1` | base image `cuda11.8` | `docker/Dockerfile.{server,client}` |
| 4 | Accountant = **RDP** (Sec 3.4.3) | `accountant: "moments"` di server_config | `accountant: "rdp"` | `configs/server_config.yaml` |
| 5 | Optimizer = **AdamW** (Tabel 3.5) | `optimizer: "SGD"` di client_config (kode sudah AdamW, tapi config dead) | `optimizer: "AdamW"` | `configs/client_config.yaml` |
| 6 | Augmentasi explicit: ±15° rotasi, HSV, flip, scale 0.8–1.2, mosaic (Sec 3.3.2) | hanya `mosaic`; sisanya implicit (default Ultralytics) | Tambah `degrees=15`, `fliplr=0.5`, `scale=0.2`, `translate=0.1`, `hsv_h/s/v` di config + diteruskan ke `model.train()` | `configs/client_config.yaml`, `client/fed_client.py` |
| 7 | σ ∈ **0.5–3.2** (Tabel 3.5) | preset hanya σ ∈ {0, 0.8, 1.5, 3.2}, batas bawah 0.5 hilang | Tambah preset `very_weak_privacy()` (σ=0.5, ε=12.0) | `utils/differential_privacy.py` |

---

## B. Gap yang Harus Diperbaiki di LaTeX (Overleaf)

Berikut yang **tidak bisa diselesaikan dari sisi kode** karena murni isu redaksional, format, atau bergantung pada output run eksperimen.

### B1. Halaman ACHIEVEMENT (hal. xv)
Saat ini berisi placeholder `AAAAA`. Ganti dengan daftar publikasi/penghargaan
yang relevan, contoh:
```
1. R. Dianty, F. Dewanta, T. S. Gunawan, "FedX-Palm: ...", APWIMOB 2024.
```

### B2. Broken Citations `[? ]`
Ada 5 sitasi rusak yang harus dilengkapi entry-nya di `.bib`:

| Lokasi (Sec) | Konteks | Saran sumber |
|--------------|---------|--------------|
| 2.2.1 (YOLOv11 deskripsi) | "iterasi terbaru ... [? ]" | Khanam, R. & Hussain, M. (2024) *YOLOv11: An Overview of the Key Architectural Enhancements*, arXiv:2410.17725 |
| 2.3.3 (DP-FedAvg) | "level gradien ... [? ]" | McMahan, B. et al. (2018) *Learning Differentially Private Recurrent Language Models*, ICLR |
| 2.3.4 (Dirichlet Non-IID) | "telah menjadi standar ... [? ]" | Hsu, T.-M. H., Qi, H., Brown, M. (2019) *Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification*, arXiv:1909.06335 |
| 2.4 (Docker container) | "disebut sebagai container [? ]" | Merkel, D. (2014) *Docker: Lightweight Linux Containers for Consistent Development and Deployment*, Linux Journal |
| 3.3.4 (Dirichlet sampling, ulangan) | "Pendekatan ini ... [? ]" | sama dengan 2.3.4 (Hsu dkk., 2019) |

### B3. List of Figures: Gambar 4.4 hilang
LoF lompat 4.3 → 4.5. Periksa nomor figure di BAB 4 — kemungkinan Gambar 4.4
ada di teks tapi belum di-`\caption`, atau ada nomor yang di-skip.
Solusi: rapikan label/caption Gambar 4.4 ATAU renumber 4.5 → 4.4.

### B4. Inkonsistensi Penamaan Kelas C1–C6
Antara BAB 1/3 dan BAB 4 ada konflik:

| Sumber | C1 | C2 | C3 | C4 | C5 | C6 |
|--------|----|----|----|----|----|----|
| BAB 1.4 (urutan) | Unripe | Underripe | Ripe | Overripe | Empty Bunch | Abnormal |
| Tabel 3.2 keterangan | "Sangat Matang" (Overripe) | ? | ? | ? | ? | "Janjang Kosong" (Empty Bunch) |
| Tabel 4.9 (BAB 4) | Unripe | Underripe | **Ripening** ⚠️ | Ripe | Overripe | Abnormal/Empty Bunch |
| `configs/data_sample.yaml` (kode) | Abnormal | Empty Bunch | Overripe | Ripe | Underripe | Unripe (alphabetical Roboflow) |

**Saran**: pilih ONE urutan kanonik dan terapkan konsisten di seluruh bab.
Mengingat dataset Roboflow alfabetis (sesuai kode), opsi paling aman:
- C1=Abnormal, C2=Empty Bunch, C3=Overripe, C4=Ripe, C5=Underripe, C6=Unripe.
- Kelas "Ripening" di Tabel 4.9 tampaknya typo dari "Underripe" atau "Overripe".

### B5. Sample Count Eksperimen (Tabel 3.3)
Tesis menyebut Client A=2289, B=2638, C=2555, D=3000, total=10.482 citra.
**Tidak hardcoded di repo** — script `scripts/download_dataset.py` mengunduh dari
Roboflow (angka eksak tergantung versi dataset). Tambahkan footnote di tesis:

> "Jumlah sampel diperoleh dari Roboflow version <X> (alphabetical class order)
> setelah pembagian Dirichlet dengan seed <Y>. Angka eksak dapat direplikasi
> menggunakan `scripts/download_dataset.py --seed Y`."

### B6. σ Range "0.5–3.2" (Tabel 3.5) vs Skenario "{0.8, 1.5, 3.2}" (Tabel 3.4)
Tabel 3.5 menyebut range **σ ∈ 0.5–3.2**, tapi Tabel 3.4 hanya menampilkan
4 skenario (σ ∈ {0, 0.8, 1.5, 3.2}). Lower bound 0.5 tidak punya skenario
ekuivalen.

**Pilihan**:
- (a) Ubah Tabel 3.5: σ range → "0.8–3.2" (sesuai 3 skenario dengan DP).
- (b) Tambah baris "Very Weak Privacy: ε≈12.0, σ=0.5" di Tabel 3.4 (kode sudah disiapkan: `PrivacyConfig.very_weak_privacy()`).

Rekomendasi: **(a)** lebih sederhana untuk konsistensi.

### B7. Penyebutan "Flower" vs Stub Demo
Tesis konsisten menyebut Flower (flwr). Kode utama (`server/fed_server.py`,
`client/fed_client.py`) memang pakai Flower. Tapi `server/grpc_server.py`
adalah implementasi **Flask REST** alternatif (legacy/dev), dan
`docker/server_demo.py` adalah **stub Flask** untuk demo defense (bukan
training nyata).

Untuk menghindari kebingungan reviewer, di BAB 3 atau Appendix tambahkan
catatan:
> "Implementasi utama menggunakan Flower (`fed_server.py`, `fed_client.py`).
> Modul `grpc_server.py` adalah varian REST untuk pengembangan, sedangkan
> `docker-compose.demo.yml` dengan `*_demo.py` adalah stub ringan untuk live
> demo saat sidang (output mAP di stub adalah mock, bukan hasil training)."

### B8. Typo yang perlu dikoreksi di teks
| Lokasi | Ditulis | Seharusnya |
|--------|---------|-----------|
| Sec 3.2.2 hal. 25 | "Dakam menjamin" | "Dalam menjamin" |
| Sec 4.1.2 hal. 39 | "lingkungan penelotian" | "lingkungan penelitian" |
| Sec 4.7.1 hal. 52 | "Hasil pebneliatn" | "Hasil penelitian" |

### B9. Lampiran A (disebut di Sec 3.2.2.2)
Teks: "Implementasi lengkap Dockerfile dan docker-compose.yml dapat dilihat
pada Lampiran A." → Lampiran A belum ada.

**Saran**: tambah Lampiran A berisi listing dari:
- `docker/Dockerfile.server` (server image)
- `docker/Dockerfile.client` (client image)
- `docker-compose.yml` (orkestrator 4-client + server, GPU profile)

Sumber kode terkini dapat dilihat di branch `claude/upbeat-goldberg-piyFr`
pada repo `https://github.com/rachmadiantyy/fedx-palm`.

---

## C. Branch & Artefak Repository (Catatan Maintenance)

| Branch | Status | Catatan |
|--------|--------|---------|
| `main` | minimal (LICENSE + README) | branch default, isi belum sinkron dengan kode |
| `claude/upbeat-goldberg-piyFr` | **AKTIF** (PR sinkronisasi ini) | rekomendasi untuk merge ke main |
| `claude/yolov11-roboflow-setup-JlyWD` | berisi demo Docker | sudah di-merge ke branch aktif |
| `feature/federated-learning-system` | branch lama | dapat dihapus setelah merge |

Folder `yolov11-roboflow-setup-JlyWD/notebooks/` mengandung notebook
centralized lama (hash beda dari `notebooks/`). Disarankan diverifikasi
mana yang final lalu hapus yang outdated.

---

## D. Quick-check Sinkronisasi Akhir

Setelah commit ini, verifikasi cepat di tesis bahwa nilai-nilai berikut **konsisten**
dengan kode:

- [x] Port FL = **5000** (Sec 3.2.3.1) — kode di `configs/server_config.yaml:7`
- [x] Port TensorBoard = **6006** (Tabel 3.1) — `docker-compose.yml:26`
- [x] PyTorch 2.1.0 + CUDA **11.8** — `docker/Dockerfile.server:5`
- [x] Optimizer **AdamW**, LR 0.01 — `configs/client_config.yaml:15`, `client/fed_client.py:323`
- [x] Local epochs 5, batch 16, rounds 100 — `configs/*.yaml`, `docker-compose.yml`
- [x] Privacy accountant **RDP** — `configs/server_config.yaml:35`, `utils/differential_privacy.py:66`
- [x] Clipping C=1.0, σ ∈ {0.5, 0.8, 1.5, 3.2} — `utils/differential_privacy.py`
- [x] Dirichlet α ∈ {0.1, 0.3, 0.5, 0.7} per 4 client — `docker-compose.yml` env `DIRICHLET_ALPHA`
- [x] Augmentasi ±15° rotasi, HSV, flip, scale, mosaic — `client/fed_client.py:_train_local()`
- [x] Grad-CAM++ + Average Drop + FRR — `xai/explainer.py`

Sumber kode terkini: branch `claude/upbeat-goldberg-piyFr`.
