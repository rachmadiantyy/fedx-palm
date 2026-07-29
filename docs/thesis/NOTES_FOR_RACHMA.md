# Catatan Serah Terima -- FedX-Palm v2

## Update 2026-07-28 -- Cakupan dipersempit ke B1/B2 saja

Kamu meng-upload manuskrip JUTIF (`docs/thesis/manuscript/
JUTIF_Manuscript_FedXPalm_B1B2.docx`) yang sudah berisi hasil B1/B2 **nyata**
dari GPU-mu (bukan lagi placeholder): B1 mAP50 *held-out test* = 0,8820,
B2 mAP50 *validation* rata-rata tiga-*seed* = 0,8775 (SD = 0,0157). Atas
permintaanmu, Bab 1-5 di `docs/thesis/` sudah dirombak total di *branch*
`claude/tesis-b1-b2-c77afk` supaya cakupannya dipersempit **hanya ke B1
(tersentral) dan B2 (federasi FedAvg tanpa DP, K=4)**, mengikuti persis
cakupan manuskrip tersebut:

- DP-SGD (E1/E2), *Explainable AI* (Grad-CAM++), dan *deployment* Docker
  yang sebelumnya jadi tulang punggung kerangka FedX-Palm sekarang hanya
  disebut sebagai **arah pengembangan lanjutan yang direncanakan** (Subbab
  1.4, 3.9, 5.3) -- bukan lagi bagian dari rumusan masalah/tujuan/
  hipotesis/hasil bab-bab ini. Infrastruktur kodenya (`src/fedxpalm/
  privacy/`, `src/fedxpalm/xai/`, `deployment/`, `scripts/07-10`) tetap
  ada di repo untuk tahap lanjutan, hanya tidak dijalankan/dilaporkan di
  sini.
- *Sweep* K $\in \{2,4,8,12,16\}$ juga dipersempit ke **K=4 saja**,
  mengikuti manuskrip. Infrastruktur *sweep* K lain tetap tersedia di
  `configs/fl_config.yaml`.
- Bab 4 sekarang berisi tabel hasil B1 per-kelas dan B2 tiga-*seed* yang
  asli, dikutip langsung dari manuskrip -- bukan lagi template kosong.
- Item `[TODO: ...]` yang masih ada di manuskrip asli (penyebab performa
  lemah kelas *Ripe*, nama co-author, email, tanggal submisi, pernyataan
  ketersediaan kode/data) dipertahankan sebagai `[TODO]` di Bab 4, **jangan
  diisi dengan tebakan**.
- Jika suatu saat kamu lanjut ke DP-SGD/XAI/deployment, itu jadi laporan
  terpisah yang membangun di atas titik rujukan B1/B2 di *branch* ini --
  bukan menyisipkannya kembali ke Bab 1-5 yang sudah dipersempit ini.

Ringkasan apa yang sudah dikerjakan di sesi ini, apa yang perlu kamu
jalankan sendiri di GPU, dan apa yang perlu dikirim balik supaya Bab 4-5
bisa ditulis dari angka asli (bukan tebakan).

## Kenapa dipecah begini

Sandbox yang menulis kode ini **tidak punya GPU** dan **tidak bisa
mengakses api.roboflow.com** (diblokir kebijakan jaringan environment-nya).
Jadi semua kode sudah ditulis dan diuji secara mekanis (arsitektur, alur
FedAvg, DP-SGD, Grad-CAM++, deployment) memakai dataset sintetis + CPU di
sandbox itu -- termasuk menemukan dan memperbaiki dua bug nyata (lihat
bagian "Bug yang ditemukan" di bawah). Tapi angka mAP/epsilon/AD/FRR yang
sesungguhnya baru bisa didapat setelah kamu jalankan di GPU-mu sendiri.

## Yang sudah beres (siap pakai)

- `configs/dataset.yaml`, `fl_config.yaml`, `dp_config.yaml` -- semua
  hiperparameter di satu tempat.
- `src/fedxpalm/` -- paket Python lengkap: data (download/split/partition),
  model (BatchNorm->GroupNorm), federated (FedAvg), privacy (DP-SGD +
  akuntansi PRV), xai (Grad-CAM++ + Average Drop/Focus Retention Rate),
  eval (wrapper mAP/P/R/F1).
- `scripts/01_*.py` s.d. `scripts/10_*.py` -- pipeline lengkap dari unduh
  dataset sampai ekspor tabel hasil.
- `deployment/` -- Dockerfile + layanan Flask + Grad-CAM++ overlay,
  sudah diuji end-to-end (upload gambar -> deteksi -> heatmap) di sandbox.
- `notebooks/FedXPalm_v2_Colab.ipynb` -- notebook yang menjalankan semua
  langkah di atas berurutan, dengan sel uji-cepat opsional sebelum sweep
  penuh.
- `docs/thesis/01_pendahuluan.md`, `02_landasan_teori.md`,
  `03_metodologi.md` -- Bab 1-3 lengkap, ditulis mengikuti kode yang
  sungguh-sungguh ada di repo ini.
- `docs/thesis/04_hasil_dan_pembahasan.md`,
  `05_kesimpulan_dan_saran.md` -- **template**, section-by-section, siap
  diisi begitu kamu punya hasil nyata.

## Yang perlu kamu jalankan (di GPU-mu)

1. Clone branch `claude/palm-oil-yolov11-federated-m4o613`, buka
   `notebooks/FedXPalm_v2_Colab.ipynb` di Jupyter/Colab.
2. Jalankan sel demi sel berurutan. **Sangat disarankan** jalankan dulu
   versi uji-cepat tiap blok (K=4 saja, rounds=5) sebelum commit ke sweep
   penuh (bisa berjam-jam untuk grid E1/E2 penuh).
3. Setelah selesai, `scripts/10_export_results.py` menghasilkan CSV di
   `results/` yang jadi sumber semua tabel Bab 4.

## Yang WAJIB diverifikasi sebelum Bab 1-3 dianggap final

Ditandai `[VERIFIKASI: ...]` di teks masing-masing bab:

1. **Nama & urutan 6 kelas** -- sudah diperbarui di `configs/dataset.yaml`
   (`Abnormal, Empty Bunch, Overripe, Ripe, Underripe, Unripe`, alfabetis)
   berdasarkan cross-check terhadap branch lain di repo yang sama
   (`claude/thesis-rebuild-dp-sgd`, lihat bagian "Catatan tentang branch
   lain" di bawah) yang sudah pernah mengunduh dataset ini secara nyata.
   Tetap **cek ulang** terhadap `data.yaml` hasil unduhan
   `scripts/01_download_dataset.py` versimu sendiri untuk memastikan.
2. **Spesifikasi GPU** yang sungguh-sungguh kamu pakai -- dari `nvidia-smi`
   yang kamu tunjukkan di sesi ini: **NVIDIA GeForce RTX 4080, 16 GB
   VRAM**. Sudah bisa dipakai mengisi Bab 1 Subbab 1.5.1 dan Bab 3 Subbab
   3.2.1 langsung.
3. **Karakteristik dataset** (jumlah citra, jumlah instans per kelas) --
   isi di Bab 3 Subbab 3.3.1 setelah unduhan selesai.

## Catatan tentang branch lain di repo ini (`claude/thesis-rebuild-dp-sgd`)

Selama sesi ini ketahuan ada branch **lain** di repo yang sama
(`claude/thesis-rebuild-dp-sgd`) yang tampaknya jauh lebih maju --
Bab 1-5 sudah lengkap (`thesis_rebuild/chapters/bab1_pendahuluan.md` s.d.
`bab5_kesimpulan.md`), ada hasil eksperimen asli
(`thesis_rebuild/tables/*.csv`), deployment VM yang sudah jalan dan
diukur latensinya, bahkan draft paper untuk submit ke jurnal (IEEE Access/
Sensors/dst.). Riwayat commit-nya menyebut deadline ~27 Juni 2026 (sudah
lewat saat sesi ini berlangsung, 3 Juli 2026), jadi kemungkinan besar
sudah final/mendekati final.

Kamu memutuskan **tetap lanjut di branch ini** (`claude/palm-oil-yolov11-
federated-m4o613`) dan training dari nol, bukan memakai branch itu. Kalau
di kemudian hari berubah pikiran, branch itu masih aman tersimpan di
`origin/claude/thesis-rebuild-dp-sgd` -- tidak disentuh oleh apapun di
sesi ini.

## Bug nyata yang ditemukan & diperbaiki (relevan untuk Bab 3)

Dua bug ini bukan sekadar detail teknis -- kalau tidak diperbaiki, hasil
eksperimen B2/E1/E2 akan salah secara diam-diam:

1. **SiLU in-place merusak hook Opacus** -- aktivasi bawaan YOLOv11
   (`inplace=True`) bikin DP-SGD *crash* saat backward. Diperbaiki di
   `src/fedxpalm/models/groupnorm.py` (`disable_inplace_ops`).
2. **`YOLO(path).train()` diam-diam membalikkan GroupNorm ke BatchNorm**
   setiap dipanggil, karena Ultralytics selalu membangun ulang arsitektur
   dari `.yaml` lalu cuma mentransplantasi tensor yang cocok nama+bentuk.
   Ini baru ketahuan lewat *crash* FedAvg di ronde kedua. Diperbaiki
   dengan `src/fedxpalm/federated/trainer_utils.py` -- lihat penjelasan
   lengkap di komentar file itu dan di Bab 3 Subbab 3.6.2.

Keduanya sudah divalidasi lewat smoke test 3-ronde FedAvg + DP-SGD di
sandbox (CPU, data sintetis) sebelum kode ini di-commit.

## Yang perlu kamu kirim balik untuk Bab 4-5

Setelah sweep selesai (boleh bertahap, tidak harus sekaligus):

1. Isi seluruh `results/*.csv` (atau zip `results/` seperti sel terakhir
   notebook).
2. Beri tahu K berapa yang kamu pilih sebagai "model operasional" untuk
   Subbab 4.1/4.10/4.11 (biasanya K yang mAP-nya paling dekat B1).
3. Screenshot antarmuka `deployment/` setelah `docker build` + `docker run`
   (untuk Gambar 4.6/4.7) -- atau jalankan `deployment/app.py` langsung
   tanpa Docker untuk demo cepat.

Begitu itu semua ada, Bab 4 (Hasil dan Pembahasan) dan Bab 5 (Kesimpulan
dan Saran) bisa ditulis lengkap berdasarkan angka sungguhan, bukan
placeholder.
