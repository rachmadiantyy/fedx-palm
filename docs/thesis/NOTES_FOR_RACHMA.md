# Catatan Serah Terima -- FedX-Palm v2

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

1. **Nama & urutan 6 kelas** -- cek `data.yaml` hasil
   `scripts/01_download_dataset.py`, cocokkan dengan `names` di
   `configs/dataset.yaml`. Kalau beda, kode lain (partisi, evaluasi) tetap
   jalan benar (semua baca dari `configs/dataset.yaml`), tapi **teks Bab
   1/Bab 2** yang menyebut nama kelas perlu disesuaikan.
2. **Spesifikasi GPU** yang sungguh-sungguh kamu pakai (tipe RTX, VRAM) --
   isi di Bab 1 Subbab 1.5.1 dan Bab 3 Subbab 3.2.1.
3. **Karakteristik dataset** (jumlah citra, jumlah instans per kelas) --
   isi di Bab 3 Subbab 3.3.1 setelah unduhan selesai.

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
