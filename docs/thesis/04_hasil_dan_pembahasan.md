<!--
TEMPLATE -- BELUM BISA DITULIS PENUH karena belum ada hasil eksperimen
nyata (training belum dijalankan di GPU). Setiap subbab di bawah berisi:
(a) apa yang perlu dilaporkan, (b) file results/*.json atau
results/table_4_*.csv mana yang jadi sumber angkanya (dihasilkan
scripts/10_export_results.py), dan (c) placeholder tabel/kalimat yang
tinggal diisi. JANGAN mengisi angka dengan estimasi/tebakan -- tulis
hanya setelah scripts/05-09 benar-benar dijalankan.
Lihat docs/thesis/NOTES_FOR_RACHMA.md untuk alur kerja lengkapnya.
-->

# CHAPTER 4 -- HASIL DAN PEMBAHASAN (TEMPLATE)

## 4.1 Definisi Operasional Hasil

*(Tuliskan definisi "model operasional" yang dipilih untuk Subbab 4.10 dan
4.11 -- pada kerangka generasi pertama, ini adalah checkpoint B2 pada K
tertentu yang mAP@0.5-nya paling dekat dengan baseline tersentral B1.
Tentukan K tersebut setelah melihat `results/table_4_2_b2_federated.csv`.)*

## 4.2 *Baseline* Sentralized (B1)

Sumber: `results/b1_centralized.json` / `results/table_4_1_b1_centralized.csv`.

| Metrik | Nilai |
|---|---|
| mAP@0.5 | *(isi)* |
| mAP@0.5:0.95 | *(isi)* |
| Precision | *(isi)* |
| Recall | *(isi)* |

## 4.3 *Baseline* Federated Tanpa DP (B2)

### 4.3.1 Hasil per-K

Sumber: `results/table_4_2_b2_federated.csv`. Isi tabel mAP@0.5 /
mAP@0.5:0.95 / Precision / Recall untuk tiap $K \in \{2,4,8,12,16\}$.

### 4.3.2 Pembahasan B2

#### 4.3.2.1 Degradasi Utilitas terhadap Jumlah Klien
*(Apakah mAP menurun monoton seiring K membesar? Bandingkan dengan B1 --
berapa selisih "FL-cost" pada tiap K?)*

#### 4.3.2.2 Pengaruh Jumlah Ronde Komunikasi
*(Jika sempat menjalankan variasi jumlah ronde, bahas konvergensi di sini.
Jika tidak, hapus subbab ini atau catat sebagai keterbatasan pada 4.12.)*

#### 4.3.2.3 Pola *Precision* versus *Recall*
*(Apakah federasi menurunkan recall lebih besar dari precision, atau
sebaliknya? Kaitkan dengan heterogenitas Dirichlet per Subbab 3.4.)*

## 4.4 Eksperimen Pendahuluan: DP-FedAvg Tingkat-Klien

*(Jika dijalankan sebagai pembanding pendahuluan per Subbab 3.8, laporkan
di sini dan bandingkan bentuk degradasinya dengan E1 pada Subbab 4.5.3.2.
Jika tidak dijalankan, nyatakan eksplisit bahwa pembanding ini
tidak dieksekusi ulang pada iterasi kedua kerangka ini dan cukup merujuk
temuan kualitatif pada Bab 1/Bab 2 yang memotivasi pemilihan DP-SGD
per-sampel.)*

## 4.5 DP-SGD Federated Penuh (E1)

Sumber: `results/table_4_3_e1_dp_full.csv`.

### 4.5.1 Grid Utilitas mAP@0.5
*(Tabel K x sigma -> mAP@0.5)*

### 4.5.2 Pemetaan σ → ε per-K
*(Tabel K x sigma -> epsilon, dari kolom `epsilon` pada CSV yang sama.)*

### 4.5.3 Kurva Privasi-Utilitas

#### 4.5.3.1 Batas Atas Utilitas E1
*(mAP@0.5 tertinggi yang dicapai E1 pada sigma terkecil -- seberapa jauh
dari B1/B2?)*

#### 4.5.3.2 Bentuk Degradasi: *Gradual* versus *Cliff*
*(Plot mAP@0.5 vs epsilon per K -- lihat sel notebook Bab 11. Apakah
degradasi bertahap atau ada titik jatuh mendadak?)*

#### 4.5.3.3 Konsistensi Urutan Kurva antar-K
*(Apakah urutan kurva K konsisten di seluruh rentang epsilon, atau ada
persilangan?)*

### 4.5.4 Kurva-K
*(mAP@0.5 sebagai fungsi K untuk tiap sigma -- menguji H2-K, Subbab 1.6.)*

### 4.5.5 Analisis Penyebab E1 Berhenti di Rezim *Degraded*
*(Jika ada titik di mana E1 gagal konvergen/collapse, diskusikan penyebab
teknisnya di sini -- mis. rasio noise-to-signal pada gradien kepala
deteksi.)*

## 4.6 DP-SGD Federated Parsial (E2): *Backbone* Beku

Sumber: `results/table_4_5_e2_dp_partial.csv`.

### 4.6.1 Grid Utilitas mAP@0.5

### 4.6.2 Perbandingan Langsung E1 dan E2 pada (K, σ) Identik
*(Tabel sisi-berdampingan E1 vs E2 pada K,sigma yang sama.)*

### 4.6.3 Pembahasan: Penolakan/Penerimaan Hipotesis *Partial*-DP

#### 4.6.3.1 Ketidaksesuaian *Backbone* COCO dengan Domain Sawit
#### 4.6.3.2 *Noise* Relatif pada Gradien Kepala Deteksi
#### 4.6.3.3 Keterbatasan Anggaran *Epoch* untuk *Head-only*

## 4.7 Sintesis Privasi-Utilitas: Tiga Rezim Operasi

*(Rangkum B1/B2 sebagai rezim "utilitas tinggi", E1/E2 pada sigma kecil
sebagai "moderat", dan sigma besar sebagai "terdegradasi" -- tabel
ringkasan mirip Tabel 4.7 kerangka generasi pertama.)*

## 4.8 Catatan Ketangguhan Statistik

*(Jika sempat menjalankan replikasi/seed berbeda, laporkan varians di
sini. Jika tiap sel grid hanya satu run, nyatakan eksplisit sebagai
keterbatasan pada Subbab 4.12 -- jangan mengklaim signifikansi statistik
tanpa replikasi.)*

## 4.9 Validasi Hipotesis

| Hipotesis | Didukung? | Bukti |
|---|---|---|
| H1 (kelayakan B2 Non-IID) | *(isi)* | *(rujuk 4.3)* |
| H2 (bentuk trade-off gradual) | *(isi)* | *(rujuk 4.5.3.2)* |
| H2-K (arah pengaruh K berkebalikan) | *(isi)* | *(rujuk 4.5.4)* |
| H3 (validitas interpretasi visual) | *(isi)* | *(rujuk 4.10)* |

## 4.10 Analisis Transparansi Model (XAI)

Sumber: `results/xai_<tag>.json` / `results/table_4_9_xai_faithfulness.csv`
(dihasilkan `scripts/09_evaluate_xai.py`).

### 4.10.1 Hasil Global dan Per-Kelas (model operasional, Subbab 4.1)
### 4.10.2 Komparasi dengan *Centralized* (B1)
*(Jalankan `scripts/09_evaluate_xai.py --weights <B1 best.pt> --tag b1` juga,
lalu bandingkan AD/FRR B1 vs model operasional federasi.)*
### 4.10.3 Analisis Per-Kelas
*(Kelas mana yang AD/FRR-nya paling rendah? Kaitkan dengan kemiripan visual
antarkelas, mis. Ripe vs Overripe.)*

## 4.11 Demonstrasi Operasional: *Deployment* Layanan Inferensi

### 4.11.1 Arsitektur *Deployment*
*(Rujuk Subbab 3.12 -- deployment/Dockerfile, deployment/app.py.)*
### 4.11.2 Antarmuka Layanan
*(Screenshot deployment/templates/index.html setelah dijalankan --
lampirkan sebagai Gambar 4.6.)*
### 4.11.3 Inferensi *Live* dan Visualisasi Grad-CAM++
*(Screenshot hasil /predict pada beberapa citra uji -- Gambar 4.7.)*
### 4.11.4 Konsistensi Visual dengan Validasi Kuantitatif
*(Apakah heatmap yang tampil di layanan konsisten dengan AD/FRR kuantitatif
pada 4.10?)*
### 4.11.5 Implikasi Kelayakan Praktis

## 4.12 Keterbatasan Penelitian

*(Simulasi FL pada satu GPU, bukan multi-host fisik; satu run per sel grid
tanpa replikasi statistik (lihat 4.8); dataset baru belum tentu memiliki
karakteristik kelas yang identik dengan asumsi Bab 1/2 -- cocokkan; dst.)*
