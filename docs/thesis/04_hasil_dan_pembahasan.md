<!--
CATATAN PENULISAN (hapus komentar ini sebelum submit):
Bab ini diisi dengan angka NYATA hasil eksperimen B1/B2/E1/E2/XAI/biaya
yang sudah dijalankan penulis di GPU (NVIDIA GeForce RTX 4080, 16 GB
VRAM). Bagian B1/B2 dikutip langsung dari
docs/thesis/manuscript/JUTIF_Manuscript_FedXPalm_B1B2.docx dan Bab 3.
Bagian E1/E2 (DP-SGD canonical, 20 ronde, seed 42), audit pipeline DP
(17/17, results/audit_dp_seedfix/dp_pipeline_sanity_audit.json),
perbandingan tersandingkan B2-E1-E2 pada held-out test, analisis XAI
matched-sample (Grad-CAM++), dan laporan biaya komputasi/komunikasi
berasal dari results/final_dp_canonical/, results/final_b2/,
results/xai_matched/, dan results/cost_report/ (lihat scripts/36, 38,
44, 45, 47, 48, 49, 50, 51, 52). Item [TODO: ...] yang masih tersisa
(penyebab pasti kelas Ripe lemah, detail spesifikasi VPS/pengujian
fungsional deployment, code/data availability statement) dipertahankan
sebagai [TODO] -- JANGAN diisi dengan tebakan.
-->

# CHAPTER 4 -- HASIL DAN PEMBAHASAN

## 4.1 Konfigurasi dan Protokol Eksperimen

### 4.1.1 Lingkungan Komputasi

Seluruh pelatihan (B1, B2, E1, E2) dijalankan pada satu *workstation*
dengan akselerator GPU **NVIDIA GeForce RTX 4080 (16 GB VRAM)**, CPU
multi-*core*, dan RAM sistem 32 GB (Subbab 3.2.1) berjalan sistem CUDA
12.1. Implementasi memakai Python 3.10 dengan versi pustaka terkunci
(*pinned*) pada `requirements.txt`:

Tabel 4.0. Konfigurasi perangkat lunak

| Pustaka | Versi | Peran |
|---|---|---|
| PyTorch | 2.5.1 (CUDA 12.1) | *Backend deep learning* |
| Ultralytics | 8.4.51 | Arsitektur dan *training loop* YOLOv11 |
| Opacus | 1.5.4 | Mekanisme DP-SGD (*per-sample gradient*, *accountant*) |
| Roboflow | 1.3.11 | Pengunduhan dataset |

Ukuran citra masukan tetap **960×960** di seluruh eksperimen (B1/B2/E1/E2),
dan *optimizer* SGD polos (bukan varian adaptif) dipakai seragam agar
selisih utilitas antarblok mencerminkan efek federasi/DP yang diteliti,
bukan pergantian *optimizer* (Subbab 3.7.1). Protokol DP-SGD *final* (E1,
E2) memakai σ=0,75, C=1,0 (*flat clipping*, Subbab 4.4), δ=1e-5, dan
*accountant* PRV (*Privacy Random Variable*).

### 4.1.2 Dataset dan Pembagian Data

Dataset mencakup enam kelas kematangan TBS sawit (*Abnormal*, *Empty
Bunch*, *Overripe*, *Ripe*, *Underripe*, *Unripe*; Subbab 2.1.2), total
10.814 citra sebelum pembagian ulang bebas-kebocoran. Citra dikelompokkan
berdasarkan identitas kelompok sumber (*source group*, diuraikan dari
`bunch_id` pada nama berkas -- Subbab 3.3.2) sebelum dipisah menjadi *split
train*/*validation*/*held-out test* pada level kelompok sumber, mencegah
kebocoran akibat beberapa foto dari tandan fisik yang sama tersebar lintas
*split*.

Tabel 4.0b. Pembagian dataset bebas-kebocoran (Tabel 3.1)

| Split | Citra | Kelompok sumber |
|---|---|---|
| Training | 8.937 | 72 |
| Validation | 826 | 8 |
| Held-out test | 1.051 | 11 |

Audit kebocoran mengonfirmasi tidak ada kelompok sumber yang tersebar
lintas *split* (Subbab 3.3.2). *Split train* dipartisi lebih lanjut menjadi
K=4 klien federasi lewat Dirichlet ($\alpha=0{,}5$, *partition seed*=42),
menghasilkan ketimpangan substansial antar klien (Tabel 3.2, direplikasi
di bawah untuk kemudahan rujukan):

Tabel 4.0c. Distribusi citra klien federasi (K=4, Dirichlet α=0,5, seed=42)

| Klien | Citra | Porsi (%) |
|---|---|---|
| Klien 0 | 860 | 9,62 |
| Klien 1 | 775 | 8,67 |
| Klien 2 | 5.305 | 59,36 |
| Klien 3 | 1.997 | 22,35 |
| **Total** | **8.937** | **100,00** |

Partisi klien yang identik ini dipakai di seluruh eksperimen B2, E1, dan
E2 (*partition seed* tetap 42), sehingga perbedaan hasil antar-blok
mencerminkan mekanisme pelatihan (federasi non-privat versus DP-SGD),
bukan perbedaan pembagian data.

### 4.1.3 Konfigurasi Model dan Federated Learning

Detektor yang dipakai adalah **YOLOv11n** (2.591.010 parameter setelah
penyesuaian kepala berkelas-6, Subbab 3.5.1), dengan seluruh 81 lapisan
`BatchNorm2d` dikonversi menjadi `GroupNorm` (Subbab 3.5.3) -- audit
arsitektur mengonfirmasi 0 modul BatchNorm dan 81 modul GroupNorm tersisa
pada setiap *checkpoint* yang dilaporkan bab ini (B1/B2/E1/E2). FedAvg
(`src/fedxpalm/federated/server.py`) mengorkestrasi K=4 klien sebagai
simulasi sekuensial pada satu GPU. Hiperparameter lokal seragam untuk
B2/E1/E2: 2 *epoch* lokal per ronde, *optimizer* SGD (*lr0*=0,01,
*momentum*=0,9, *weight decay*=0,0005), ukuran *batch* logis 64 (E1/E2,
dipecah jadi *batch* fisik 8 lewat Opacus `BatchMemoryManager`) atau 8
(B2, tanpa pemecahan *batch*), tanpa *warmup*.

B2 direplikasi pada tiga *seed* pelatihan independen (42, 123, 2026,
*partition seed* tetap 42) untuk menguji stabilitas konvergensi FedAvg di
bawah ketimpangan data (Subbab 4.2.2). Pemilihan mekanisme *clipping* untuk
E1/E2 (*flat* versus *per-layer*, Subbab 4.4) juga divalidasi pada ketiga
*seed* yang sama sebelum dikunci. Namun, karena keterbatasan waktu
komputasi (setiap *run* 20 ronde memakan 2-2,5 jam, Subbab 4.8.1),
konfigurasi E1/E2 *final* yang dilaporkan (Subbab 4.5 dst.) hanya
dijalankan pada **satu *seed* (42)** -- dicatat sebagai keterbatasan
(Subbab 4.10.3).

### 4.1.4 Protokol Pemilihan Model dan *Held-Out Test*

Seluruh eksperimen (B1, B2, E1, E2) mengikuti disiplin yang sama:
*checkpoint* dipilih **hanya** berdasarkan mAP50 *validation* tertinggi
selama pelatihan, tanpa pernah mengakses *split held-out test*. Untuk
E1/E2 dan B2 *seed* 42 secara khusus, *checkpoint* terpilih dikunci lewat
manifes *immutability* sebelum *held-out test* dijalankan: setiap *file*
JSON hasil pelatihan, *checkpoint* `.pt`, kode sumber (`dp_sgd.py`,
`scripts/38`), dan berkas konfigurasi di-*hash* SHA256, dan seluruh
kontrak protokol (nilai σ, C, δ, *accountant*, *seed*, ukuran *batch*,
jumlah parameter *trainable*/beku, status BatchNorm/GroupNorm, ketiadaan
NaN/Inf) diverifikasi ulang secara otomatis -- proses berhenti (*hard-stop*)
apabila ada satupun ketidaksesuaian (`scripts/44`, `47`). Evaluasi
*held-out test* kemudian dijalankan **tepat satu kali** per *checkpoint*,
dengan *hash checkpoint* diverifikasi ulang tepat sebelum evaluasi untuk
memastikan tidak ada pergantian *checkpoint* di antara penguncian dan
pengujian (`scripts/45`, `48`). *Split held-out test* yang sama ini
kemudian dipakai kembali **hanya** untuk analisis *post-hoc* XAI (Subbab
4.7) -- tidak pernah untuk pemilihan *checkpoint* atau konfigurasi
tambahan apapun.

## 4.2 Hasil *Baseline* Tanpa *Differential Privacy*

### 4.2.1 Hasil B1: Pelatihan Terpusat

Tabel 4.1 melaporkan hasil akhir B1 pada *split held-out test*, setelah
*checkpoint* dipilih berdasarkan performa *validation*.

Tabel 4.1. B1 *baseline* tersentral -- hasil *held-out test*

| Kelas | AP50 | AP50-95 |
|---|---|---|
| Abnormal | 0,9924 | 0,8462 |
| Empty Bunch | 0,9817 | 0,7521 |
| Overripe | 0,9406 | 0,6968 |
| Ripe | 0,5571 | 0,4757 |
| Underripe | 0,8275 | 0,7382 |
| Unripe | 0,9929 | 0,8766 |
| **mAP keseluruhan** | **0,8820** | **0,7309** |

*Precision* keseluruhan = 0,8745; *Recall* keseluruhan = 0,8653.

*Ripe* adalah kelas dengan performa terlemah pada B1, baik dari sisi AP50
maupun AP50-95, dengan selisih besar terhadap kelas lain. *Confusion
matrix* ternormalisasi (`confusion_matrix_normalized.png`, Gambar 4.1)
mengonfirmasi penyebabnya secara langsung: dari seluruh instans *Ripe*
sebenarnya, hanya 60% yang diprediksi benar sebagai *Ripe*, sedangkan 35%
salah diprediksi sebagai *Underripe* dan 5% sebagai *Overripe* -- artinya
kesalahan model pada kelas *Ripe* terkonsentrasi hampir seluruhnya pada
dua tahap kematangan yang bersebelahan dengannya, bukan pada kelas yang
secara visual tidak berkaitan. Konsisten dengan struktur kelas pada
Subbab 2.1.2, dua kelas yang secara struktural paling berbeda (*Abnormal*,
*Empty Bunch*) serta tahap kematangan yang paling tidak ambigu secara
visual (*Unripe*) mencapai nilai AP50 tertinggi (seluruhnya
$\ge 0{,}9817$), sedangkan *Ripe* -- tahap tengah yang diapit *Underripe*
dan *Overripe* -- mencapai AP50 terendah (0,5571). *Underripe* (0,8275),
juga tahap tengah, menunjukkan nilai AP50 menengah yang secara umum
konsisten dengan pola ini, dan pada *confusion matrix* yang sama juga
terlihat 35% instans *Underripe* sebenarnya salah diprediksi sebagai
*Ripe* (kesalahan batas *Ripe*/*Underripe* bersifat dua arah/simetris),
sedangkan *Overripe* (0,9406) hanya tertukar dengan *Ripe* pada 5% instans
sebenarnya -- menjelaskan mengapa *Overripe* tidak sepenuhnya sesuai
dengan pola tersebut.

**Gambar 4.1** dan **4.1b** BUKAN grafik batang gambar-tangan: keduanya
diambil langsung dari `confusion_matrix_normalized.png` dan
`box_PR_curve.png` yang dihasilkan Ultralytics sendiri saat mengevaluasi B1
pada *split held-out test* (jalankan `python
scripts/42_generate_b1_test_plots.py --weights
runs/b1_centralized_leakagefree/train/weights/best.pt --data
data/splits_v2/data.yaml`, lihat Subbab 3.6/3.8) -- angka yang ditampilkan
dijamin sama persis dengan Tabel 4.1 karena berasal dari evaluasi yang
sama, hanya dengan `plots=True` dinyalakan. Kedua berkas ini sudah diterima
dan disimpan di `docs/thesis/manuscript/` (bersama variannya
`confusion_matrix.png` / `box_F1_curve.png` / `box_P_curve.png` /
`box_R_curve.png`), dan telah diverifikasi memberikan angka yang sama
persis dengan Tabel 4.1 (mAP@0,5 = 0,882; Ripe = 0,557; dst). **Gambar
4.2**: 4 citra *held-out test* dengan kotak deteksi, label kelas, dan skor
keyakinan hasil prediksi B1 (output asli `model.predict()`, bukan gambar
tangan), dihasilkan `scripts/53_generate_b1_qualitative_examples.py`
melalui seleksi berbasis GT (satu citra berbeda per kelas, tanpa
*cherry-picking*), sudah diterima dan disimpan sebagai
`docs/thesis/manuscript/qual_ripe.jpg` / `qual_abnormal.jpg` /
`qual_unripe.jpg` / `qual_overripe.jpg`. Sudah diverifikasi memuat kotak
deteksi asli beserta skor keyakinan; `qual_ripe.jpg` menunjukkan kelas
*Ripe* (kelas terlemah B1) terdeteksi benar dengan keyakinan tinggi
(0,96). Catatan: `qual_ripe.jpg` dan `qual_abnormal.jpg` adalah dua crop
augmentasi (flip) dari *scene* multi-tandan yang sama (Underripe,
Abnormal, Ripe, dan Overripe muncul bersamaan dalam satu bingkai),
sedangkan `qual_unripe.jpg` dan `qual_overripe.jpg` masing-masing berasal
dari *scene* yang berbeda.

### 4.2.2 Hasil B2: Federated Learning Tanpa DP

Tabel 4.2 melaporkan hasil *validation* B2 pada tiga *seed* pelatihan.

Tabel 4.2. B2 (FedAvg, tanpa DP) -- hasil *validation* tiga *seed*

| Seed | Ronde terbaik | mAP50 | mAP50-95 | Precision / Recall |
|---|---|---|---|---|
| 42 | 9 | 0,8850 | 0,7431 | 0,8717 / 0,8616 |
| 123 | 11 | 0,8594 | 0,7145 | 0,8139 / 0,8422 |
| 2026 | 40 | 0,8880 | 0,7544 | 0,8868 / 0,9104 |
| **Rata-rata** | -- | **0,8775** | **0,7374** | 0,8574 / 0,8714 |
| **SD** | -- | **0,0157** | **0,0206** | 0,0385 / 0,0352 |

Ronde *checkpoint* terbaik-*validation* bervariasi cukup jauh antar *seed*
(ronde 9/11/40), mengindikasikan kecepatan konvergensi di bawah partisi
Non-IID ini sensitif terhadap trayektori stokastik pelatihan (**Gambar
4.3**, `docs/thesis/manuscript/figure_b2_convergence.png` -- kurva
konvergensi mAP50 *validation* sungguhan per ronde, bukan grafik batang
tiga titik akhir), meski performa *validation* akhir tetap sebanding di
ketiga *seed* (rentang mAP50: 0,8594-0,8880).

*Checkpoint validation*-terbaik B2 *seed* 42 (ronde 9) kemudian dikunci
(manifes *immutability*, Subbab 4.1.4) dan dievaluasi satu kali pada
*split held-out test* -- hasil ini dipakai sebagai titik pembanding
tersandingkan-penuh terhadap E1/E2 (Subbab 4.5.4) dan bukan untuk memilih
ulang *seed*/ronde manapun.

Tabel 4.2b. B2 *seed* 42 -- hasil *held-out test* (ronde 9, terkunci)

| | Nilai |
|---|---|
| mAP50 | 0,7951 |
| mAP50-95 | 0,6528 |
| Precision | 0,7648 |
| Recall | 0,7316 |

### 4.2.3 Perbandingan B1 dan B2

Selisih antara mAP50 *held-out test* B1 (0,8820) dan mAP50 *validation*
rata-rata tiga-*seed* B2 (0,8775, SD=0,0157) tergolong kecil (0,0045),
mengindikasikan performa operasional yang sebanding antara konfigurasi
tersentral dan federasi Non-IID di bawah partisi yang diuji. Salah satu
kemungkinan penyebabnya adalah FedAvg membobotkan pembaruan lokal tiap
klien berdasarkan jumlah sampel lokalnya; karena Klien 2 menyumbang 59,36%
data latih, sinyal gradiennya dapat mendominasi pembaruan teragregasi,
sebagian mengimbangi heterogenitas dari ketiga klien yang lebih kecil.

Namun, ketika B2 *seed* 42 dievaluasi tersandingkan penuh pada *split
held-out test* yang sama dengan B1 (Tabel 4.2b), selisih *validation*-ke-
*test*-nya ternyata cukup besar (0,8850 → 0,7951, turun ~0,09) -- jauh
lebih besar dari selisih indikatif B1-versus-B2 di atas. Ini dibahas lebih
lanjut pada Subbab 4.10.3.

Tabel 4.3. Ringkasan B1 vs B2 (validation dan held-out test)

| Model | Setup | Val mAP50 | Test mAP50 | mAP50-95 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| B1 | Tersentral | -- | 0,8820 | 0,7309 | 0,8745 | 0,8653 |
| B2 (seed 42) | FedAvg, tanpa DP | 0,8850 | 0,7951 | 0,6528 | 0,7648 | 0,7316 |

## 4.3 Verifikasi Implementasi DP-SGD

Sebelum melaporkan hasil DP-SGD *final*, subbab ini mendokumentasikan
proses verifikasi implementasi yang memastikan hasil E1/E2 (Subbab 4.5)
mencerminkan efek *Differential Privacy* yang sesungguhnya, bukan artefak
kesalahan implementasi.

### 4.3.1 Audit Kontrak *Loss* Ultralytics dan Opacus

Opacus mengasumsikan nilai `loss` yang diteruskan ke `backward()`
merupakan rata-rata murni per-sampel (`loss_reduction="mean"`), dan
mengoreksi rekonstruksi *per-sample gradient*-nya (`grad_sample`) dengan
mengalikan `backprops` dengan ukuran *microbatch* aktual $n$. Audit
terhadap `v8DetectionLoss.loss()` milik Ultralytics menemukan bahwa nilai
`loss` yang dikembalikan sesungguhnya berbentuk $\bar{L} \times n$ (rata-
rata per-sampel yang diskalakan ulang dengan ukuran *batch*), bukan rata-
rata murni. Ketidaksesuaian kontrak ini menyebabkan `grad_sample` yang
direkonstruksi Opacus terinflasi oleh faktor ukuran *microbatch* fisik --
dikonfirmasi melalui pengukuran langsung pada model nyata (rasio inflasi
persis sebesar ukuran *microbatch* fisik).

### 4.3.2 Evaluasi Pendekatan *Direct-Sum*

Perbaikan pertama yang diuji adalah `loss_reduction="sum"` -- pendekatan
ini secara benar memperbaiki pengukuran `grad_sample`, namun turut
menonaktifkan pembagian oleh `expected_batch_size` pada
`DPOptimizer.scale_grad()` (Opacus hanya menerapkan pembagian ini pada
mode `"mean"`), sehingga setiap pembaruan SGD akhir menjadi terlalu besar
dengan faktor sebesar ukuran *microbatch* fisik. *Pilot* lima ronde pada
konfigurasi P2 menghasilkan mAP50 terbaik hanya 0,027 dengan osilasi
*precision*/*recall* yang tidak stabil -- jauh di bawah baseline
sebelumnya (0,1766 pada jumlah ronde yang sama). Pendekatan ini **ditolak**
dan dipertahankan sebagai catatan diagnostik, bukan dihapus.

### 4.3.3 Implementasi *Canonical* DP-SGD

Konfigurasi yang divalidasi dan dipakai untuk seluruh hasil E1/E2 pada bab
ini mempertahankan `loss_reduction="mean"` (sehingga pembagian
`scale_grad()` tetap aktif), namun secara eksplisit membatalkan penskalaan
ulang Ultralytics sebelum `backward()`:

```python
loss_reduction = "mean"
total_loss = loss.sum() / actual_microbatch_size
total_loss.backward()
```

dengan `actual_microbatch_size` diambil dari ukuran *batch* fisik yang
sesungguhnya diproses pada pemanggilan `forward()` tersebut (bukan ukuran
*batch* logis maupun nilai konfigurasi manapun). Verifikasi dilakukan
melalui turunan kalkulus pada model mainan (*toy model*) dan pengukuran
tiga-arah (`legacy`/`direct-sum`/`canonical`) pada model nyata: pendekatan
*canonical* menghasilkan `grad_sample` mentah yang sama persis dengan
`direct-sum` (pengukuran per-sampel yang benar), sekaligus pembaruan
parameter akhir yang sama persis dengan `legacy` (magnitudo pembaruan yang
benar) -- memperbaiki kedua aspek secara simultan, di seluruh rezim
`clip_fraction`, bukan hanya kebetulan pada rezim tersaturasi penuh.

### 4.3.4 Audit Pipeline Privasi

Pengujian *sanity pipeline* DP-SGD (`scripts/36_dp_pipeline_sanity_audit.py`,
dijalankan pada beberapa ronde/klien nyata konfigurasi P2, bukan
*synthetic*-*only*) menunjukkan bahwa seluruh **17 pemeriksaan berhasil
dilalui (17/17 PASS)**, mencakup konsistensi *seed*, pembentukan
*per-sample gradient*, *clipping*, penambahan *noise*, akumulasi *logical
batch*, *privacy accounting*, pembatasan parameter *trainable*, dan
stabilitas numerik (`results/audit_dp_seedfix/dp_pipeline_sanity_audit.json`).
Tidak ditemukan NaN/Inf, wilayah parameter beku tetap identik-*byte*
setelah pelatihan nyata, dan wilayah parameter *trainable* terbukti
benar-benar berubah (bukan sekadar tidak-berubah pada wilayah beku).
Bagian ini menunjukkan bahwa hasil DP pada Subbab 4.5 bukan akibat
implementasi yang keliru.

## 4.4 Analisis dan Pemilihan Strategi *Clipping*

### 4.4.1 Pengaruh Nilai *Flat Clipping Threshold*

Sebelum mekanisme *canonical* di atas dikunci, ambang *clipping* norma-
*flat* $C$ disapu pada rentang $C \in \{0{,}3, 0{,}5, 0{,}7, 1, 2, 5, 10,
15, 20, 30\}$ (konfigurasi P2, σ=0,75, 5 ronde diagnostik). mAP50 terbaik
tercapai pada **$C=1$** (satu puncak bersih), menurun pada $C$ terlalu
kecil (*clipping* berlebihan memotong sinyal gradien) maupun terlalu besar
($C=10$-$20$ berosilasi, $C=30$ menyebabkan kolaps total, mAP50=0,0046).
Pada $C=30$ secara khusus, *clipping-only* (tanpa *noise*) justru
menghasilkan mAP50=0,8291 (mendekati/melampaui *baseline* tanpa DP) --
menunjukkan bahwa kolaps pada $C=30$+*noise* murni disebabkan oleh skala
*noise* $\sigma C$ yang membesar linier terhadap $C$ (30 kali lebih besar
dari $C=1$), bukan oleh *clipping* itu sendiri.

### 4.4.2 Penurunan *Threshold Per-Layer*

Sebagai kandidat pembanding, ambang *clipping* per-tensor diturunkan dari
median norma gradien per-tensor yang terukur (bukan ditebak):

$$C_i = \frac{m_i}{\lVert m \rVert_2} \, C_{\text{total}}$$

dengan $m_i$ median norma gradien per-sampel tensor $i$ yang terukur, dan
$C_{\text{total}}=1{,}0$ disamakan dengan $C$ *flat* terpilih, sehingga
$\lVert C_{\text{vec}} \rVert_2 = 1{,}0$ -- menyamakan sensitivitas total
dan skala *noise* dengan *baseline flat* $C=1$, menjadikan perbandingan ini
murni perbandingan realokasi sinyal, bukan perbandingan anggaran privasi
yang berbeda.

### 4.4.3 Perbandingan *Flat* dan *Per-Layer* Tiga *Seed*

Tabel 4.4. Perbandingan *flat* versus *per-layer clipping* -- tiga *seed*,
σ=0,75, P2, 5 ronde

| Seed | *Flat* (C=1) | *Per-layer* | Delta |
|---|---:|---:|---:|
| 42 | 0,1778 | 0,1506 | −0,0272 |
| 123 | 0,1944 | 0,1515 | −0,0429 |
| 2026 | 0,1531 | 0,1436 | −0,0095 |
| **Rata-rata** | **0,1751** | **0,1486** | **−0,0265** |
| **SD** | **0,0208** | **0,0043** | 0,0167 |

### 4.4.4 Pemilihan Konfigurasi Final

> *Flat clipping* (C=1) memberikan *consistent directional advantage*
> pada ketiga *seed* dan dipilih sebagai konfigurasi final E1 dan E2.

Keterbatasan yang perlu dicatat: perbandingan ini hanya memakai tiga
*seed*; besar selisih bervariasi cukup lebar (0,0095-0,0429), dan pada
*seed* 2026 selisihnya mendekati skala variasi *flat* sendiri antar-*seed*
(SD=0,0208); kalibrasi *per-layer* bersifat *data-dependent* (diturunkan
dari pengukuran gradien P2 spesifik) dan tidak termasuk dalam *privacy
accounting* utama (hanya memengaruhi realokasi sinyal, bukan anggaran ε);
dan hasil ini **tidak membuktikan** *flat clipping* unggul secara
universal -- hanya pada konfigurasi P2, σ=0,75, $C_{\text{total}}=1{,}0$,
5 ronde yang diuji. Menariknya, *per-layer clipping* justru jauh lebih
stabil antar-*seed* (SD=0,0043 berbanding 0,0208 milik *flat*) --
trade-off rata-rata-versus-varians, bukan kekalahan *per-layer* di semua
aspek.

## 4.5 Hasil Final *Differential Privacy*

### 4.5.1 Hasil E1: *Full* DP-SGD

E1 (seluruh parameter *trainable*, σ=0,75, C=1, 20 ronde) mencapai mAP50
*validation* terbaik 0,2195 pada ronde 19, dengan progresi yang secara
umum meningkat meski berosilasi (khas DP-SGD) sepanjang 20 ronde --
tidak terjadi NaN/Inf, dan wilayah parameter *trainable* terbukti berubah
sesuai spesifikasi *full*-DP. ε maksimum terakumulasi mencapai 22,106
pada ronde terpilih (19) dan 22,763 pada ronde akhir (20).

### 4.5.2 Hasil E2: *Partial* DP-SGD P2

E2 (hanya *stage* 16, 19, 22, 23 yang *trainable* -- 929.522 dari
2.591.010 parameter total, sisanya 1.661.488 parameter dibekukan)
mencapai mAP50 *validation* terbaik 0,2100 pada ronde 19 -- ε maksimum
identik dengan E1 pada ronde yang sama (22,106 di ronde 19, 22,763 di
ronde 20), karena ε ditentukan oleh jumlah langkah optimisasi privat per
klien per ronde, bukan oleh jumlah parameter *trainable* (Subbab 4.6.3).

### 4.5.3 Perbandingan E1 dan E2

Tabel 4.5. E1 vs E2 -- validation dan held-out test, 20 ronde, σ=0,75, C=1

| Model | Ronde terpilih | Val mAP50 | Test mAP50 | Test mAP50-95 | Precision | Recall | ε maks |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 *Full* DP | 19 | 0,2195 | 0,2116 | 0,1384 | 0,3070 | 0,5919 | 22,106 |
| E2 *Partial* DP | 19 | 0,2100 | 0,2127 | 0,1326 | 0,3342 | 0,4985 | 22,106 |

E1 lebih tinggi pada *validation* (selisih 0,0095), namun keduanya
**praktis setara pada *held-out test*** (bahkan sedikit terbalik: E2
0,2127 berbanding E1 0,2116). E2 memiliki *precision* lebih tinggi
(0,3342 vs 0,3070), E1 memiliki *recall* lebih tinggi (0,5919 vs 0,4985)
-- suatu *trade-off*, bukan satu konfigurasi yang unggul telak. Secara
keseluruhan, pembatasan cakupan parameter *trainable* (*partial* DP)
**belum meningkatkan utilitas** dibanding DP atas seluruh parameter
(*full*) pada protokol ini.

### 4.5.4 Perbandingan B2, E1, dan E2

Tabel 4.6. Penurunan utilitas akibat DP-SGD (*held-out test*, tersandingkan
penuh)

| Model | Privasi | Test mAP50 |
|---|---|---:|
| B2 | Tanpa DP | 0,7951 |
| E1 | *Full* DP | 0,2116 |
| E2 | *Partial* DP | 0,2127 |

Penambahan DP-SGD (σ=0,75, C=1) menyebabkan ***severe utility degradation***
(mAP50 turun dari 0,7951 ke kisaran 0,21) namun **bukan *total collapse*
untuk keseluruhan model** -- kedua model DP tetap menunjukkan pembelajaran
yang koheren (tidak ada NaN/Inf, kurva *validation* naik pada mayoritas
ronde) dan *recall* yang cukup tinggi pada sebagian kelas (Subbab 4.5.5).

### 4.5.5 Analisis Per Kelas

Tabel 4.7. AP50 / AP50-95 / Precision / Recall / jumlah instans per kelas
-- E1 dan E2 (*held-out test*, n=3.821 kotak GT total)

| Kelas | Instans | E1 AP50 | E1 AP50-95 | E1 P/R | E2 AP50 | E2 AP50-95 | E2 P/R |
|---|---:|---:|---:|---|---:|---:|---|
| Abnormal | 408 | 0,2015 | 0,1191 | 0,1808 / 0,4436 | 0,3180 | 0,1966 | 0,3378 / 0,4975 |
| Empty Bunch | 760 | 0,1535 | 0,0706 | 1,0000 / 0,0000 | 0,1391 | 0,0578 | 1,0000 / 0,0000 |
| Overripe | 797 | 0,1317 | 0,0712 | 0,0984 / 0,3112 | 0,0969 | 0,0455 | 0,0588 / 0,1230 |
| Ripe | 490 | 0,2017 | 0,1452 | 0,1603 / 0,9592 | 0,1668 | 0,1098 | 0,1625 / 0,7469 |
| Underripe | 679 | 0,3545 | 0,2656 | 0,2065 / 0,9205 | 0,2535 | 0,1756 | 0,2175 / 0,7599 |
| Unripe | 687 | 0,2266 | 0,1588 | 0,1961 / 0,9170 | 0,3019 | 0,2106 | 0,2284 / 0,8637 |

### 4.5.6 Analisis Kegagalan Kelas *Empty Bunch*

*Empty Bunch* mengalami **kegagalan deteksi total pada kedua model DP**:
*precision*=1,0000, *recall*=0,0000, identik pada E1 maupun E2 -- **0 dari
760 instans** berhasil ter-*recall*. *Precision*=1,0000 di sini **tidak**
diartikan sebagai kinerja sempurna: nilai ini muncul karena tidak ada
satupun prediksi positif untuk kelas ini pada ambang operasi standar,
bukan karena seluruh prediksi benar. AP50 yang tetap non-nol (0,1535 dan
0,1391) semata karena AP mengintegralkan seluruh kurva *precision-recall*
lintas ambang keyakinan. Sebagai pembanding, B2 (tanpa DP) mencapai
*recall* 0,3212 untuk kelas ini -- jauh lebih rendah dari kelas lain
(0,49-0,99) namun tidak nol. Kondisi ini disebut sebagai ***zero-recall
class collapse at the locked evaluation operating point***: DP-SGD tampak
memperparah kelemahan struktural yang sudah ada pada model non-DP menjadi
kegagalan absolut, bukan menciptakan kelemahan baru dari titik nol.

## 4.6 Analisis *Privacy-Utility Trade-off*

### 4.6.1 Perkembangan *Privacy Budget*

ε maksimum (*worst-case* lintas klien) terakumulasi dari 5,84 pada ronde 1
menjadi 22,11 pada ronde 19 (ronde terpilih) dan 22,76 pada ronde 20
(ronde akhir), identik untuk E1 maupun E2 pada setiap ronde yang sama
(Subbab 4.5.1-4.5.2) -- konsisten dengan mekanisme *accountant* PRV yang
mengakumulasi berdasarkan jumlah langkah optimisasi, bukan jumlah
parameter yang diperbarui.

### 4.6.2 *Privacy Budget* per Klien

Karena ukuran sampel antar klien berbeda substansial (Tabel 4.0c), laju
*sampling* (*sample rate* $q$) yang dipakai *accountant* privasi juga
berbeda per klien, sehingga jumlah langkah optimisasi privat per ronde
juga berbeda:

Tabel 4.8. Laju *sampling*, langkah optimisasi privat, dan ε per klien
(logical batch=64, 2 epoch lokal/ronde) -- ε identik antara E1 dan E2 pada
klien dan ronde yang sama

| Klien | Citra | Laju *sampling* $q$ | Langkah privat/ronde | Langkah kumulatif @ ronde 20 | ε @ ronde 19 (terpilih) | ε @ ronde 20 (akhir) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 860 | 0,071429 | 28 | 560 | 21,2486 | 21,8765 |
| 1 | 775 | 0,076923 | 26 | 520 | **22,1062** | **22,7626** |
| 2 | 5.305 | 0,012048 | 166 | 3.320 | **7,8085** | **8,0190** |
| 3 | 1.997 | 0,031250 | 64 | 1.280 | 13,4831 | 13,8627 |

Nilai ε per klien di atas identik persis antara E1 dan E2 (diverifikasi
langsung dari `epsilon_per_client_at_best_round`/`epsilon_per_client_at
_final_round` pada kedua berkas JSON hasil), menegaskan ulang bahwa ε
tidak bergantung pada cakupan parameter *trainable* (Subbab 4.6.3). Pola
yang menonjol: **Klien 1 (klien terkecil, 775 citra) mencapai ε
tertinggi** (22,1062 @ ronde 19), sedangkan **Klien 2 (klien terbesar,
5.305 citra) mencapai ε terendah** (7,8085) -- berlawanan dengan intuisi
"makin banyak data, makin boros anggaran privasi". Penyebabnya adalah laju
*sampling* $q$: Klien 1 memiliki $q$ tertinggi (0,076923) meski jumlah
langkah privat per rondenya paling sedikit (26), sedangkan Klien 2
memiliki $q$ terendah (0,012048) meski menempuh langkah privat terbanyak
(166) -- pada mekanisme *accountant* PRV, laju *sampling* yang tinggi
menaikkan kebocoran privasi per langkah secara lebih dominan daripada
penurunan yang diperoleh dari lebih sedikitnya jumlah langkah. ε maksimum
yang dilaporkan sebagai ringkasan *worst-case* di seluruh bab ini (Subbab
4.5.1-4.5.2, 4.6.1) karenanya berasal dari Klien 1, bukan klien dengan
data terbanyak.

### 4.6.3 Hubungan Privasi dan Utilitas

Utilitas turun signifikan pada σ=0,75 (Subbab 4.5.4) sementara E1 dan E2
mengakumulasi ε yang identik (Subbab 4.6.1) -- menegaskan bahwa ε
ditentukan oleh laju *sampling*, *noise multiplier*, jumlah langkah, dan
δ (Persamaan privasi PRV), **bukan** oleh jumlah parameter *trainable*.
Konsekuensinya, membatasi cakupan parameter *trainable* (*partial* DP, E2)
tidak secara otomatis memberikan anggaran privasi yang "lebih murah" --
sejalan dengan tidak adanya keunggulan utilitas E2 yang jelas atas E1
(Subbab 4.5.3). ε maksimum yang dicapai (22,11 pada ronde terpilih)
tergolong relatif longgar (*loose*) sebagai jaminan privasi formal;
implikasinya dibahas pada Subbab 4.10.3.

## 4.7 Analisis *Explainable AI* dengan Grad-CAM++

### 4.7.1 Protokol *Matched-Sample*

Untuk membandingkan bagaimana DP-SGD memengaruhi perhatian internal model,
Grad-CAM++ dijalankan pada sampel yang **identik** untuk B2 (*checkpoint
seed* 42, ronde 9), E1, dan E2: seluruh 3.821 kotak *ground-truth* pada
*split held-out test* (populasi yang sama dengan Tabel 4.7), dipilih murni
dari label (tanpa melibatkan prediksi model apapun), dengan identitas
setiap sampel (jalur citra, *hash* SHA256, indeks kotak) dikunci dalam
satu manifes yang dipakai untuk ketiga model
(`results/xai_matched/sample_manifest.json`). *Checkpoint* ketiga model
diverifikasi ulang *hash*-nya terhadap manifes yang sudah dikunci
sebelumnya (Subbab 4.1.4) sebelum dievaluasi.

### 4.7.2 Metrik *Faithfulness*

*Target layer* 22 (blok `C3k2` terakhir yang memberi masukan ke kepala
`Detect`), imgsz=960, `top_fraction`=0,2 dipakai seragam untuk ketiga
model. **Average Drop (AD)** mengukur seberapa besar keyakinan model
turun ketika wilayah dengan aktivasi CAM tertinggi (20% teratas) di-
*occlude* -- AD tinggi mengindikasikan wilayah yang disorot benar-benar
memuat bukti yang dipakai model. **Focus Retention Rate (FRR)** mengukur
fraksi massa aktivasi CAM yang jatuh di dalam kotak *ground-truth*, versus
bocor ke latar belakang/konteks.

### 4.7.3 Hasil Kuantitatif XAI

Dari 3.821 sampel × 3 model (11.463 pasangan sampel-model), **seluruhnya
berhasil dihitung tanpa kegagalan** (0 gagal).

Tabel 4.9. AD dan FRR global -- B2, E1, E2

| Model | AD (rata-rata ± SD) | FRR (rata-rata ± SD) |
|---|---|---|
| B2 (tanpa DP) | 0,499 ± 0,412 | 0,109 ± 0,088 |
| E1 (*Full* DP) | 0,948 ± 0,162 | 0,224 ± 0,141 |
| E2 (*Partial* DP) | 0,919 ± 0,197 | 0,167 ± 0,116 |

### 4.7.4 Analisis *Paired Delta*

Tabel 4.10. Selisih tersandingkan (per-sampel)

| Perbandingan | ΔAD (rata-rata) | ΔFRR (rata-rata) |
|---|---:|---:|
| E1 − B2 | +0,449 | +0,114 |
| E2 − B2 | +0,420 | +0,058 |
| E2 − E1 | −0,029 | −0,057 |

Analisis ini bersifat ***paired descriptive analysis conditional on the
locked checkpoints*** -- bukan uji signifikansi populasi. Secara angka
mentah, kedua model DP menunjukkan AD dan FRR yang lebih tinggi dari B2.
**Ini TIDAK ditafsirkan sebagai "model DP memiliki penjelasan yang lebih
*faithful*"**: rumus $AD = \max(0, Y_c-O_c)/Y_c$ bersifat bias terhadap
model dengan keyakinan (*confidence*) dasar yang rendah dan rapuh --
karena *precision*/*recall* E1/E2 jauh lebih rendah dari B2 (Tabel 4.6),
keyakinan dasarnya sudah rapuh sehingga *occlusion* apapun (bukan hanya
pada wilayah relevan) cenderung menjatuhkan keyakinan mendekati nol,
menghasilkan AD mendekati nilai maksimumnya (1,0) bukan karena
penjelasannya lebih akurat. FRR yang lebih tinggi pada model DP juga
bersifat ambigu tanpa inspeksi visual langsung. Selisih E2-versus-E1
relatif kecil (ΔAD=−0,029, ΔFRR=−0,057), konsisten dengan performa
deteksi kedua model yang nyaris setara (Tabel 4.5).

### 4.7.5 Visualisasi Grad-CAM++ *Matched-Sample*

**Gambar 4.4** menyandingkan Citra Asli+GT | B2 | E1 | E2 dengan skala
warna dan *alpha overlay* CAM yang identik untuk ketiga model, dan
*caption* per panel mencantumkan skor kelas, AD, FRR, serta penanda
*correct-detection* (IoU$\ge$0,5 terhadap kotak GT). Dari 12 panel yang
tersedia (`results/xai_matched/visual_panels/`, 2 sampel per kelas), badan
tesis menampilkan tiga representatif: satu kelas dengan deteksi berhasil
pada ketiga model (mis. *Unripe* atau *Underripe*), satu kelas sulit
(*Ripe* atau *Overripe*), dan satu-dua panel *Empty Bunch* (Subbab 4.7.6);
sisanya dilampirkan pada **Lampiran A**.

### 4.7.6 Interpretasi XAI pada *Empty Bunch*

Untuk kelas *Empty Bunch* khususnya, setiap panel diberi catatan eksplisit
bahwa *recall held-out test* E1/E2 untuk kelas ini adalah 0,0000 (Tabel
4.6, Subbab 4.5.6). CAM yang ditampilkan pada E1/E2 untuk kelas ini
karenanya menggambarkan ***respons internal model terhadap kelas Empty
Bunch, bukan penjelasan atas deteksi yang berhasil*** -- prinsip yang
berlaku umum untuk setiap sampel dengan `correct_detection=False` pada
Tabel 4.10, tidak terbatas pada *Empty Bunch* saja.

## 4.8 Analisis Biaya Komputasi dan Komunikasi

### 4.8.1 Biaya Komputasi

Tabel 4.11. Waktu pelatihan -- B1, B2, E1, E2 (dibaca dari log masing-
masing *run*, tidak diukur ulang)

| Model | Unit pelatihan | Total waktu | Waktu/ronde | *Overhead* vs B2 |
|---|---:|---:|---:|---:|
| B1 | 97 *epoch* | 7.958,3 dtk | -- | -- |
| B2 | 40 ronde | 12.482,8 dtk | 312,1 dtk | *baseline* |
| E1 | 20 ronde | 9.041,9 dtk | 452,1 dtk | +44,9% |
| E2 | 20 ronde | 7.676,1 dtk | 383,8 dtk | +23,0% |

E2 tercatat **~15,1% lebih cepat per ronde** dibanding E1 (383,8 dtk
berbanding 452,1 dtk) -- mekanisme *per-sample gradient* Opacus tidak
perlu dihitung untuk 1.661.488 parameter yang dibekukan pada E2, sehingga
E2 memangkas sekitar separuh (~48,8%) dari *overhead* tambahan yang
dikenakan DP-SGD dibanding B2 (23,0% berbanding 44,9%). Angka ini adalah
*observed wall-clock time*, bukan estimasi teoretis.

### 4.8.2 Biaya Komunikasi

Tabel 4.12. Volume komunikasi (teoretis -- simulasi satu-GPU ini tidak
benar-benar mengirim *byte* lewat jaringan)

| Model | Ronde | Komunikasi/ronde | Total |
|---|---:|---:|---:|
| B2 | 40 | 0,0772 GB | 3,089 GB |
| E1 | 20 | 0,0772 GB | 1,544 GB |
| E2 | 20 | 0,0772 GB | 1,544 GB |

### 4.8.3 Potensi Optimasi Komunikasi E2

Diverifikasi langsung dari `fedxpalm/federated/fedavg.py`: implementasi
agregasi saat ini menjumlah-berbobot **seluruh** kunci `state_dict` tanpa
memfilter berdasarkan `requires_grad`, sehingga parameter yang dibekukan
pada E2 **tetap ditransmisikan dan diagregasi penuh** setiap ronde --
komunikasi aktual E1 dan E2 karenanya identik (1,544 GB) meski E2
membekukan 64,1% parameter. Sebagai ilustrasi terpisah (bukan hasil
implementasi aktual): skenario hipotetis di mana hanya parameter
*trainable* yang dikomunikasikan menghasilkan estimasi 0,554 GB untuk E2
(~64,1% lebih hemat) -- potensi optimasi yang belum diimplementasikan
pada kode saat ini, dicatat sebagai arah kerja mendatang (Subbab 4.10.4).

## 4.9 Implementasi dan *Deployment* Sistem

[TODO: spesifikasi VPS sudah dikonfirmasi (Tabel 4.13), namun status
*build* Docker dan hasil pengujian fungsional masih perlu dikonfirmasi.
Jangan menambahkan klaim *latency* atau *throughput* bila belum dilakukan
pengukuran formal.]

### 4.9.1 Arsitektur *Deployment*

`deployment/app.py` mengimplementasikan alur: pengguna mengunggah citra →
*preprocessing* (resize ke imgsz model) → *inference* YOLOv11 →
keluaran kotak deteksi dan kelas → visualisasi Grad-CAM++ opsional
menggunakan mekanisme yang sama dengan Subbab 4.7 (`fedxpalm.xai.gradcam`).
[TODO: konfirmasi *checkpoint* mana yang dipakai untuk *deployment* --
B1, B2, atau salah satu varian DP.]

### 4.9.2 *Containerization* Menggunakan Docker

`deployment/Dockerfile` mendefinisikan *image* aplikasi. [TODO: rincian
*dependency*, pemetaan *port*, dan strategi pemasangan *volume*/*checkpoint*
model perlu dikonfirmasi dari `Dockerfile` dan catatan *deployment*
penulis.]

### 4.9.3 *Deployment* pada VPS

Aplikasi di-*deploy* pada sebuah VPS (*Virtual Private Server*) dengan
spesifikasi berikut:

Tabel 4.13. Spesifikasi VPS *deployment*

| Parameter | Nilai |
|---|---|
| CPU | 2 vCPU |
| Memori | 4 GB RAM |
| Penyimpanan | 60 GB SSD |
| Sistem operasi | Ubuntu 24.04 |
| Wilayah (*region*) | Jakarta |

[TODO: struktur *service* (mis. `systemd`/`docker run` langsung/*reverse
proxy*) dan mekanisme *startup*/*restart* belum dikonfirmasi.]

### 4.9.4 Pengujian Fungsional

[TODO: konfirmasi hasil pengujian -- apakah model berhasil dimuat,
*endpoint* menerima masukan dan mengembalikan hasil deteksi dengan benar,
serta keluaran tervisualisasi sesuai ekspektasi. Jangan mengklaim
*latency*/*throughput* tanpa pengukuran formal.]

## 4.10 Pembahasan Menyeluruh

### 4.10.1 Jawaban terhadap Tujuan Penelitian

1. **Kemampuan YOLOv11n pada *centralized* dan *federated*** (Subbab 4.2,
   4.10.2): performa sebanding secara indikatif (B1 mAP50-test=0,8820,
   B2 mAP50-val rata-rata=0,8775), meski perbandingan tersandingkan penuh
   B2 *seed* 42 menunjukkan gap *validation*-ke-*test* yang lebih besar
   dari ekspektasi (Subbab 4.2.3).
2. **Pengaruh DP-SGD terhadap utilitas** (Subbab 4.5.4): *severe
   degradation* namun bukan *total collapse* -- mAP50 turun dari 0,7951
   (B2) ke ~0,21 (E1/E2) pada σ=0,75, C=1.
3. **Perbandingan *Full* DP dan *Partial* DP** (Subbab 4.5.3, 4.6.3):
   praktis setara pada *held-out test*; pembatasan parameter *trainable*
   tidak memberikan keunggulan privasi maupun utilitas yang jelas pada
   protokol ini, hanya keunggulan komputasi (Subbab 4.8.1).
4. **Pengaruh strategi *clipping*** (Subbab 4.4): *flat clipping* (C=1)
   konsisten lebih baik secara arah pada tiga *seed*, namun *per-layer
   clipping* lebih stabil (SD lebih kecil) -- *trade-off*, bukan
   kesimpulan mutlak.
5. **Interpretabilitas Grad-CAM++** (Subbab 4.7): AD/FRR numerik lebih
   tinggi pada model DP, namun secara metodologis tidak dapat ditafsirkan
   langsung sebagai "lebih *faithful*" karena keterbatasan metrik AD
   terhadap model *low-confidence* (Subbab 4.7.4).
6. **Biaya komputasi dan komunikasi** (Subbab 4.8): DP-SGD menambah biaya
   komputasi nyata (23-45% lebih lambat per ronde dari B2); *partial* DP
   menghemat komputasi (~15% vs *full* DP) namun **belum** menghemat
   komunikasi pada implementasi saat ini.

### 4.10.2 Temuan Utama

- B2 (FedAvg tanpa DP) mempertahankan performa yang sebanding dengan B1
  secara indikatif, meski gap *validation*-ke-*test*-nya (seed 42) lebih
  besar dari yang terlihat pada perbandingan indikatif semata.
- DP-SGD menyebabkan *severe utility degradation* (bukan *total
  collapse*) pada σ=0,75, C=1.
- E1 dan E2 praktis setara pada *held-out test*, meski ε yang diakumulasi
  identik dan cakupan parameter *trainable*-nya sangat berbeda
  (2.590.994 vs 929.522).
- E2 lebih hemat komputasi ~15% dibanding E1, namun belum menghemat
  komunikasi pada implementasi *fedavg* saat ini.
- *Flat clipping* lebih konsisten arahnya dibanding *per-layer* pada tiga
  *seed*, tetapi *per-layer* lebih stabil variansnya.
- *Empty Bunch* menjadi *failure case* utama: sudah lemah pada B2
  (*recall*=0,3212), *collapse* total pada E1/E2 (*recall*=0,0000).
- XAI menunjukkan perbedaan AD/FRR yang mencolok antar-model, namun
  interpretasinya memerlukan kehati-hatian metodologis terhadap
  keterbatasan metrik AD.

### 4.10.3 Keterbatasan Penelitian

Pertama, pelatihan federasi disimulasikan secara sekuensial pada satu
GPU; volume komunikasi yang dilaporkan (Subbab 4.8.2) bersifat teoretis,
bukan pengukuran jaringan sungguhan. Kedua, dataset berasal dari satu
sumber, sehingga generalisasi ke perkebunan, kultivar, kamera, atau
kondisi pencahayaan lain belum dievaluasi langsung -- tidak ada klaim
generalisasi ke seluruh kondisi lapangan. Ketiga, meski B2 dan pemilihan
strategi *clipping* (Subbab 4.4) divalidasi tiga *seed*, konfigurasi
E1/E2 *final* (Subbab 4.5-4.7) hanya dijalankan pada **satu *seed* (42)**
karena keterbatasan waktu komputasi -- replikasi multi-*seed* untuk hasil
DP-SGD *final* akan memperkuat klaim generalisasi. Keempat, ε maksimum
yang dicapai (22,11) tergolong relatif longgar sebagai jaminan privasi
formal; `secure_mode` Opacus (RNG kriptografis) tidak diaktifkan pada
seluruh eksperimen tahap ini (dicatat, bukan diaktifkan diam-diam) --
konsisten untuk perbandingan internal, namun perlu diaktifkan sebelum
klaim jaminan privasi produksi. Kelima, keempat klien adalah pecahan data
tersimulasi dari satu *split train*, bukan representasi organisasi fisik
yang berbeda. Keenam, kalibrasi ambang *per-layer* (Subbab 4.4.2) bersifat
*data-dependent* dan tidak masuk dalam *privacy accounting* formal.
Ketujuh, kelas *Empty Bunch* mengalami *zero-recall collapse* pada kedua
model DP (Subbab 4.5.6) -- kesimpulan privasi-utilitas bab ini tidak
berlaku merata di seluruh kelas. Kedelapan, analisis XAI bersifat
*post-hoc* semata (Subbab 4.7) dan tidak memengaruhi pemilihan
*checkpoint*/konfigurasi manapun; metrik *Average Drop* memiliki
keterbatasan yang telah diketahui dalam membandingkan model dengan
tingkat keyakinan prediksi yang jauh berbeda (Subbab 4.7.4).

### 4.10.4 Implikasi dan Peluang Pengembangan

Arah kerja mendatang yang teridentifikasi langsung dari temuan bab ini:
*clipping* adaptif yang tetap *privacy-preserving* (mengatasi *trade-off*
rata-rata-versus-varians pada Subbab 4.4.4); strategi *class-aware*
(*sampling* atau fungsi kerugian berbobot) untuk mengatasi *zero-recall
collapse* pada *Empty Bunch* (Subbab 4.5.6); implementasi komunikasi yang
benar-benar hemat parameter untuk *partial* DP (Subbab 4.8.3, potensi
~64% penghematan); *secure aggregation* dan DP tingkat-klien sebagai
lapisan privasi tambahan; evaluasi multi-*seed* untuk konfigurasi E1/E2
*final*; serta validasi pada data lapangan yang lebih beragam (perkebunan,
kultivar, kondisi pencahayaan lain).

## 4.11 Ringkasan Bab

Bab ini menetapkan titik rujukan tersentral (B1, mAP50-test=0,8820) dan
federasi Non-IID non-privat (B2, mAP50-val rata-rata tiga-*seed*=0,8775),
kemudian melaporkan hasil perluasan DP-SGD *final* (E1 *Full* dan E2
*Partial*, keduanya σ=0,75, C=1, *seed* 42, 20 ronde, protokol *canonical*
yang telah diverifikasi 17/17 pada audit *pipeline*) yang tersandingkan
penuh dengan B2 pada *split held-out test* yang sama: DP-SGD menyebabkan
*severe utility degradation* (mAP50 dari 0,7951 ke ~0,21) namun bukan
*total collapse*, dengan *Empty Bunch* sebagai *failure case* ekstrem
(*recall*=0,0000). Pemilihan mekanisme *clipping* (*flat*, tervalidasi
tiga *seed*) dan cakupan parameter *trainable* (*full* vs *partial*)
dianalisis sebagai dua sumbu desain yang independen dari ε yang
teramankan. Analisis Grad-CAM++ tersandingkan pada 3.821 sampel identik
lintas ketiga model mengungkap perbedaan AD/FRR yang mencolok namun
memerlukan interpretasi hati-hati akibat keterbatasan metrik AD. Laporan
biaya menunjukkan *partial* DP menghemat komputasi (~15%) namun belum
menghemat komunikasi pada implementasi saat ini. Bersama-sama, temuan ini
menjadi dasar kesimpulan dan rekomendasi pada Bab V.
