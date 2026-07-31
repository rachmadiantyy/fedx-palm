<!--
CATATAN PENULISAN (hapus komentar ini sebelum submit):
Bab ini diperluas agar konsisten dengan cakupan penuh Bab 1, 2, dan 4:
mencakup metodologi DP-SGD (Opacus, verifikasi kontrak loss, pemilihan
strategi clipping, E1/E2, akuntansi privasi per klien), protokol XAI
Grad-CAM++ matched-sample, analisis biaya komputasi/komunikasi, dan
containerization/deployment -- yang pada draf sebelumnya sengaja dihapus
dan dipindah ke "cakupan yang ditunda" (lihat riwayat pada
docs/thesis/NOTES_FOR_RACHMA.md). Seluruh angka pada bab ini dikutip
langsung dari eksperimen nyata yang sudah dijalankan penulis di GPU
(lihat docs/thesis/04_hasil_dan_pembahasan.md untuk hasilnya) -- cross-
check terhadap configs/*.yaml dan scripts/*.py sebelum submit bila
konfigurasi berubah.
-->

# CHAPTER 3 -- METODOLOGI PENELITIAN

## 3.1 Rancangan Penelitian

Penelitian ini dirancang sebagai eksperimen kuantitatif empat-blok,
dimulai dari pembagian dataset bebas-kebocoran dan partisi Non-IID
Dirichlet, dilanjutkan empat blok eksperimen -- **B1** (*baseline*
tersentral), **B2** (*baseline* federasi FedAvg tanpa DP), **E1** (DP-SGD
*full*-parameter di atas B2), dan **E2** (DP-SGD *partial*-parameter di
atas B2) -- lalu ditutup dengan analisis interpretasi visual (XAI),
analisis biaya, dan demonstrasi *deployment*. Seluruh kode implementasi
diorganisasikan sebagai paket Python `fedxpalm` (`src/fedxpalm/`),
dikendalikan oleh berkas konfigurasi YAML (`configs/dataset.yaml`,
`configs/fl_config.yaml`), dan dijalankan melalui skrip bernomor
(`scripts/01_*.py` hingga `scripts/53_*.py`) yang mengikuti urutan tahapan
pada Subbab 1.5.2.

## 3.2 Lingkungan dan Perangkat Implementasi

### 3.2.1 Spesifikasi Perangkat Keras

Seluruh pelatihan (B1, B2, E1, E2) dijalankan pada satu *workstation*
dengan akselerator GPU **NVIDIA GeForce RTX 4080 (16 GB VRAM)**, CPU
multi-*core*, dan RAM sistem 32 GB. Simulasi federasi K = 4 klien
direalisasikan sebagai *loop* FedAvg sekuensial pada GPU tunggal tersebut
(Subbab 3.7.2), bukan sebagai proses terdistribusi lintas-*host* fisik.
Demonstrasi *deployment* (Subbab 3.14) dijalankan terpisah pada sebuah
*Virtual Private Server* (VPS): 2 vCPU, 4 GB RAM, 60 GB SSD, Ubuntu 24.04,
wilayah Jakarta.

### 3.2.2 Konfigurasi Perangkat Lunak

Implementasi memakai Python 3.10 dengan versi pustaka yang dikunci
(*pinned*) pada `requirements.txt` demi reprodusibilitas:

| Pustaka | Versi | Peran |
|---|---|---|
| PyTorch | 2.5.1 (CUDA 12.1) | *Backend deep learning* |
| Ultralytics | 8.4.51 | Arsitektur dan *training loop* YOLOv11 |
| Roboflow | 1.3.11 | Pengunduhan dataset |
| Opacus | -- | *Privacy engine* DP-SGD (E1/E2, Subbab 3.10) |

Pelatihan federasi dijalankan sebagai simulasi *sequential* FedAvg pada satu
GPU (`src/fedxpalm/federated/server.py`). *Deployment* (Subbab 3.14)
memakai *image* Docker CPU-*only* terpisah (`python:3.11-slim`), tidak
bergantung pada CUDA.

## 3.3 Dataset dan Strategi Pembagian Data

### 3.3.1 Karakteristik Dataset

Dataset diperoleh dari platform Roboflow (proyek *palm-fruit-ripeness-
detection-f6sac-ccb2z* versi 2, *workspace* *dydy-worker*), diunduh dalam
format anotasi `yolov11` melalui `scripts/01_download_dataset.py`. Dataset
mencakup enam kelas kematangan TBS sebagaimana dijabarkan pada Subbab
2.1.2, dengan total **10.814 citra** sebelum pembagian ulang
bebas-kebocoran.

### 3.3.2 Pembagian Bebas-Kebocoran Berbasis Kelompok Sumber

Citra dikelompokkan berdasarkan identitas kelompok sumbernya (*source
group*, diuraikan dari identitas `bunch_id` pada nama berkas citra --
`src/fedxpalm/data/split.py`, `leakage_free_split`) sebelum dipisah, dan
partisi *train*/*validation*/*held-out test* dibentuk pada **level
kelompok sumber** (bukan level citra), untuk mencegah kebocoran terkait
sumber antar-*split*. Pendekatan ini penting karena *split* acak per-*frame*
berisiko menempatkan beberapa foto dari tandan fisik yang sama pada *split*
*train* dan *test* sekaligus -- model kemudian sebagian "menghafal" tandan
yang justru dipakai mengujinya, mengembang-gelembungkan metrik evaluasi
secara optimistis-palsu. *Split held-out test* yang sama ini kemudian
dipakai kembali untuk seluruh evaluasi akhir (B1, B2, E1, E2) dan analisis
XAI *post-hoc* (Subbab 3.12), sehingga seluruh perbandingan antar blok
tersandingkan pada populasi sampel yang identik. Hasil pembagian dirangkum
pada Tabel 3.1.

Tabel 3.1. Pembagian dataset bebas-kebocoran

| Split | Citra | Kelompok sumber |
|---|---|---|
| Training | 8.937 | 72 |
| Validation | 826 | 8 |
| Held-out test | 1.051 | 11 |

Audit kebocoran dijalankan untuk mengonfirmasi: tidak ada kelompok sumber
yang tumpang tindih antara *split* *training*, *validation*, dan *test*;
tidak ada duplikat citra identik lintas-*split*; dan tidak ditemukan
duplikat lintas-*split* yang terkonfirmasi menurut prosedur audit yang
diimplementasikan. Audit ini dijalankan sebagai bagian integral dari
`scripts/02_prepare_splits.py`, bukan langkah manual terpisah.

## 3.4 Partisi Data Non-IID Berbasis Distribusi Dirichlet (K = 4)

### 3.4.1 Formulasi Partisi Dirichlet

`src/fedxpalm/data/partition.py` (`dirichlet_partition`) mengimplementasikan
skema *latent Dirichlet allocation* (Subbab 2.3.3) atas *split train*
(8.937 citra) saja; *split validation* dan *held-out test* tidak
dipartisi. Karena satu citra deteksi objek dapat memuat kotak dari beberapa
kelas sekaligus, setiap citra terlebih dahulu diberi **kelas primer** =
kelas yang paling sering muncul di antara kotak anotasi YOLO-nya. Federasi
memakai **K = 4 klien** di bawah partisi Dirichlet ($\alpha = 0{,}5$,
*partition seed* = 42). Klien-klien ini adalah pecahan data (*data shard*)
tersimulasi pada level sampel, bukan representasi empat perkebunan,
afdeling, organisasi, atau perangkat fisik; pelatihan disimulasikan secara
sekuensial pada satu GPU. Citra yang berasal dari satu kelompok sumber di
dalam *split train* dapat terdistribusi ke klien berbeda; hal ini tidak
melanggar pembagian bebas-kebocoran *train*/*validation*/*test* pada Subbab
3.3, karena seluruh partisi klien ditarik semata-mata dari *split train*.
Partisi klien yang sama ini dipakai identik oleh B2, E1, dan E2, sehingga
laju *sampling* per klien yang relevan bagi akuntansi privasi (Subbab 3.11)
konsisten di seluruh blok DP-SGD.

### 3.4.2 Distribusi Citra Antar Klien

Tabel 3.2 melaporkan distribusi citra hasil partisi Dirichlet aktual pada
K = 4, $\alpha = 0{,}5$, *seed* = 42. Ketimpangan yang dihasilkan
substansial: Klien 1 hanya memegang 8,67% dari *split train*, sedangkan
Klien 2 memegang 59,36%.

Tabel 3.2. Distribusi citra klien federasi (K = 4, Dirichlet α = 0,5, seed = 42)

| Klien | Citra | Porsi (%) |
|---|---|---|
| Klien 0 | 860 | 9,62 |
| Klien 1 | 775 | 8,67 |
| Klien 2 | 5.305 | 59,36 |
| Klien 3 | 1.997 | 22,35 |
| **Total** | **8.937** | **100,00** |

*Sweep* atas jumlah klien K lain (mis. {2, 8, 12, 16}) tersedia pada
infrastruktur kode (`configs/fl_config.yaml: clients.k_values`) namun tidak
dijalankan/dilaporkan pada penelitian ini (Subbab 1.4).

## 3.5 Arsitektur Detektor YOLOv11n dan Konversi GroupNorm

### 3.5.1 Struktur *Backbone-Neck-Head*

Detektor yang dipakai adalah **YOLOv11n** (varian *nano*, `yolo11n.pt`
sebagai titik awal pra-latih COCO, resolusi masukan 960x960), dengan
struktur *backbone* (tahap indeks 0-10: `Conv`, `C3k2`, `SPPF`, `C2PSA`),
*neck* PAN-FPN (tahap 11-22), dan *head* `Detect` *anchor-free* *decoupled*
(tahap 23), sebagaimana dijabarkan pada Subbab 2.2.1. Bobot pra-latih COCO
ditransplantasikan ke kepala deteksi berkelas-6
(`scripts/04_prepare_base_model.py`): tensor bobot yang cocok nama dan
bentuknya antara `yolo11n.pt` (80 kelas COCO) dan arsitektur target (6
kelas TBS sawit) disalin langsung, sedangkan lapisan klasifikasi pada
*head* yang berdimensi kelas diinisialisasi ulang dari awal. Model hasil
penyesuaian ini memuat **2.591.010 parameter**.

### 3.5.2 Fungsi Kerugian Komposit

Fungsi kerugian mengikuti Persamaan (2.2): CIoU untuk regresi kotak,
*Binary Cross-Entropy* untuk klasifikasi, dan *Distribution Focal Loss*
untuk representasi probabilistik tepi kotak, dengan pemasangan target
melalui *Task-Aligned Assigner* bawaan Ultralytics (Subbab 2.2.2). Sebuah
konvolusi tetap di dalam modul DFL ditetapkan `requires_grad=False` dan
dikecualikan dari optimisasi -- baik pada pelatihan non-DP (B1/B2) maupun
DP-SGD (E1/E2, Subbab 3.10).

### 3.5.3 Konversi *BatchNorm* ke *GroupNorm*

Sebagaimana dijabarkan pada Subbab 2.5, seluruh **81 lapisan
`BatchNorm2d`** pada YOLOv11n dikonversi menjadi `GroupNorm` melalui
`src/fedxpalm/models/groupnorm.py` (`convert_batchnorm_to_groupnorm`),
dengan jumlah grup tiap lapisan dipilih sebagai pembagi terbesar dari
jumlah kanal lapisan tersebut yang $\le 32$. Audit arsitektur akhir
mengonfirmasi **0 modul BatchNorm tersisa** dan **81 modul GroupNorm**.
Konversi ini dijalankan **sekali** oleh `scripts/04_prepare_base_model.py`
menghasilkan satu *checkpoint* dasar bersama yang menjadi titik awal
identik bagi **B1, B2, dan seluruh konfigurasi DP-SGD (E1/E2)**, agar
perbedaan performa antar blok murni mencerminkan efek federasi dan DP,
bukan perbedaan arsitektur normalisasi -- sekaligus menjadi prasyarat
arsitektural yang membuat gradien per-sampel DP-SGD (Subbab 2.4.3)
terdefinisi dengan bersih.

Satu persoalan implementasi tak terduga ditemukan dan ditangani selama
pengembangan, dicatat di sini karena memengaruhi validitas hasil B2
apabila tidak ditangani: API tingkat tinggi `YOLO(path).train(...)`
Ultralytics, ketika dimuat dari sebuah *checkpoint* `.pt`, selalu membangun
ulang arsitektur model dari berkas `.yaml`-nya dan hanya mentransplantasi
tensor yang cocok nama+bentuknya -- karena `parse_model()` Ultralytics
selalu membuat `nn.BatchNorm2d` baru untuk tiap blok `Conv`, perilaku ini
secara diam-diam **mengembalikan seluruh `GroupNorm` menjadi `BatchNorm`**
setiap kali dipanggil, ditemukan lewat kegagalan agregasi FedAvg pada ronde
kedua (`state_dict` klien tiba-tiba memiliki kunci
`running_mean`/`running_var` tambahan yang tidak dimiliki bobot global
sebelumnya). Ditangani dengan `src/fedxpalm/federated/trainer_utils.py`
(`build_trainer_from_checkpoint`): memuat model dari *checkpoint* secara
langsung sebagai objek `nn.Module` dan menetapkannya pada
`DetectionTrainer.model` **sebelum** memanggil `trainer.train()`, sehingga
arsitektur `GroupNorm` dijamin bertahan di sepanjang seluruh ronde
federasi, termasuk pada pelatihan DP-SGD (Subbab 3.10).

## 3.6 B1: Prosedur Pelatihan Tersentral

B1 adalah *baseline* tersentral (bukan batas atas matematis), dilatih atas
**seluruh** *split train* (8.937 citra) dalam satu *run* non-federasi,
memakai bobot dasar GroupNorm yang identik dengan titik awal B2 (Subbab
3.5.3), sehingga selisih performa B1-versus-B2 mencerminkan efek federasi,
bukan perbedaan data latih. Tabel 3.3 merangkum hiperparameternya.

Tabel 3.3. Hiperparameter B1

| Hiperparameter | Nilai |
|---|---|
| *Epoch* maksimum | 150 |
| *Early-stopping patience* | 30 |
| Ukuran *batch* | 16 |
| Ukuran citra | 960×960 |
| *Optimizer* | SGD |

*Checkpoint* yang dilaporkan dipilih berdasarkan performa *validation*, lalu
dievaluasi sekali pada *split held-out test* (Subbab 3.8).

## 3.7 B2: Prosedur Pelatihan Federasi (FedAvg) Tanpa DP

### 3.7.1 Hiperparameter Pelatihan Lokal dan Federasi

B2 menerapkan FedAvg [12], [13] tanpa *Differential Privacy* di bawah
konfigurasi K = 4, Dirichlet $\alpha = 0{,}5$, *partition seed* = 42 yang
sama dengan Subbab 3.4. Tabel 3.4 merangkum hiperparameternya.

Tabel 3.4. Hiperparameter B2

| Hiperparameter | Nilai |
|---|---|
| Ronde komunikasi | 40 |
| *Epoch* lokal per ronde | 2 |
| *Optimizer* | SGD |
| *Learning rate* awal | 0,01 |
| Momentum | 0,9 |
| *Weight decay* | 0,0005 |
| Ukuran *batch* | 8 |
| Ukuran citra | 960×960 |
| *Warmup* | Tidak ada |

SGD momentum dipilih dan dipakai **seragam di seluruh blok (B1, B2, E1,
E2)** -- bukan *optimizer* adaptif seperti AdamW -- agar selisih utilitas
antar blok murni mencerminkan efek federasi dan DP yang diteliti, bukan
tercampur efek pergantian *optimizer*; pilihan ini juga konsisten dengan
literatur DP-SGD yang karakterisasinya umumnya dibangun di atas SGD polos.

### 3.7.2 Algoritma FedAvg dan Replikasi Multi-*Seed*

`src/fedxpalm/federated/server.py` (`run_federated_training`)
mengorkestrasi $T=40$ ronde: pada tiap ronde, setiap klien menjalankan
*fine-tuning* lokal atas bobot global ronde tersebut
(`src/fedxpalm/federated/client.py`, `train_client_round`), lalu
`src/fedxpalm/federated/fedavg.py` (`fedavg`) merata-ratakan seluruh
`state_dict` klien secara berbobot sesuai Persamaan (2.3) -- tanpa
memfilter berdasarkan status *trainable*/beku (relevan bagi biaya
komunikasi E2, Subbab 3.13.2). Pelatihan B2 diulang pada **tiga *seed*
pelatihan** (42, 123, 2026) yang hanya memvariasikan realisasi stokastik
pelatihan dan tidak dipilih atas dasar properti numerik tertentu;
*partition seed* tetap 42 di ketiga *run*, sehingga ketiga replikasi
berjalan di atas partisi klien yang identik (Tabel 3.2) dan hanya berbeda
pada inisialisasi/urutan stokastik pelatihannya -- tujuannya menguji
stabilitas konvergensi FedAvg di bawah ketimpangan data antar klien yang
substansial.

## 3.8 Protokol Uji *Held-Out* dan Metrik Evaluasi

*Split held-out test* dicadangkan khusus untuk pelaporan akhir dan tidak
dipakai untuk memilih hiperparameter, jumlah ronde, atau *checkpoint*, pada
seluruh blok (B1, B2, E1, E2). B1 dievaluasi pada *split held-out test*
hanya setelah *checkpoint*-nya dikunci, dan hasil tersebut tidak dipakai
untuk memilih atau mengubah konfigurasi B2. Untuk B2, *checkpoint*
*validation*-terbaik pada *seed* 42 dikunci lewat manifes *immutability*
(*hash* SHA-256 *checkpoint* dan berkas konfigurasi terkait, diverifikasi
ulang tepat sebelum evaluasi) dan dievaluasi sekali pada *split held-out
test* yang sama dengan B1, menghasilkan titik pembanding tersandingkan
penuh yang juga dipakai sebagai titik rujukan bagi E1/E2 (Subbab 3.10,
4.5.4). *Checkpoint* E1 dan E2 mengikuti protokol kunci-lalu-uji yang sama.

Performa deteksi dihitung oleh `src/fedxpalm/eval/detection_metrics.py`
(membungkus `model.val()` Ultralytics) memakai metrik deteksi objek
standar: mAP@0.5, mAP@0.5:0.95, *Precision*, dan *Recall*, dihitung pada
*split* yang ditentukan pada tiap tabel hasil (Bab 4).

## 3.9 Verifikasi Implementasi dan Pemilihan Strategi *Clipping* DP-SGD

### 3.9.1 Audit Kontrak *Loss* Ultralytics-Opacus

Sebelum melatih E1/E2, kontrak numerik antara nilai `loss` yang
dikembalikan `v8DetectionLoss.loss()` Ultralytics dan asumsi Opacus
(`loss_reduction="mean"`, Subbab 2.4.5) diverifikasi terlebih dahulu.
Audit menemukan nilai `loss` Ultralytics berbentuk $\bar{L} \times n$ (rata-
rata per-sampel yang diskalakan ulang dengan ukuran *batch*), bukan rata-
rata murni -- ketidaksesuaian ini dikonfirmasi menginflasi `grad_sample`
Opacus secara diam-diam (tanpa galat) sebesar faktor ukuran *microbatch*
fisik. Pendekatan perbaikan pertama yang diuji (`loss_reduction="sum"`)
dievaluasi lewat *pilot* lima ronde dan ditolak karena menghasilkan
utilitas jauh di bawah baseline (dipertahankan sebagai catatan diagnostik,
bukan dihapus). Konfigurasi *canonical* yang divalidasi dan dipakai untuk
seluruh hasil E1/E2 mempertahankan `loss_reduction="mean"` namun secara
eksplisit membatalkan penskalaan ulang Ultralytics sebelum `backward()`
(`total_loss = loss.sum() / actual_microbatch_size`), diverifikasi lewat
turunan kalkulus pada model mainan dan pengukuran tiga-arah
(`legacy`/`direct-sum`/`canonical`) pada model nyata. Sebagai pemeriksaan
akhir sebelum pelatihan *final*, `scripts/36_dp_pipeline_sanity_audit.py`
menjalankan 17 pemeriksaan *sanity* pada beberapa ronde/klien nyata
(konsistensi *seed*, *per-sample gradient*, *clipping*, penambahan *noise*,
akumulasi *logical batch*, *privacy accounting*, pembatasan parameter
*trainable*, stabilitas numerik).

### 3.9.2 *Sweep* Ambang *Flat Clipping*

Ambang *clipping* norma-*flat* $C$ (Persamaan 2.5) disapu pada rentang
$C \in \{0{,}3, 0{,}5, 0{,}7, 1, 2, 5, 10, 15, 20, 30\}$ memakai konfigurasi
diagnostik P2 (σ=0,75, 5 ronde), untuk mengidentifikasi nilai $C$ yang
memberi utilitas terbaik sebelum konfigurasi dikunci untuk pelatihan
*final* 20 ronde.

### 3.9.3 Penurunan Ambang *Per-Layer* sebagai Pembanding

Sebagai kandidat pembanding *flat clipping*, ambang *clipping* per-tensor
diturunkan dari median norma gradien per-sampel tensor yang terukur (bukan
ditebak):

$$C_i = \frac{m_i}{\lVert m \rVert_2} \, C_{\text{total}}$$

dengan $m_i$ median norma gradien per-sampel tensor $i$, dan
$C_{\text{total}}=1{,}0$ disamakan dengan $C$ *flat* terpilih, sehingga
$\lVert C_{\text{vec}} \rVert_2 = 1{,}0$ -- menyamakan sensitivitas total
dan skala *noise* dengan *baseline flat* $C=1$, menjadikan perbandingan ini
murni perbandingan realokasi sinyal, bukan perbandingan anggaran privasi
yang berbeda.

### 3.9.4 Validasi Multi-*Seed* dan Pemilihan Konfigurasi *Final*

Kedua strategi *clipping* (*flat* C=1 vs *per-layer*) divalidasi pada tiga
*seed* pelatihan (42, 123, 2026), konfigurasi P2, σ=0,75, 5 ronde, sebelum
salah satu dipilih sebagai konfigurasi tetap untuk pelatihan *final* E1/E2
20-ronde pada Subbab 3.10 -- hasil perbandingan dilaporkan pada Subbab 4.4.

## 3.10 E1/E2: Prosedur Pelatihan DP-SGD (*Full* dan *Partial*)

### 3.10.1 Konfigurasi DP-SGD Bersama

Konfigurasi *canonical* yang divalidasi (Subbab 3.9) dipakai identik untuk
E1 maupun E2: *noise multiplier* σ=0,75, ambang *clipping* *flat* C=1,
*accountant* privasi **PRV** (Subbab 2.4.4), Opacus `PrivacyEngine`
(Subbab 2.4.5), 20 ronde komunikasi, 2 *epoch* lokal per ronde, *logical
batch size*=64, dan *partition seed*/klien identik dengan B2 (Tabel 3.2).
Kedua konfigurasi dilatih pada **satu *seed* pelatihan (42)** karena
keterbatasan waktu komputasi (Subbab 1.4).

### 3.10.2 E1: DP-SGD *Full*-Parameter

E1 menerapkan DP-SGD pada **seluruh parameter *trainable*** model (identik
dengan cakupan parameter B1/B2), sehingga seluruh gradien per-sampel
melalui mekanisme *clipping* dan penambahan *noise* Opacus.

### 3.10.3 E2: DP-SGD *Partial*-Parameter

E2 membekukan (*freeze*) sebagian besar *backbone* dan hanya melatih
parameter pada *stage* 16, 19, 22, dan 23 (929.522 dari 2.591.010 parameter
total *trainable*; 1.661.488 parameter dibekukan, 64,1% dari total) secara
privat -- parameter yang dibekukan dikecualikan dari mekanisme
`PrivacyEngine` dan tidak menerima *noise*. Desain ini menguji apakah
membatasi cakupan parameter yang dilatih secara privat memberikan
keunggulan komputasi dan/atau privasi dibanding melatih seluruh parameter
(E1) -- dianalisis pada Subbab 4.5 dan 4.8.1.

## 3.11 Analisis *Privacy-Utility* dan Akuntansi ε per Klien

Anggaran privasi $\varepsilon$ dilacak per klien per ronde memakai
*accountant* PRV (Subbab 2.4.4), dengan laju *sampling* $q$ tiap klien
diturunkan dari ukuran datanya (Tabel 3.2) dan *logical batch size*=64.
Karena laju *sampling* berbeda antar klien akibat ketimpangan ukuran data,
jumlah langkah optimisasi privat per ronde juga berbeda antar klien;
$\varepsilon$ *worst-case* (maksimum lintas klien) pada tiap ronde dilaporkan
sebagai ringkasan anggaran privasi keseluruhan (Subbab 4.6). Perbandingan
$\varepsilon$ per klien antara E1 dan E2 dipakai untuk menguji apakah
cakupan parameter *trainable* memengaruhi akuntansi privasi (Subbab 4.6.3).

## 3.12 Protokol XAI *Matched-Sample*: Grad-CAM++

### 3.12.1 Manifes Sampel Tersandingkan

Untuk membandingkan perhatian visual model B2, E1, dan E2 secara adil,
seluruh **3.821 kotak *ground-truth*** pada *split held-out test* dipilih
sebagai populasi sampel, murni berdasarkan label (tanpa melibatkan prediksi
model apapun). Identitas setiap sampel (jalur citra, *hash* SHA-256, indeks
kotak) dikunci dalam satu manifes yang dipakai identik untuk ketiga model
(`results/xai_matched/sample_manifest.json`), dan *hash checkpoint* ketiga
model diverifikasi ulang terhadap manifes *immutability* (Subbab 3.8)
sebelum evaluasi XAI dijalankan.

### 3.12.2 Perhitungan Grad-CAM++, AD, dan FRR

Grad-CAM++ (Subbab 2.6.1) dijalankan pada *target layer* 22, resolusi
960×960, `top_fraction`=0,2, seragam untuk ketiga model. *Average Drop*
(Persamaan 2.7) dan *Focus Retention Rate* (Subbab 2.6.2) dihitung per
sampel per model, disertai penanda `correct_detection` (IoU$\ge$0,5
terhadap kotak *ground-truth*) untuk membedakan CAM yang menyertai deteksi
berhasil dari CAM yang menggambarkan respons internal semata pada deteksi
yang gagal. Evaluasi ini bersifat **sepenuhnya *post-hoc***: dijalankan
setelah seluruh *checkpoint* B2/E1/E2 dikunci, dan tidak memengaruhi
pemilihan *checkpoint*/konfigurasi manapun (Subbab 1.4).

## 3.13 Analisis Biaya Komputasi dan Komunikasi

### 3.13.1 Biaya Komputasi

Waktu pelatihan tiap blok (B1, B2, E1, E2) diambil dari log masing-masing
*run* (bukan diukur ulang secara terpisah), untuk membandingkan *overhead*
komputasi relatif DP-SGD (E1, E2) terhadap federasi tanpa DP (B2), dan
*overhead* relatif E2 (*partial*) terhadap E1 (*full*).

### 3.13.2 Biaya Komunikasi

Volume komunikasi dihitung secara **teoretis** dari ukuran *state_dict*
model dikalikan jumlah ronde dan jumlah klien (simulasi satu-GPU ini tidak
benar-benar mengirim *byte* lewat jaringan fisik). Implementasi agregasi
`fedavg()` saat ini (Subbab 2.3.1) dikonfirmasi merata-ratakan **seluruh**
kunci `state_dict` tanpa memfilter berdasarkan status *trainable*/beku,
sehingga volume komunikasi aktual E2 dihitung identik dengan E1 meski E2
membekukan 64,1% parameter; sebagai pembanding, volume komunikasi
hipotetis dihitung juga untuk skenario di mana hanya parameter *trainable*
E2 yang dikomunikasikan, sebagai ilustrasi potensi optimasi yang belum
diimplementasikan (Subbab 4.8.3).

## 3.14 *Deployment*: *Containerization* dan VPS

### 3.14.1 Arsitektur Layanan Inferensi

Layanan inferensi (`deployment/app.py`) mengimplementasikan alur:
pengguna mengunggah citra melalui antarmuka web → *preprocessing* (*resize*
ke ukuran citra model) → inferensi YOLOv11 → keluaran kotak deteksi dan
kelas → visualisasi Grad-CAM++ opsional menggunakan mekanisme yang sama
dengan Subbab 3.12. *Checkpoint* yang dipakai untuk *deployment* adalah
**B2 *seed* 42** (*checkpoint* federasi terkunci yang sama dengan Subbab
4.2.2), bukan E1/E2, karena tujuan demonstrasi ini adalah kelayakan
operasional model federasi non-privat sebagai titik rujukan utilitas.

### 3.14.2 *Containerization* dengan Docker

*Image* aplikasi dibangun dari `deployment/Dockerfile` (basis
`python:3.11-slim`, CPU-*only*, tanpa dependensi GPU), dengan *checkpoint*
model dipasang sebagai *bind mount* ke dalam *container* (bukan di-*bake*
ke dalam *image*), sehingga *checkpoint* dapat diperbarui tanpa membangun
ulang *image*.

### 3.14.3 Prosedur Pengujian Fungsional

Pengujian fungsional dilakukan dengan mengunggah citra tandan sawit nyata
lewat antarmuka *dashboard*, memverifikasi bahwa: model berhasil dimuat
saat *container* dijalankan (tidak ada galat *startup*), *endpoint*
menerima berkas gambar dan mengembalikan hasil deteksi dengan kelas dan
skor keyakinan yang benar, dan visualisasi Grad-CAM++ tervisualisasi sesuai
ekspektasi. Pengujian ini menguji kelayakan operasional fungsional, bukan
pengukuran performa produksi (*latency*, *throughput*) yang berada di luar
cakupan penelitian ini (Subbab 1.4).

## 3.15 Reproduktibilitas dan Ketersediaan Kode

Seluruh kode -- pemrosesan data, pelatihan tersentral dan federasi,
DP-SGD, evaluasi, XAI, dan *deployment* -- tersedia pada repositori
`fedx-palm`, terorganisasi sebagai:

```
configs/            konfigurasi dataset dan FL (YAML)
src/fedxpalm/       paket Python inti (data, models, federated, privacy, xai, eval)
scripts/            skrip orkestrasi bernomor (01-52), mengikuti urutan tahapan Bab 1-4
deployment/         layanan inferensi Docker (app.py, Dockerfile, templates/)
docs/thesis/        bab-bab tesis ini, termasuk manuskrip JUTIF (docs/thesis/manuscript/)
```

Versi pustaka dikunci pada `requirements.txt`; *seed* acak untuk pembagian
data (42), partisi Dirichlet (42), tiga *seed* pelatihan B2 (42, 123, 2026),
dan *seed* pelatihan DP-SGD E1/E2 (42) didokumentasikan pada
`configs/*.yaml` dan subbab metodologi terkait, guna menjamin hasil dapat
direproduksi ulang oleh peneliti lain di lingkungan komputasi yang berbeda.
Implementasi kode, manifes pembagian dataset, dan berkas konfigurasi
tersedia dari penulis koresponden atas permintaan yang wajar.
