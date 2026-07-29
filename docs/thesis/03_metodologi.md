<!--
CATATAN PENULISAN (hapus komentar ini sebelum submit):
Bab ini dipersempit selaras dengan Bab 1-2: hanya B1 (tersentral) dan B2
(federasi FedAvg tanpa DP, K=4) yang dibahas sebagai metodologi inti.
Subbab DP-SGD (Opacus, akuntansi privasi ε), DP-FedAvg tingkat-klien,
evaluasi Grad-CAM++/AD/FRR, dan deployment Docker DIHAPUS dari bab ini
dan dipindah menjadi catatan cakupan yang ditunda (Subbab 3.9). Seluruh
angka pada bab ini (jumlah citra, distribusi klien, hiperparameter)
dikutip langsung dari eksperimen nyata yang sudah dijalankan penulis di
GPU (lihat docs/thesis/manuscript/JUTIF_Manuscript_FedXPalm_B1B2.docx),
BUKAN dari config template -- cross-check terhadap configs/*.yaml sebelum
submit bila konfigurasi berubah.
-->

# CHAPTER 3 -- METODOLOGI PENELITIAN

## 3.1 Rancangan Penelitian

Penelitian tahap ini dirancang sebagai eksperimen kuantitatif dua-blok,
dimulai dari pembagian dataset bebas-kebocoran dan partisi Non-IID
Dirichlet, dilanjutkan dua blok eksperimen -- **B1** (*baseline* tersentral)
dan **B2** (*baseline* federasi FedAvg tanpa DP) -- dan dievaluasi pada satu
dimensi: utilitas deteksi. Seluruh kode implementasi diorganisasikan sebagai
paket Python `fedxpalm` (`src/fedxpalm/`), dikendalikan oleh berkas
konfigurasi YAML (`configs/dataset.yaml`, `configs/fl_config.yaml`), dan
dijalankan melalui skrip bernomor (`scripts/01_*.py` hingga
`scripts/06_*.py`) yang mengikuti urutan tahapan pada Subbab 1.5.2. Blok
eksperimen DP-SGD (E1/E2) yang juga tersedia pada repositori kode
(`scripts/07_*.py`, `scripts/08_*.py`) tidak dijalankan/dilaporkan pada
tahap ini -- lihat Subbab 3.9.

## 3.2 Lingkungan dan Perangkat Implementasi

### 3.2.1 Spesifikasi Perangkat Keras

Seluruh pelatihan (B1, B2) dijalankan pada satu *workstation* dengan
akselerator GPU **NVIDIA GeForce RTX 4080 (16 GB VRAM)**, CPU multi-*core*,
dan RAM sistem 32 GB. Simulasi federasi K = 4 klien direalisasikan sebagai
*loop* FedAvg sekuensial pada GPU tunggal tersebut (Subbab 3.6.2), bukan
sebagai proses terdistribusi lintas-*host* fisik.

### 3.2.2 Konfigurasi Perangkat Lunak

Implementasi memakai Python 3.10 dengan versi pustaka yang dikunci
(*pinned*) pada `requirements.txt` demi reprodusibilitas:

| Pustaka | Versi | Peran |
|---|---|---|
| PyTorch | 2.5.1 (CUDA 12.1) | *Backend deep learning* |
| Ultralytics | 8.4.51 | Arsitektur dan *training loop* YOLOv11 |
| Roboflow | 1.3.11 | Pengunduhan dataset |

Pelatihan federasi dijalankan sebagai simulasi *sequential* FedAvg pada satu
GPU (`src/fedxpalm/federated/server.py`). Dependensi Opacus tetap terkunci
pada `requirements.txt` sebagai persiapan tahap DP-SGD lanjutan (Subbab
3.9), namun tidak dipakai pada eksekusi B1/B2 di bab ini.

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
secara optimistis-palsu. Hasil pembagian dirangkum pada Tabel 3.1.

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
dijalankan/dilaporkan pada tahap penelitian ini -- lihat Subbab 3.9 dan
5.3.

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
dikecualikan dari optimisasi.

### 3.5.3 Konversi *BatchNorm* ke *GroupNorm*

Sebagaimana dijabarkan pada Subbab 2.5, seluruh **81 lapisan
`BatchNorm2d`** pada YOLOv11n dikonversi menjadi `GroupNorm` melalui
`src/fedxpalm/models/groupnorm.py` (`convert_batchnorm_to_groupnorm`),
dengan jumlah grup tiap lapisan dipilih sebagai pembagi terbesar dari
jumlah kanal lapisan tersebut yang $\le 32$. Audit arsitektur akhir
mengonfirmasi **0 modul BatchNorm tersisa** dan **81 modul GroupNorm**.
Konversi ini dijalankan **sekali** oleh `scripts/04_prepare_base_model.py`
menghasilkan satu *checkpoint* dasar bersama yang menjadi titik awal
identik bagi **B1 maupun B2**, agar perbedaan performa antar blok murni
mencerminkan efek federasi, bukan perbedaan arsitektur normalisasi.

Satu persoalan implementasi tak terduga ditemukan dan ditangani selama
pengembangan, dicatat di sini karena memengaruhi validitas hasil B2 apabila
tidak ditangani: API tingkat tinggi `YOLO(path).train(...)` Ultralytics,
ketika dimuat dari sebuah *checkpoint* `.pt`, selalu membangun ulang
arsitektur model dari berkas `.yaml`-nya dan hanya mentransplantasi tensor
yang cocok nama+bentuknya -- karena `parse_model()` Ultralytics selalu
membuat `nn.BatchNorm2d` baru untuk tiap blok `Conv`, perilaku ini secara
diam-diam **mengembalikan seluruh `GroupNorm` menjadi `BatchNorm`** setiap
kali dipanggil, ditemukan lewat kegagalan agregasi FedAvg pada ronde kedua
(`state_dict` klien tiba-tiba memiliki kunci `running_mean`/`running_var`
tambahan yang tidak dimiliki bobot global sebelumnya). Ditangani dengan
`src/fedxpalm/federated/trainer_utils.py`
(`build_trainer_from_checkpoint`): memuat model dari *checkpoint* secara
langsung sebagai objek `nn.Module` dan menetapkannya pada
`DetectionTrainer.model` **sebelum** memanggil `trainer.train()`, sehingga
arsitektur `GroupNorm` dijamin bertahan di sepanjang seluruh ronde
federasi.

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

SGD momentum dipilih dan dipakai **seragam di B1 maupun B2** (bukan
*optimizer* adaptif seperti AdamW), agar selisih utilitas federasi
($\Delta_{FL}$, B1 vs B2) murni mencerminkan efek federasi yang diteliti,
bukan tercampur efek pergantian *optimizer* antarblok; pilihan ini juga
konsisten dengan literatur DP-SGD yang menjadi rujukan tahap penelitian
lanjutan (Subbab 3.9), yang karakterisasinya dibangun di atas SGD polos.

### 3.7.2 Algoritma FedAvg dan Replikasi Multi-*Seed*

`src/fedxpalm/federated/server.py` (`run_federated_training`)
mengorkestrasi $T=40$ ronde: pada tiap ronde, setiap klien menjalankan
*fine-tuning* lokal atas bobot global ronde tersebut
(`src/fedxpalm/federated/client.py`, `train_client_round`), lalu
`src/fedxpalm/federated/fedavg.py` (`fedavg`) merata-ratakan seluruh
`state_dict` klien secara berbobot sesuai Persamaan (2.3). Pelatihan B2
diulang pada **tiga *seed* pelatihan** (42, 123, 2026) yang hanya
memvariasikan realisasi stokastik pelatihan dan tidak dipilih atas dasar
properti numerik tertentu; *partition seed* tetap 42 di ketiga *run*,
sehingga ketiga replikasi berjalan di atas partisi klien yang identik
(Tabel 3.2) dan hanya berbeda pada inisialisasi/urutan stokastik
pelatihannya -- tujuannya menguji stabilitas konvergensi FedAvg di bawah
ketimpangan data antar klien yang substansial.

## 3.8 Protokol Uji *Held-Out* dan Metrik Evaluasi

*Split held-out test* dicadangkan khusus untuk pelaporan akhir dan tidak
dipakai untuk memilih hiperparameter, jumlah ronde, atau *checkpoint*. B1
dievaluasi pada *split held-out test* hanya setelah *checkpoint*-nya
dikunci, dan hasil tersebut tidak dipakai untuk memilih atau mengubah
konfigurasi B2. Untuk B2, ketiga *checkpoint seed* yang terkunci ditujukan
untuk evaluasi *held-out test* yang sama pada pelaporan berikutnya; pada
tahap penulisan ini, hasil B2 dilaporkan pada *split validation*, dan
metrik *validation* serta *held-out test* tidak dicampur dalam
perbandingan langsung pada Bab 4 (lihat Subbab 4.3 dan 4.6).

Performa deteksi dihitung oleh `src/fedxpalm/eval/detection_metrics.py`
(membungkus `model.val()` Ultralytics) memakai metrik deteksi objek
standar: mAP@0.5, mAP@0.5:0.95, *Precision*, dan *Recall*, dihitung pada
*split* yang ditentukan pada tiap tabel hasil (Bab 4) -- *split held-out
test* untuk B1, *split validation* untuk B2.

## 3.9 Cakupan yang Ditunda: DP-SGD, XAI, dan *Deployment*

Bab ini secara sengaja tidak membahas metodologi tiga komponen berikut,
yang tersedia sebagai infrastruktur kode pada repositori namun berada di
luar cakupan rumusan masalah tahap penelitian ini (Subbab 1.4):

1. **DP-SGD per-sampel** (`src/fedxpalm/privacy/`, `scripts/07_*.py`,
   `scripts/08_*.py`) -- proteksi privasi formal berbasis Opacus
   (`PrivacyEngine`), direncanakan sebagai perluasan langsung di atas
   *checkpoint* GroupNorm B1/B2 yang sudah kompatibel secara arsitektural
   (Subbab 2.5).
2. **Explainable AI berbasis Grad-CAM++** (`src/fedxpalm/xai/`,
   `scripts/09_*.py`) -- validasi interpretasi visual model.
3. **Deployment layanan inferensi berbasis Docker** (`deployment/`) --
   demonstrasi operasional model.

Ketiganya direncanakan sebagai materi Bab 3-5 pada laporan penelitian
lanjutan yang dibangun di atas titik rujukan B1/B2 yang ditetapkan pada bab
ini (lihat Subbab 5.3).

## 3.10 Reproduktibilitas dan Ketersediaan Kode

Seluruh kode -- pemrosesan data, pelatihan tersentral dan federasi, serta
evaluasi -- tersedia pada repositori `fedx-palm` (*branch*
`claude/tesis-b1-b2-c77afk`), terorganisasi sebagai:

```
configs/            konfigurasi dataset dan FL (YAML)
src/fedxpalm/         paket Python inti (data, models, federated, eval; privacy/xai disiapkan untuk tahap lanjutan)
scripts/             skrip orkestrasi bernomor (01-06 dipakai pada tahap ini; 07-10 untuk tahap lanjutan)
docs/thesis/          bab-bab tesis ini, termasuk manuskrip JUTIF (docs/thesis/manuscript/)
```

Versi pustaka dikunci pada `requirements.txt`; *seed* acak untuk pembagian
data (42), partisi Dirichlet (42), dan tiga *seed* pelatihan B2 (42, 123,
2026) didokumentasikan pada `configs/*.yaml` dan Subbab 3.7.2, guna
menjamin hasil dapat direproduksi ulang oleh peneliti lain di lingkungan
komputasi yang berbeda.
