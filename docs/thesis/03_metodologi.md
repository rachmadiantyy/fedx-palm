<!--
CATATAN PENULISAN (hapus komentar ini sebelum submit):
Bab ini merombak total Bab 3 dan ditulis SUPAYA PERSIS MENGIKUTI kode
implementasi pada repo ini (src/fedxpalm/, scripts/, configs/*.yaml).
Setiap nilai hiperparameter dikutip langsung dari configs/*.yaml --
jika kamu mengubah file konfigurasi tersebut sebelum menjalankan
eksperimen, perbarui juga angka-angka di bab ini agar tetap konsisten.
Referensi path file (mis. `src/fedxpalm/...`) sengaja dipertahankan
sebagai jejak audit ke kode sesungguhnya.
-->

# CHAPTER 3 -- METODOLOGI PENELITIAN

## 3.1 Rancangan Penelitian

Penelitian ini dirancang sebagai eksperimen kuantitatif bertingkat, dimulai
dari pembagian dataset bebas-kebocoran dan partisi Non-IID Dirichlet,
dilanjutkan empat blok eksperimen (B1, B2, E1, E2), dan dievaluasi pada tiga
dimensi: utilitas deteksi, privasi formal, dan keterjelasan (*explainability*).
Seluruh kode implementasi diorganisasikan sebagai paket Python `fedxpalm`
(`src/fedxpalm/`), dikendalikan oleh berkas konfigurasi YAML
(`configs/dataset.yaml`, `configs/fl_config.yaml`, `configs/dp_config.yaml`),
dan dijalankan melalui sepuluh skrip bernomor (`scripts/01_*.py` hingga
`scripts/10_*.py`) yang mengikuti urutan tahapan pada Subbab 1.5.2.

## 3.2 Lingkungan dan Perangkat Implementasi

### 3.2.1 Spesifikasi Perangkat Keras

[VERIFIKASI: lengkapi tipe GPU (mis. NVIDIA RTX ????, kapasitas VRAM) dan
spesifikasi CPU/RAM workstation/server yang benar-benar dipakai untuk
menjalankan `notebooks/FedXPalm_v2_Colab.ipynb` atau `scripts/05`-`08`.]
Seluruh pelatihan (B1, B2, E1, E2) dijalankan pada satu GPU; simulasi
federasi K klien direalisasikan sebagai *loop* FedAvg sekuensial pada GPU
tunggal tersebut (Subbab 3.6.2), bukan sebagai proses terdistribusi
lintas-*host* fisik. *Deployment* akhir (Subbab 3.12) dijalankan pada
sebuah *Virtual Private Server* (VPS) CPU-*only* untuk menguji portabilitas
praktis tanpa ketergantungan GPU pada tahap produksi.

### 3.2.2 Konfigurasi Perangkat Lunak

Implementasi memakai Python 3.11 dengan versi pustaka yang dikunci
(*pinned*) pada `requirements.txt` demi reprodusibilitas:

| Pustaka | Versi | Peran |
|---|---|---|
| PyTorch | 2.5.1 | *Backend deep learning* |
| torchvision | 0.20.1 | Utilitas visi pendukung PyTorch |
| Ultralytics | 8.4.51 | Arsitektur dan *training loop* YOLOv11 |
| Opacus | 1.5.4 | DP-SGD per-sampel (`PrivacyEngine`) |
| Roboflow | 1.3.11 | Pengunduhan dataset |

Pelatihan federasi dijalankan sebagai simulasi *sequential* FedAvg pada satu
GPU (`src/fedxpalm/federated/server.py`). Model akhir di-*deploy* sebagai
layanan inferensi Flask + Ultralytics (CPU) di dalam *image* Docker
(`deployment/`), dijalankan pada satu VPS.

## 3.3 Dataset dan Strategi Pembagian Bebas-Kebocoran

### 3.3.1 Karakteristik Dataset

Dataset diperoleh dari platform Roboflow, proyek
*palm-fruit-ripeness-detection-f6sac-ccb2z* versi 2 pada *workspace*
*dydy-worker*, diunduh dalam format anotasi `yolov11` melalui
`scripts/01_download_dataset.py` (membungkus
`src/fedxpalm/data/download.py`). [VERIFIKASI setelah pengunduhan: jumlah
total citra, jumlah instans per kelas, dan resolusi citra asli -- isi tabel
karakteristik dataset di sini berdasarkan `data.yaml` hasil unduhan
sesungguhnya, bukan asumsi.] Dataset mencakup enam kelas kematangan TBS
sebagaimana dijabarkan pada Subbab 2.1.2.

### 3.3.2 Strategi Pemisahan Berbasis Identitas Tandan (`bunch_id`)

*Split* bawaan Roboflow (train/valid/test) bersifat per-*frame* dan acak,
sehingga berisiko menempatkan beberapa foto dari tandan fisik yang sama pada
*split* train dan test sekaligus -- model kemudian sebagian "menghafal"
tandan yang justru dipakai mengujinya, mengembang-gelembungkan metrik
evaluasi secara optimistis-palsu. `src/fedxpalm/data/split.py`
(`leakage_free_split`) menanggulangi ini dengan mengumpulkan ulang seluruh
citra lintas *split* bawaan Roboflow, mengelompokkannya berdasarkan
`bunch_id` yang diuraikan dari nama berkas citra (pola *regex*
`^(?P<bunch_id>.+?)(?:_\d+)?\.(jpg|jpeg|png)$`, dengan citra yang tidak
cocok polanya diperlakukan sebagai *bunch* tunggal beranggota satu citra),
lalu membagi ulang pada **level *bunch*** (bukan level citra) mengikuti
rasio 70% *train* : 15% *validation* : 15% *test* (*seed* = 42). Algoritma
pembagian bersifat *greedy*: setiap *bunch* (diacak urutannya terlebih
dahulu) dialokasikan ke *split* yang jumlah citranya paling jauh di bawah
target proporsinya saat itu, sehingga rasio akhir tetap mendekati target
meski ukuran tiap *bunch* (jumlah foto per tandan) bervariasi.

### 3.3.3 Audit Kebocoran Data

Setelah pembagian, `leakage_free_split` menjalankan audit otomatis:
memverifikasi bahwa tidak ada satupun `bunch_id` yang muncul pada lebih dari
satu *split* (`assert not leaked`). Audit ini dijalankan sebagai bagian
integral dari `scripts/02_prepare_splits.py`, bukan langkah manual terpisah,
sehingga kebocoran data akan menghentikan pipeline dengan galat eksplisit
alih-alih lolos secara diam-diam.

## 3.4 Partisi Data Non-IID Berbasis Distribusi Dirichlet

### 3.4.1 Formulasi Partisi Dirichlet

`src/fedxpalm/data/partition.py` (`dirichlet_partition`) mengimplementasikan
skema *latent Dirichlet allocation* (Subbab 2.3.3) atas *split* train.
Karena satu citra deteksi objek dapat memuat kotak dari beberapa kelas
sekaligus, setiap citra terlebih dahulu diberi **kelas primer** = kelas
yang paling sering muncul di antara kotak anotasi YOLO-nya
(`_primary_class`). Untuk tiap kelas $c \in \{0,\ldots,5\}$, proporsi
sampel yang jatuh ke $K$ klien ditarik dari $\mathrm{Dir}(\alpha,\ldots,
\alpha)$ dengan $\alpha = 0{,}5$ (*seed* = 42), lalu citra kelas tersebut
dibagi mengikuti proporsi itu. Citra tanpa anotasi (jika ada) dibagi
*round-robin* antar klien agar tidak ada klien yang sama sekali tidak
kebagian sampel. Hasil partisi disimpan sebagai `data/splits/
federated_partitions/k{K}.json` (peta `client_id -> daftar nama berkas`),
lalu dimaterialisasikan menjadi folder per-klien berisi *symlink* citra +
label dan `data.yaml` masing-masing oleh
`src/fedxpalm/federated/client_data.py` (`materialize_clients`) --
setiap klien memakai *split* *validation*/*test* global yang sama (bukan
lokal), karena evaluasi federasi dilakukan di sisi *server* atas data yang
tidak pernah "dimiliki" satu klien manapun.

### 3.4.2 Konfigurasi Jumlah Klien (K)

Jumlah klien di-*sweep* pada $K \in \{2, 4, 8, 12, 16\}$
(`configs/fl_config.yaml: clients.k_values`), merepresentasikan skenario
jumlah kebun/afdeling yang berpartisipasi dalam federasi -- dari kolaborasi
kecil (2 kebun) hingga jaringan yang lebih luas (16 kebun). Untuk tiap K,
partisi dan materialisasi klien dijalankan sekali oleh
`scripts/03_partition_clients.py` dan dipakai ulang di seluruh blok
eksperimen (B2, E1, E2) pada K tersebut, sehingga perbandingan antar blok
pada K yang sama selalu berjalan di atas pembagian data klien yang identik.

## 3.5 Arsitektur Detektor YOLOv11

### 3.5.1 Struktur *Backbone-Neck-Head*

Detektor yang dipakai adalah **YOLOv11n** (varian *nano*, ~2,6 juta
parameter, `yolo11n.pt` sebagai titik awal pra-latih COCO), dengan struktur
*backbone* (tahap indeks 0-10: `Conv`, `C3k2`, `SPPF`, `C2PSA`), *neck*
PAN-FPN (tahap 11-22), dan *head* `Detect` *anchor-free* *decoupled* (tahap
23), sebagaimana dijabarkan pada Subbab 2.2.1. Struktur tahap ini
diverifikasi identik pada seluruh varian ukuran YOLOv11 (n/s/m/l/x), sebuah
sifat yang dimanfaatkan agar pemilihan lapisan target Grad-CAM++ (Subbab
3.9.1) dan indeks tahap *backbone* yang dibekukan pada eksperimen E2
(Subbab 3.10.4) tetap sahih bila varian model diganti di kemudian hari.

Bobot pra-latih COCO ditransplantasikan ke kepala deteksi berkelas-6
(`scripts/04_prepare_base_model.py`): tensor bobot yang cocok nama dan
bentuknya antara `yolo11n.pt` (80 kelas COCO) dan arsitektur target (6
kelas TBS sawit) disalin langsung -- meliputi seluruh *backbone* dan *neck*
karena keduanya tidak bergantung pada jumlah kelas -- sedangkan lapisan
klasifikasi pada *head* yang berdimensi kelas diinisialisasi ulang dari
awal.

### 3.5.2 Fungsi Kerugian Komposit

Fungsi kerugian mengikuti Persamaan (2.2): CIoU untuk regresi kotak,
*Binary Cross-Entropy* untuk klasifikasi, dan *Distribution Focal Loss*
untuk representasi probabilistik tepi kotak, dengan pemasangan target
melalui *Task-Aligned Assigner* bawaan Ultralytics (Subbab 2.2.2). Bobot
komponen kerugian ($\lambda_{box}=7{,}5$; $\lambda_{cls}=0{,}5$;
$\lambda_{dfl}=1{,}5$) memakai nilai baku Ultralytics untuk YOLOv11, tidak
diubah pada penelitian ini.

### 3.5.3 Konversi *BatchNorm* ke *GroupNorm*

Sebagaimana dijabarkan pada Subbab 2.6, seluruh **81 lapisan
`BatchNorm2d`** pada YOLOv11n dikonversi menjadi `GroupNorm` melalui
`src/fedxpalm/models/groupnorm.py` (`convert_batchnorm_to_groupnorm`),
dengan jumlah grup tiap lapisan dipilih sebagai pembagi terbesar dari
jumlah kanal lapisan tersebut yang $\le 32$. Bobot dan bias afin
`GroupNorm` diinisialisasi dari `BatchNorm` yang digantikannya (statistik
`running_mean`/`running_var`, yang tidak dimiliki `GroupNorm`, tidak
ditransfer). Konversi ini dijalankan **sekali** oleh
`scripts/04_prepare_base_model.py` menghasilkan satu *checkpoint* dasar
bersama (`models/base_groupnorm.pt`) yang menjadi titik awal identik bagi
**seluruh** blok eksperimen B1, B2, E1, dan E2 -- bukan hanya blok berDP --
agar perbedaan performa antar blok murni mencerminkan perbedaan mekanisme
FL/DP yang diuji, bukan perbedaan arsitektur normalisasi.

Dua persoalan implementasi tak terduga ditemukan dan ditangani selama
pengembangan kerangka ini, dicatat di sini sebagai bagian metodologi karena
keduanya memengaruhi validitas hasil apabila tidak ditangani:

1. **Aktivasi SiLU *in-place* merusak *hook* Opacus.** Blok `Conv`
   Ultralytics menjalankan aktivasi SiLU dengan `inplace=True`, yang
   menimpa tensor aktivasi sebelum `GradSampleModule` Opacus sempat
   membaca nilainya pada *backward pass*, memicu galat *"Output ... is a
   view and is being modified inplace"*. Ditangani dengan menonaktifkan
   `inplace` pada seluruh modul aktivasi model
   (`disable_inplace_ops`) sebagai bagian dari persiapan model DP.
2. ***Auto-fusing* Ultralytics tidak kompatibel dengan `GroupNorm`.**
   `model.val()`/`.predict()` Ultralytics secara otomatis menyatukan
   (*fuse*) tiap `Conv`+`BatchNorm` menjadi satu operasi untuk mempercepat
   inferensi, dengan asumsi keras bahwa lapisan normnya adalah
   `BatchNorm2d` (mengakses atribut `running_var`). Ditangani dengan
   menambal (*monkeypatch*) `BaseModel.fuse()` agar melewati lapisan
   `GroupNorm` alih-alih menyatukannya, diterapkan otomatis pada saat
   modul `fedxpalm` diimpor (`src/fedxpalm/models/groupnorm.py:
   patch_fuse_for_groupnorm`).

## 3.6 Prosedur Pelatihan Federasi (FedAvg)

### 3.6.1 Hiperparameter Pelatihan Lokal

| Hiperparameter | Nilai | Sumber |
|---|---|---|
| *Epoch* lokal per ronde | 2 | `fl_config.yaml: local_training.epochs_per_round` |
| Ukuran *batch* | 16 | `local_training.batch_size` |
| Ukuran citra | 640×640 | `model.imgsz` |
| *Optimizer* | AdamW | `local_training.optimizer` |
| *Learning rate* awal | 0,001 | `local_training.lr0` |
| Momentum | 0,9 | `local_training.momentum` |
| *Weight decay* | 0,0005 | `local_training.weight_decay` |
| Jumlah ronde federasi | 40 | `federated.rounds` |

Pemilihan AdamW (bukan SGD momentum seperti pada iterasi pertama kerangka
ini) didasarkan pada rezim pelatihan federasi: dengan hanya 2 *epoch* lokal
per ronde sebelum agregasi, *learning rate* adaptif AdamW memberi
konvergensi yang lebih andal tanpa penjadwalan *learning rate* panjang yang
biasanya dibutuhkan SGD momentum.

### 3.6.2 Algoritma FedAvg

`src/fedxpalm/federated/server.py` (`run_federated_training`) mengorkestrasi
$T=40$ ronde: pada tiap ronde, setiap klien memanggil
`src/fedxpalm/federated/client.py` (`train_client_round`) untuk melakukan
*fine-tuning* lokal atas bobot global ronde tersebut, lalu
`src/fedxpalm/federated/fedavg.py` (`fedavg`) merata-ratakan seluruh
`state_dict` klien secara berbobot sesuai Persamaan (2.3). Bobot global
baru disimpan sebagai *checkpoint* `global_round_{t}.pt` dan dikirim ke
ronde berikutnya.

Implementasi `train_client_round` **tidak** memakai API tingkat tinggi
`YOLO(path).train(...)` Ultralytics secara langsung. Ditemukan bahwa API
tersebut, ketika dimuat dari sebuah *checkpoint* `.pt`, selalu membangun
ulang arsitektur model dari berkas `.yaml`-nya dan hanya mentransplantasi
tensor yang cocok nama+bentuknya (`Model.train()`'s
`self.trainer.model = self.trainer.get_model(weights=self.model,
cfg=self.model.yaml)`) -- karena `parse_model()` Ultralytics selalu
membuat `nn.BatchNorm2d` baru untuk tiap blok `Conv`, perilaku ini secara
diam-diam **mengembalikan seluruh `GroupNorm` menjadi `BatchNorm`** setiap
kali dipanggil, ditemukan lewat kegagalan agregasi FedAvg pada ronde kedua
(`state_dict` klien tiba-tiba memiliki kunci `running_mean`/`running_var`
tambahan yang tidak dimiliki bobot global sebelumnya). Sebagai gantinya,
`train_client_round` memakai `src/fedxpalm/federated/trainer_utils.py`
(`build_trainer_from_checkpoint`): memuat model dari *checkpoint* secara
langsung sebagai objek `nn.Module`, menetapkannya pada
`DetectionTrainer.model` **sebelum** memanggil `trainer.train()` --
`setup_model()` Ultralytics melewati pembangunan-ulang bila model yang
diberikan sudah berupa `nn.Module` -- sehingga arsitektur `GroupNorm`
dijamin bertahan di sepanjang seluruh 40 ronde federasi.

## 3.7 Mekanisme *Differential Privacy* Berbasis DP-SGD Per-Sampel

### 3.7.1 Integrasi DP-SGD melalui Opacus `PrivacyEngine`

`src/fedxpalm/privacy/dp_sgd.py` (`train_client_round_dp`) merealisasikan
Subbab 2.4 dan 2.7. Karena Opacus membutuhkan kendali langsung atas
`model`, `optimizer`, dan `DataLoader` -- kendali yang tidak diekspos API
tingkat tinggi `YOLO.train()` -- fungsi ini memakai pola serupa
`build_trainer_from_checkpoint` untuk memperoleh `DetectionTrainer` dengan
model, *optimizer* (AdamW, hiperparameter sama seperti Subbab 3.6.1), dan
*dataloader* lokal klien yang sudah disiapkan Ultralytics, lalu
membungkus ketiganya dengan `opacus.PrivacyEngine.make_private(...,
poisson_sampling=True)`. Presisi campuran (AMP) dinonaktifkan
(`amp=False`) karena Opacus tidak secara resmi mendukungnya.

### 3.7.2 Operasi DP-SGD Per-Langkah

Berbeda dari `train_client_round` (B2) yang memanggil `trainer.train()`
penuh (dengan penjadwalan *learning rate*, EMA, dsb. otomatis dari
Ultralytics), jalur DP menjalankan ***loop* langkah manual**: untuk tiap
*batch* dari `DPDataLoader` (ukuran *batch* bervariasi karena *Poisson
sampling*, Subbab 2.7), citra dipraproses lewat `trainer.preprocess_batch`,
diteruskan lewat model taat-DP untuk memperoleh kerugian
(`dp_model(batch)`, memakai jalur `BaseModel.loss()` Ultralytics yang
mengembalikan kerugian langsung tanpa perlu *forward* eksplisit terpisah),
lalu `loss.sum().backward()` dan `dp_optimizer.step()` -- langkah inilah
yang, di balik layar, menjalankan *clipping* per-sampel (2.5) dan mekanisme
Gaussian (2.6) secara otomatis pada `DPOptimizer`. *Batch* kosong (dapat
terjadi karena *Poisson sampling*) dilewati.

### 3.7.3 Akuntansi Anggaran Privasi (*PRV Accountant*)

Setelah seluruh langkah lokal satu ronde selesai, $\varepsilon$ kumulatif
klien tersebut dihitung memakai *accountant* PRV Opacus
(`privacy_engine.get_epsilon(delta=10^{-5})`), sesuai Subbab 2.4.4.
$\varepsilon$ dilaporkan per klien per ronde dalam `history.json` tiap
*run* federasi; nilai $\varepsilon$ akhir yang dilaporkan pada Bab 4 untuk
tiap konfigurasi $(K,\sigma)$ adalah **nilai maksimum lintas klien** pada
ronde terakhir, karena jaminan privasi keseluruhan sistem dibatasi oleh
klien dengan kebocoran privasi terbesar. `src/fedxpalm/privacy/
accounting.py` menyediakan utilitas mandiri untuk memproyeksikan tabel
$\sigma \to \varepsilon$ tanpa perlu menjalankan pelatihan penuh terlebih
dahulu, berguna untuk perencanaan sebelum menjalankan *sweep* GPU yang
mahal.

## 3.8 Mekanisme Pembanding: DP-FedAvg Tingkat-Klien

Sebagai pembanding pendahuluan (bukan eksperimen utama, lihat Subbab
2.4.5), *noise* Gaussian disuntikkan pada level pembaruan bobot
(*delta*) tiap klien setelah pelatihan lokal (tanpa DP) selesai, sebelum
dikirim ke *server* -- berbeda dari DP-SGD per-sampel yang menyuntikkan
*noise* pada tiap langkah gradien selama optimisasi. Hasil eksperimen
pendahuluan ini (Bab 4) memotivasi pemilihan DP-SGD per-sampel sebagai
mekanisme privasi utama penelitian, dengan dua *confound* (perbedaan
BatchNorm/GroupNorm dan cara pemetaan $\varepsilon$ antara kedua skema)
diakui secara eksplisit sebagai keterbatasan perbandingan tersebut.

## 3.9 Evaluasi Kualitas Penjelasan Berbasis Grad-CAM++

### 3.9.1 Pemilihan Lapisan Target (*Target Layer*)

Sesuai Subbab 2.8.1, Grad-CAM++ (`src/fedxpalm/xai/gradcam.py`,
`YOLOGradCAMPlusPlus`) dipasang pada tahap indeks **ke-22** arsitektur
YOLOv11 (blok C3k2 terakhir pada *neck*, sebelum *head* `Detect`), melalui
*forward hook* dan *full backward hook* PyTorch pada modul tersebut.

### 3.9.2 Metrik *Average Drop* (AD) dan *Focus Retention Rate* (FRR)

Kedua metrik mengikuti definisi operasional Persamaan (2.9) dan (2.10),
diimplementasikan pada `src/fedxpalm/xai/metrics.py`. Oklusi AD dilakukan
pada **20% piksel teratas** berdasarkan nilai aktivasi Grad-CAM++
(`top_fraction=0.2`), diganti dengan rata-rata kanal citra yang
bersangkutan sebagai *fill* netral.

### 3.9.3 Prosedur Evaluasi Penjelasan

`src/fedxpalm/xai/evaluate.py` (`evaluate_faithfulness`) menjalankan
evaluasi atas *split* *test* global: untuk setiap kotak *ground-truth*
pada setiap citra, *anchor* dengan skor tertinggi model untuk kelas kotak
tersebut dipilih, Grad-CAM++ dan kedua metrik faithfulness dihitung untuk
*anchor* itu, lalu dirata-ratakan per kelas dan secara global. Prosedur ini
dijalankan pada *checkpoint* yang ditetapkan sebagai "model operasional"
(lihat Subbab 4.1) melalui `scripts/09_evaluate_xai.py`, yang juga dapat
menyimpan visualisasi tumpang-tindih (*overlay*) peta panas untuk
inspeksi kualitatif (Gambar 4.5/4.7 pada Bab 4).

## 3.10 Skenario Eksperimen

Empat blok eksperimen dijalankan berurutan mengikuti `scripts/05` hingga
`scripts/08`; ringkasannya:

### 3.10.1 B1 -- *Baseline* Tersentral (*Centralized*)

`scripts/05_train_b1_centralized.py`. *Fine-tuning* atas **seluruh** *split*
*train* terkumpul tanpa federasi maupun DP, 150 *epoch* dengan
*early-stopping* (*patience* 30). Berfungsi sebagai batas atas (*upper
bound*) utilitas yang dibandingkan dengan seluruh blok federasi/privat.

### 3.10.2 B2 -- *Baseline* Federasi tanpa DP

`scripts/06_train_b2_federated.py`. FedAvg murni (Subbab 3.6), di-*sweep*
atas seluruh $K \in \{2,4,8,12,16\}$, 40 ronde per konfigurasi.

### 3.10.3 E1 -- DP-SGD Penuh

`scripts/07_train_e1_dp_full.py`. DP-SGD (Subbab 3.7) diterapkan pada
**seluruh** parameter yang dapat dilatih, di-*sweep* atas *grid* penuh
$K \times \sigma$, dengan $\sigma \in \{0{,}5;\ 1{,}0;\ 1{,}5;\ 2{,}0;\
3{,}0\}$ dan $C=1{,}0$ tetap (`configs/dp_config.yaml`).

### 3.10.4 E2 -- DP-SGD Parsial (*Backbone* Beku)

`scripts/08_train_e2_dp_partial.py`. Identik dengan E1, kecuali seluruh
tahap *backbone* (indeks 0-10, lihat Subbab 3.5.1) dibekukan
(`freeze=[0,...,10]`, memakai mekanisme *freeze* bawaan `DetectionTrainer`
Ultralytics) -- hanya *neck* dan *head* yang menerima pembaruan DP-SGD.
Menguji apakah pembekuan *backbone* pra-latih menguntungkan performa pada
domain sawit, sebagaimana disarankan literatur klasifikasi citra untuk
domain yang jauh dari data pra-latih.

Tabel 3.1 merangkum keempat blok.

| Kode | Federasi | DP | Parameter di-DP | Jumlah *run* |
|---|---|---|---|---|
| B1 | Tidak | Tidak | - | 1 |
| B2 | Ya | Tidak | - | 5 (per K) |
| E1 | Ya | Ya (*full*) | Seluruh | 25 (5 K × 5 σ) |
| E2 | Ya | Ya (*partial*) | *Neck*+*head* saja | 25 (5 K × 5 σ) |

## 3.11 Metrik Evaluasi dan Ambang Operasional

### 3.11.1 Metrik Utilitas Deteksi

Dihitung oleh `src/fedxpalm/eval/detection_metrics.py`
(membungkus `model.val()` Ultralytics) atas *split* *test* global:
mAP@0.5, mAP@0.5:0.95, *Precision*, *Recall*, dan F1-*Score*, secara
global maupun per kelas.

### 3.11.2 Metrik Privasi

*Privacy budget* $\varepsilon$ (Subbab 3.7.3) pada $\delta=10^{-5}$ tetap.

### 3.11.3 Ambang Operasional

Mengikuti H1 (Subbab 1.6), model federasi dikategorikan **layak**
(*acceptable*) bila mAP@0.5 $\ge 0{,}70$ dan selisihnya terhadap B1
(*FL-cost*) kecil. Ambang ini dipakai konsisten di seluruh Bab 4 untuk
menandai konfigurasi mana yang dianggap "model operasional" layak
di-*deploy* (Subbab 3.12).

### 3.11.4 Catatan Metrik Per-Kelas

Karena partisi Dirichlet (Subbab 3.4) menghasilkan distribusi kelas yang
timpang antar klien, metrik per-kelas (terutama *recall* pada kelas
minoritas seperti *Empty Bunch*/*Abnormal*) dilaporkan terpisah dari metrik
makro pada Bab 4, karena rata-rata makro dapat menyembunyikan degradasi
performa yang terkonsentrasi pada kelas jarang.

## 3.12 *Deployment* Layanan Inferensi Berbasis Docker

Model operasional (Subbab 3.11.3) dikemas menjadi layanan inferensi
berbasis Flask (`deployment/app.py`) di dalam *image* Docker
(`deployment/Dockerfile`, CPU-*only*, wheel PyTorch CPU untuk memperkecil
ukuran *image*). Layanan menerima unggahan citra TBS, menjalankan deteksi
kelas dengan keyakinan tertinggi, menghitung Grad-CAM++ untuk deteksi
tersebut (memakai mesin `YOLOGradCAMPlusPlus` yang sama dengan Subbab
3.9.1, bukan implementasi terpisah), dan mengembalikan citra hasil dengan
kotak deteksi serta tumpang-tindih peta panas dalam satu halaman web
(`deployment/templates/index.html`), mendemonstrasikan bahwa transparansi
XAI tidak hanya alat analisis Bab 4, tetapi juga bagian dari antarmuka
produksi.

## 3.13 Reproduktibilitas dan Ketersediaan Kode

Seluruh kode -- pemrosesan data, pelatihan federasi, DP-SGD, evaluasi XAI,
dan *deployment* -- tersedia pada repositori `fedx-palm`
(*branch* `claude/palm-oil-yolov11-federated-m4o613`), terorganisasi
sebagai:

```
configs/            konfigurasi dataset, FL, dan DP (YAML)
src/fedxpalm/        paket Python inti (data, models, federated, privacy, xai, eval)
scripts/             sepuluh skrip orkestrasi bernomor (01-10)
deployment/          Dockerfile, layanan Flask, template HTML
notebooks/           notebook Colab/Jupyter end-to-end
docs/thesis/          bab-bab tesis ini
```

Versi pustaka dikunci pada `requirements.txt`; *seed* acak untuk pembagian
data (42), partisi Dirichlet (42), dan inisialisasi Ultralytics
didokumentasikan pada `configs/*.yaml` masing-masing tahap, guna menjamin
hasil dapat direproduksi ulang oleh peneliti lain di lingkungan komputasi
yang berbeda.
