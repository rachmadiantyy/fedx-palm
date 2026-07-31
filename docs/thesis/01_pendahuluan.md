<!--
CATATAN PENULISAN (hapus komentar ini sebelum submit):
Bab ini mencakup SELURUH kerangka FedX-Palm: B1 (baseline tersentral),
B2 (baseline federasi tanpa DP), E1/E2 (DP-SGD full/partial), XAI
(Grad-CAM++), dan deployment Docker -- seluruhnya sudah dikerjakan dan
dilaporkan dengan hasil nyata pada Bab 4 (lihat
docs/thesis/04_hasil_dan_pembahasan.md untuk seluruh angka rujukan pada
bab ini). Draf sebelumnya sempat mempersempit cakupan bab ini hanya ke
B1/B2 (lihat riwayat pada docs/thesis/NOTES_FOR_RACHMA.md); penyempitan
itu sudah TIDAK berlaku lagi sejak E1/E2/XAI/deployment selesai
dikerjakan dan dilaporkan penuh di Bab 4 -- rumusan masalah/tujuan/
batasan/hipotesis di bawah ini diperluas agar konsisten dengan cakupan
Bab 4 yang sebenarnya.
-->

# CHAPTER 1 -- PENDAHULUAN

## 1.1 Latar Belakang

Kelapa sawit (*Elaeis guineensis* Jacq.) menjadi komoditas dengan produktivitas
minyak per hektar tertinggi di antara tanaman penghasil minyak nabati, dan
menempatkan Indonesia bersama Malaysia sebagai dua produsen terbesar minyak
sawit dunia. Mutu *Crude Palm Oil* (CPO) yang dihasilkan sangat ditentukan
oleh ketepatan waktu panen, yang bergantung pada tingkat kematangan Tandan
Buah Segar (TBS) saat dipotong. Tandan yang dipanen terlalu dini memiliki
kandungan minyak rendah, sedangkan yang terlambat dipanen mengalami
peningkatan kadar Asam Lemak Bebas sehingga menurunkan mutu CPO. Pada
praktiknya, penilaian kematangan ini masih banyak bergantung pada inspeksi
visual pemanen. Cara tersebut subjektif: hasilnya berbeda antar pemanen,
terpengaruh kelelahan, dan menurun akurasinya pada pencahayaan lapangan yang
kurang baik. Persoalan konsistensi inilah yang membuat otomatisasi berbasis
visi komputer relevan untuk diterapkan pada rantai pascapanen sawit.

Model *object detection* satu-tahap seperti keluarga YOLO telah banyak
dipakai untuk tugas ini karena menggabungkan kecepatan inferensi *real-time*
dengan akurasi deteksi yang kompetitif. Pada praktiknya, citra TBS sering
tersebar di berbagai sumber -- lokasi panen, kelompok pengambilan data, atau
mitra pengumpul data yang berbeda -- alih-alih berada dalam satu repositori
terpusat. Mengumpulkan seluruh citra tersebut ke satu server pusat untuk
pelatihan model menimbulkan persoalan berbagi data, logistik, dan privasi,
yang melatarbelakangi penggunaan *Federated Learning* (FL) sebagai
alternatif. Pada skema FL, tiap *client node* melatih model secara lokal di
atas datanya sendiri, dan hanya pembaruan parameter model -- bukan citra
mentah -- yang dipertukarkan dengan *server* pengagregasi.

Meski demikian, FL saja tidak serta-merta menjadi jaminan privasi yang
lengkap. Literatur keamanan mencatat bahwa pembaruan model dan gradien tetap
berpotensi membocorkan informasi tentang data latihnya (*gradient inversion
attack*), dan model yang sudah terlatih dapat membocorkan informasi
keanggotaan (*membership*) sampel latih tertentu. *Differential Privacy*
(DP), khususnya DP-SGD, menyediakan jaminan formal dan terukur yang
membatasi pengaruh satu sampel data terhadap model yang dirilis -- namun
dengan konsekuensi *noise* yang disuntikkan dan penurunan utilitas model
yang perlu dikuantifikasi secara empiris sebelum DP-SGD dapat diadopsi
secara bertanggung jawab pada *pipeline* visi komputer pertanian terapan.
Di sisi lain, penurunan utilitas akibat DP-SGD juga menimbulkan pertanyaan
tentang keterjelasan model (*explainability*): apakah model yang sudah
"dirusak" oleh *noise* privasi masih memberi perhatian visual yang masuk
akal terhadap objek yang dideteksinya, atau visualisasi tersebut sekadar
artefak dari model yang kepercayaan dirinya sudah rapuh? Pertanyaan ini
dijawab melalui analisis *Explainable AI* (XAI) berbasis Grad-CAM++ yang
dibandingkan secara tersandingkan (*matched-sample*) antar model dengan dan
tanpa DP. Akhirnya, sebagai penutup rangkaian penelitian, kelayakan
operasional dari model yang dihasilkan didemonstrasikan melalui *deployment*
layanan inferensi berbasis Docker pada sebuah *Virtual Private Server*
(VPS), untuk menguji portabilitas praktis di luar lingkungan pelatihan GPU.

Penelitian ini karenanya dirangkai sebagai empat blok eksperimen berurutan:
**B1** (baseline tersentral), **B2** (baseline federasi Non-IID tanpa DP),
**E1/E2** (perluasan DP-SGD -- *full* dan *partial* parameter -- di atas
titik rujukan B2), dan analisis **XAI** (Grad-CAM++ tersandingkan pada B2,
E1, E2), yang seluruhnya ditutup dengan **demonstrasi *deployment***.
B1 adalah model YOLOv11n yang dilatih secara tersentral di atas seluruh
data latih; B2 adalah model YOLOv11n yang dilatih melalui *Federated
Averaging* (FedAvg) di atas partisi klien Non-IID (distribusi Dirichlet,
α = 0,5, K = 4 klien tersimulasi); E1 menerapkan DP-SGD pada seluruh
parameter *trainable* B2, sedangkan E2 menerapkan DP-SGD hanya pada
sebagian parameter (*backbone* dibekukan). Seluruh lapisan *Batch
Normalization* pada keempat model dikonversi menjadi *Group Normalization*,
karena DP-SGD per-sampel mensyaratkan gradien tiap sampel terdefinisi
secara independen -- sebuah sifat yang tidak dimiliki *Batch Normalization*
-- sekaligus menghilangkan ketergantungan pada statistik lintas sampel yang
menjadi tidak stabil pada *batch* kecil dan Non-IID per klien. Evaluasi
dilakukan pada dataset TBS sawit publik (Roboflow, 10.814 citra enam kelas
kematangan), dengan pembagian *train*/*validation*/*held-out test* yang
bebas-kebocoran pada level kelompok sumber citra (lihat Subbab 3.3), agar
selisih performa antar blok murni mencerminkan efek federasi dan DP,
bukan pencemaran data antar-*split*.

## 1.2 Rumusan Masalah

Praktik penilaian kematangan TBS yang berjalan saat ini bersifat subjektif
dan tidak konsisten antar pemanen, sehingga otomatisasi berbasis *deep
learning* menjadi kebutuhan. Otomatisasi itu sendiri menuntut data citra
dalam jumlah besar, sementara pengumpulan terpusat berisiko menyingkap
informasi operasional kebun -- inilah yang melatari kebutuhan pendekatan
*Federated Learning*. Namun, FL saja belum menjadi jaminan privasi formal,
dan mengadopsi DP-SGD sebagai lapisan proteksi tambahan tanpa mengetahui
berapa besar utilitas yang harus "dibayar" membuat keputusan adopsinya sulit
dipertanggungjawabkan secara kuantitatif. Lebih jauh, jika DP-SGD memang
diadopsi, praktisi juga perlu tahu apakah model yang dihasilkan masih dapat
dipercaya secara kualitatif (lewat interpretasi visual) dan apakah model
tersebut benar-benar dapat dioperasikan di luar lingkungan pelatihan.
Berdasarkan latar tersebut, rumusan masalah penelitian ini adalah:

1. Mengingat pendekatan pelatihan terpusat berisiko membocorkan data
   operasional perkebunan, bagaimana arsitektur *Federated Learning* (FL)
   berbasis FedAvg dapat dirancang agar model deteksi objek YOLOv11n dapat
   dilatih secara kolaboratif tanpa harus memindahkan citra mentah dari
   masing-masing sumber data?

2. Sejauh mana pelatihan federasi Non-IID (B2) mampu mendekati utilitas
   deteksi model yang dilatih secara tersentral (B1) pada tugas deteksi
   kematangan TBS enam kelas, di bawah partisi klien Dirichlet dengan
   ketimpangan data substansial antar klien?

3. Seberapa besar penurunan utilitas deteksi yang ditimbulkan oleh
   penambahan *Differential Privacy* (DP-SGD) di atas titik rujukan
   federasi B2, dan apakah pembatasan cakupan parameter yang dilatih secara
   privat (*full* vs *partial* DP) memberikan keunggulan privasi atau
   utilitas yang jelas?

4. Bagaimana pengaruh strategi *gradient clipping* (norma *flat* versus
   *per-layer*) terhadap stabilitas dan besaran utilitas model DP-SGD, dan
   bagaimana anggaran privasi (ε) terdistribusi antar klien yang ukuran
   datanya timpang?

5. Apakah visualisasi Grad-CAM++ model dengan DP-SGD (E1/E2) menunjukkan
   perhatian visual yang secara kualitatif berbeda dari model tanpa DP
   (B2) ketika dibandingkan secara tersandingkan pada sampel yang identik,
   dan apakah model operasional yang dihasilkan dapat didemonstrasikan
   berjalan pada infrastruktur *deployment* di luar lingkungan pelatihan?

## 1.3 Tujuan Penelitian

Sejalan dengan rumusan masalah di atas, penelitian ini diarahkan untuk:

1. Membangun dan mengevaluasi arsitektur *Federated Learning* berbasis
   FedAvg yang terintegrasi dengan model YOLOv11n untuk klasifikasi enam
   tingkat kematangan TBS kelapa sawit, dievaluasi pada dataset publik dari
   Roboflow dengan pembagian data bebas-kebocoran.

2. Mengukur dan membandingkan utilitas deteksi model tersentral (B1) dengan
   model federasi Non-IID (B2, direplikasi pada tiga *seed* pelatihan
   independen) menggunakan metrik deteksi standar (mAP@0.5, mAP@0.5:0.95,
   *Precision*, *Recall*), termasuk perbandingan tersandingkan penuh pada
   *split held-out test* yang sama untuk *checkpoint* validasi-terbaik B2.

3. Menerapkan dan mengevaluasi DP-SGD di atas titik rujukan B2, pada dua
   konfigurasi cakupan parameter (E1 *full* dan E2 *partial*), serta
   mengukur besaran penurunan utilitas dan anggaran privasi (ε) yang
   dihasilkan, termasuk distribusinya antar klien.

4. Menganalisis pengaruh strategi *gradient clipping* (*flat* versus
   *per-layer*) terhadap stabilitas dan besaran utilitas model DP-SGD
   melalui *sweep* nilai ambang dan validasi multi-*seed*.

5. Mengevaluasi interpretasi visual (Grad-CAM++, *Average Drop*, *Focus
   Retention Rate*) model B2, E1, dan E2 secara tersandingkan pada sampel
   identik, serta mendemonstrasikan kelayakan operasional model melalui
   *deployment* layanan inferensi berbasis Docker pada VPS.

## 1.4 Batasan Masalah

Agar fokus dan hasilnya tetap terukur, penelitian ini dibatasi pada cakupan
berikut:

1. Subjek deteksi adalah citra Tandan Buah Segar (TBS) kelapa sawit yang
   dikategorikan ke dalam enam kelas kematangan: *Abnormal*, *Empty Bunch*,
   *Overripe*, *Ripe*, *Underripe*, dan *Unripe* (urutan alfabetis sesuai
   `configs/dataset.yaml`).

2. Arsitektur deteksi yang dievaluasi adalah YOLOv11n (varian *nano*,
   2.591.010 parameter), dengan seluruh 81 lapisan *Batch Normalization*
   dikonversi menjadi *Group Normalization* (0 BatchNorm tersisa setelah
   audit arsitektur). Varian YOLOv11 lain (s/m/l/x) berada di luar cakupan.

3. Lingkungan FL dirancang sebagai satu *server aggregator* dan **K = 4**
   *client node* tersimulasi secara sekuensial pada satu GPU (NVIDIA
   GeForce RTX 4080, 16 GB VRAM), dengan pembagian data Non-IID berbasis
   distribusi Dirichlet (α = 0,5, *partition seed* = 42). *Sweep* atas
   jumlah klien K lain (mis. {2, 8, 12, 16}) tersedia pada infrastruktur
   kode (`configs/fl_config.yaml: clients.k_values`) namun **tidak**
   dilaporkan pada penelitian ini; hanya K = 4 yang dibahas.

4. B2 direplikasi pada tiga *seed* pelatihan (42, 123, 2026) untuk menguji
   stabilitas konvergensi; *partition seed* untuk pembagian klien tetap 42
   di ketiga *run*. Konfigurasi DP-SGD *final* (E1/E2, Subbab 4.5) hanya
   dijalankan pada **satu *seed* (42)** karena keterbatasan waktu komputasi
   -- replikasi multi-*seed* untuk hasil DP-SGD *final* diserahkan sebagai
   arah pengembangan lanjutan (Subbab 5.3).

5. DP-SGD diterapkan pada level *sampel* (Opacus `PrivacyEngine`), bukan
   *secure aggregation* atau DP tingkat-klien; `secure_mode` Opacus (RNG
   kriptografis) tidak diaktifkan pada seluruh eksperimen tahap ini, dan ε
   maksimum yang dicapai (22,11) tergolong relatif longgar sebagai jaminan
   privasi formal -- hasil DP-SGD di sini karenanya bersifat karakterisasi
   *trade-off* privasi-utilitas, bukan klaim kesiapan jaminan privasi
   tingkat produksi.

6. Analisis *Explainable AI* (Grad-CAM++, *Average Drop*, *Focus Retention
   Rate*) bersifat *post-hoc* semata: dijalankan setelah seluruh
   *checkpoint* B2/E1/E2 dikunci, dan sama sekali tidak memengaruhi
   pemilihan *checkpoint* atau konfigurasi manapun.

7. Demonstrasi *deployment* (Docker pada VPS) bertujuan menguji kelayakan
   operasional fungsional (model berhasil dimuat dan menghasilkan deteksi
   yang benar), bukan pengukuran performa produksi (*latency*,
   *throughput*, ketahanan terhadap beban) yang berada di luar cakupan
   penelitian ini.

## 1.5 Metode Penelitian

Penelitian ini menggunakan paradigma eksperimen kuantitatif yang
dikombinasikan dengan pendekatan *Research and Development* (R&D).
Eksperimen difokuskan pada pengukuran selisih utilitas deteksi antara
pelatihan tersentral (B1), pelatihan federasi Non-IID tanpa DP (B2), dan
pelatihan federasi dengan DP-SGD (E1/E2) pada YOLOv11n, dilengkapi analisis
interpretasi visual dan demonstrasi operasional.

### 1.5.1 Lingkungan dan Alat Implementasi

1. **Platform Komputasi.** Seluruh pelatihan dijalankan pada satu
   *workstation* dengan GPU NVIDIA GeForce RTX 4080 (16 GB VRAM), CPU
   multi-*core*, dan RAM 32 GB. Bobot hasil pelatihan disimpan dalam format
   `.pt` sebagai *checkpoint* untuk tahap evaluasi. Demonstrasi *deployment*
   dijalankan pada VPS terpisah (2 vCPU, 4 GB RAM, Ubuntu 24.04, wilayah
   Jakarta).

2. **Stack Perangkat Lunak.** Implementasi memakai Python 3.10/3.11,
   PyTorch 2.5.1 (CUDA 12.1 untuk pelatihan, CPU-*only* untuk *deployment*),
   Ultralytics 8.4.51 untuk YOLOv11, dan Opacus untuk DP-SGD. Pelatihan
   federasi dijalankan sebagai simulasi *sequential* FedAvg pada satu GPU
   -- klien dijalankan bergiliran dalam tiap ronde komunikasi, bukan
   secara konkuren pada mesin fisik terpisah.

3. **Dataset.** Citra TBS sawit enam kelas kematangan diperoleh dari
   platform Roboflow (10.814 citra), dibagi bebas-kebocoran pada level
   kelompok sumber citra menjadi *train* (8.937 citra, 72 kelompok
   sumber) / *validation* (826 citra, 8 kelompok sumber) / *held-out test*
   (1.051 citra, 11 kelompok sumber), lalu *split train* dipartisi ke
   K = 4 klien mengikuti distribusi Dirichlet (α = 0,5).

### 1.5.2 Tahapan Penelitian

Realisasi penelitian ini dijalankan melalui delapan tahap berurutan.

1. **Studi Literatur.** Difokuskan pada empat bidang: deteksi objek
   berbasis YOLO untuk kematangan TBS sawit, *Federated Learning* di bawah
   data Non-IID, *Differential Privacy* (DP-SGD) pada konteks *deep
   learning*, dan *Explainable AI* berbasis metode *Class Activation
   Mapping*, guna memetakan celah riset seputar karakterisasi *trade-off*
   privasi-utilitas yang tersandingkan penuh dan diaudit kebocoran datanya
   secara eksplisit.

2. **Perancangan Sistem.** Topologi FL dirancang dalam pola *server-client*
   terpusat dengan satu *server aggregator* dan K = 4 *client node*.
   YOLOv11n dipilih sebagai model dasar karena menyatukan tugas lokalisasi
   dan klasifikasi pada satu *inference pipeline* dan ringan untuk
   kebutuhan komputasi terbatas; seluruh BatchNorm dikonversi ke GroupNorm
   agar kompatibel dengan gradien per-sampel yang dibutuhkan DP-SGD.

3. **Penyiapan dan Distribusi Dataset.** Citra TBS sawit diunduh dari
   Roboflow, dipecah ulang menjadi *train*/*validation*/*held-out test*
   berbasis identitas kelompok sumber untuk mencegah kebocoran data
   (*leakage*) antar-*split*, lalu *split train*-nya dipecah ke 4 klien
   melalui *sampling* Dirichlet, sehingga proporsi kelas kematangan tidak
   seragam antar klien.

4. **Pelatihan Tersentral (B1).** YOLOv11n dilatih di atas seluruh *split
   train* dalam satu *run* non-federasi, menghasilkan titik rujukan
   tersentral yang dievaluasi sekali pada *split held-out test* setelah
   *checkpoint* dipilih berdasarkan performa validasi.

5. **Pelatihan Federasi Tanpa DP (B2, FedAvg).** Pada setiap ronde, tiap
   klien menjalankan *fine-tuning* lokal atas bobot global ronde tersebut;
   *server* merata-ratakan bobot klien secara berbobot ukuran data lokal
   (FedAvg). Diulang pada tiga *seed* pelatihan independen untuk menguji
   stabilitas konvergensi di bawah ketimpangan data antar klien.

6. **Pemilihan Strategi *Clipping* dan Pelatihan DP-SGD (E1/E2).** Ambang
   *gradient clipping* disapu dan divalidasi tiga *seed* (*flat* vs
   *per-layer*), lalu konfigurasi terpilih dipakai untuk melatih E1 (DP-SGD
   pada seluruh parameter *trainable*) dan E2 (DP-SGD pada sebagian
   parameter, *backbone* dibekukan) di atas titik rujukan B2.

7. **Analisis Interpretasi Visual (XAI).** Grad-CAM++ dijalankan pada
   sampel yang identik untuk B2, E1, dan E2, diukur dengan *Average Drop*
   dan *Focus Retention Rate*, untuk membandingkan perhatian visual model
   dengan dan tanpa DP-SGD.

8. **Evaluasi, Perbandingan, dan Demonstrasi *Deployment*.** Seluruh model
   dievaluasi dengan metrik deteksi standar (mAP@0.5, mAP@0.5:0.95,
   *Precision*, *Recall*), dianalisis biaya komputasi/komunikasinya, lalu
   model operasional dikemas sebagai layanan inferensi Docker dan
   didemonstrasikan berjalan pada VPS.

## 1.6 Hipotesis

Mengacu pada kerangka teoretis dan rancangan metodologi yang telah
diuraikan, penelitian ini menetapkan tiga hipotesis kerja.

**H1 -- Kelayakan Deteksi YOLOv11n pada Skenario Federasi Non-IID.**
Pelatihan kolaboratif YOLOv11n dengan algoritma *Federated Averaging*
(FedAvg) tanpa DP di atas *client node* berdistribusi Non-IID (K = 4,
Dirichlet α = 0,5) diperkirakan mampu menghasilkan model global dengan
*mean Average Precision* (mAP@0.5) yang mendekati performa pelatihan
tersentral (B1), dengan konvergensi yang relatif stabil di seluruh *seed*
pelatihan yang diuji, meski ketimpangan proporsi data antar klien
substansial.

**H2 -- Penurunan Utilitas Akibat DP-SGD.** Penambahan DP-SGD di atas
titik rujukan B2 diperkirakan menimbulkan penurunan utilitas deteksi yang
substansial dibanding B2 tanpa DP, dengan besaran penurunan yang bergantung
pada anggaran privasi (ε) yang tercapai; pembatasan cakupan parameter yang
dilatih secara privat (*partial* DP, E2) diperkirakan tidak serta-merta
memberikan anggaran privasi yang lebih murah, karena ε ditentukan oleh laju
*sampling*, *noise multiplier*, dan jumlah langkah optimisasi -- bukan oleh
jumlah parameter *trainable*.

**H3 -- Konsistensi Interpretasi Visual Grad-CAM++.** Model dengan DP-SGD
(E1/E2) diperkirakan menunjukkan pola perhatian visual (Grad-CAM++) yang
secara kualitatif berbeda dari model tanpa DP (B2) ketika dibandingkan pada
sampel identik, namun perbedaan metrik kuantitatif (*Average Drop*, *Focus
Retention Rate*) semata tidak dapat langsung ditafsirkan sebagai perbedaan
kualitas penjelasan, mengingat sensitivitas metrik tersebut terhadap tingkat
keyakinan dasar model yang berbeda antar konfigurasi.
