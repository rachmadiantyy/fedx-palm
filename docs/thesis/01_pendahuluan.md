<!--
CATATAN PENULISAN (hapus komentar ini sebelum submit):
Bab ini dipersempit cakupannya secara sengaja pada tahap penulisan ini:
HANYA B1 (baseline tersentral) dan B2 (baseline federasi tanpa DP) yang
menjadi materi inti Bab 1-5. DP-SGD (E1/E2), Explainable AI (Grad-CAM++),
dan deployment Docker -- yang pada draf sebelumnya menjadi tulang punggung
kerangka FedX-Palm -- kini disebut hanya sebagai arah pengembangan lanjutan
yang direncanakan, BUKAN bagian dari rumusan masalah/tujuan/hipotesis/
batasan penelitian tahap ini. Ini selaras dengan manuskrip JUTIF yang
sudah ditulis berdasarkan hasil eksperimen nyata (lihat
`docs/thesis/manuscript/JUTIF_Manuscript_FedXPalm_B1B2.docx`), yang secara
eksplisit membingkai B1/B2 sebagai "non-private utility reference point"
dan menyatakan hasil DP-SGD "outside the scope of the present manuscript".
Angka pada bab ini (jumlah citra, distribusi klien, dst.) dikutip langsung
dari manuskrip tersebut, hasil eksperimen nyata di GPU penulis (NVIDIA
GeForce RTX 4080, 16 GB VRAM). Lihat docs/thesis/NOTES_FOR_RACHMA.md untuk
riwayat keputusan penyempitan cakupan ini.
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
mentah -- yang dipertukarkan dengan *server* pengagregasi. Meski demikian,
literatur keamanan mencatat bahwa pembaruan model dan gradien tetap berpotensi
membocorkan informasi tentang data latihnya, dan model yang sudah terlatih
dapat membocorkan informasi keanggotaan (*membership*) sampel latih tertentu.
*Differential Privacy* (DP), khususnya DP-SGD, menyediakan jaminan formal dan
terukur yang membatasi pengaruh satu sampel data terhadap model yang
dirilis -- namun dengan konsekuensi *noise* yang disuntikkan dan penurunan
utilitas model. Memahami *trade-off* privasi-utilitas ini penting sebelum
DP-SGD dapat diadopsi secara bertanggung jawab pada *pipeline* visi komputer
pertanian terapan; oleh karena itu, penelitian ini memposisikan diri sebagai
**tahap awal yang non-privat** dari sebuah program riset yang lebih besar,
dan secara eksplisit membatasi cakupannya (Subbab 1.4) pada perbandingan
pelatihan tersentral versus federasi Non-IID sebagai titik rujukan utilitas
sebelum lapisan DP-SGD ditambahkan pada tahap lanjutan.

Penelitian ini menetapkan titik rujukan (*baseline*) tersebut dengan
membandingkan dua model: **B1**, model YOLOv11n yang dilatih secara
tersentral di atas seluruh data latih, dan **B2**, model YOLOv11n yang
dilatih melalui *Federated Averaging* (FedAvg) di atas partisi klien
Non-IID (distribusi Dirichlet, α = 0,5, K = 4 klien tersimulasi). Seluruh
lapisan *Batch Normalization* pada kedua model dikonversi menjadi *Group
Normalization*, untuk menghilangkan ketergantungan pada statistik lintas
sampel yang menjadi tidak stabil pada *batch* kecil dan Non-IID per klien --
substitusi ini sekaligus menjadikan arsitektur kompatibel secara struktural
dengan rencana perluasan ke DP-SGD per-sampel pada tahap penelitian
berikutnya, meski DP-SGD itu sendiri berada di luar cakupan bab ini.
Evaluasi dilakukan pada dataset TBS sawit publik (Roboflow, 10.814 citra
enam kelas kematangan), dengan pembagian *train*/*validation*/*held-out
test* yang bebas-kebocoran pada level kelompok sumber citra (lihat Subbab
3.3), agar selisih performa antara B1 dan B2 murni mencerminkan efek
federasi, bukan pencemaran data antar-*split*.

## 1.2 Rumusan Masalah

Praktik penilaian kematangan TBS yang berjalan saat ini bersifat subjektif
dan tidak konsisten antar pemanen, sehingga otomatisasi berbasis *deep
learning* menjadi kebutuhan. Otomatisasi itu sendiri menuntut data citra
dalam jumlah besar, sementara pengumpulan terpusat berisiko menyingkap
informasi operasional kebun -- inilah yang melatari kebutuhan pendekatan
*Federated Learning*. Namun, mengadopsi FL tanpa mengetahui berapa besar
utilitas yang harus "dibayar" dibandingkan pelatihan tersentral membuat
keputusan tersebut sulit dipertanggungjawabkan secara kuantitatif.
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

Rumusan masalah terkait proteksi privasi formal (DP-SGD) dan keterjelasan
model (*Explainable AI*) yang menjadi bagian kerangka FedX-Palm secara lebih
luas **tidak** dijawab pada tahap penelitian ini; keduanya diposisikan
sebagai arah pengembangan lanjutan (Subbab 1.4 dan 5.3), dibangun di atas
titik rujukan utilitas B1/B2 yang dilaporkan di sini.

## 1.3 Tujuan Penelitian

Sejalan dengan rumusan masalah di atas, penelitian ini diarahkan untuk:

1. Membangun dan mengevaluasi arsitektur *Federated Learning* berbasis
   FedAvg yang terintegrasi dengan model YOLOv11n untuk klasifikasi enam
   tingkat kematangan TBS kelapa sawit, dievaluasi pada dataset publik dari
   Roboflow dengan pembagian data bebas-kebocoran.

2. Mengukur dan membandingkan utilitas deteksi model tersentral (B1) dengan
   model federasi Non-IID (B2, direplikasi pada tiga *seed* pelatihan
   independen) menggunakan metrik deteksi standar (mAP@0.5, mAP@0.5:0.95,
   *Precision*, *Recall*), sebagai titik rujukan non-privat sebelum
   perluasan DP-SGD pada tahap penelitian berikutnya.

## 1.4 Batasan Masalah

Agar fokus dan hasilnya tetap terukur, penelitian pada tahap ini dibatasi
pada cakupan berikut:

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
   dilaporkan pada tahap penelitian ini; hanya K = 4 yang dibahas.

4. Perlindungan privasi formal melalui DP-SGD per-sampel, teknik
   *Explainable AI* (Grad-CAM++), dan *deployment* layanan inferensi
   berbasis Docker **berada di luar cakupan** rumusan masalah, tujuan, dan
   hipotesis penelitian tahap ini, meski infrastrukturnya (Opacus,
   `src/fedxpalm/xai/`, `deployment/`) sudah tersedia pada repositori kode
   sebagai persiapan tahap lanjutan (Subbab 5.3). Konversi BatchNorm ->
   GroupNorm pada Butir 2 tetap diterapkan karena juga bermanfaat langsung
   bagi stabilitas B2 (Subbab 2.5), terlepas dari rencana DP-SGD.

5. B2 direplikasi pada tiga *seed* pelatihan (42, 123, 2026) yang hanya
   memvariasikan realisasi stokastik pelatihan; *partition seed* untuk
   pembagian klien tetap 42 di ketiga *run*. B2 pada tahap ini dilaporkan
   pada *split validation*; evaluasi B2 pada *split held-out test* yang
   sama dengan B1 ditunda ke pelaporan berikutnya (Subbab 4.5, 5.3).

## 1.5 Metode Penelitian

Penelitian ini menggunakan paradigma eksperimen kuantitatif yang
dikombinasikan dengan pendekatan *Research and Development* (R&D).
Eksperimen difokuskan pada pengukuran selisih utilitas deteksi antara
pelatihan tersentral (B1) dan pelatihan federasi Non-IID tanpa DP (B2) pada
YOLOv11n.

### 1.5.1 Lingkungan dan Alat Implementasi

1. **Platform Komputasi.** Seluruh pelatihan dijalankan pada satu
   *workstation* dengan GPU NVIDIA GeForce RTX 4080 (16 GB VRAM), CPU
   multi-*core*, dan RAM 32 GB. Bobot hasil pelatihan disimpan dalam format
   `.pt` sebagai *checkpoint* untuk tahap evaluasi.

2. **Stack Perangkat Lunak.** Implementasi memakai Python 3.10, PyTorch
   2.5.1 (CUDA 12.1), dan Ultralytics 8.4.51 untuk YOLOv11. Pelatihan
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

Realisasi penelitian ini dijalankan melalui enam tahap berurutan.

1. **Studi Literatur.** Difokuskan pada dua bidang utama: deteksi objek
   berbasis YOLO untuk kematangan TBS sawit, dan *Federated Learning* di
   bawah data Non-IID, guna memetakan celah riset seputar titik rujukan
   utilitas tersentral-versus-federasi yang diaudit kebocorannya secara
   eksplisit.

2. **Perancangan Sistem.** Topologi FL dirancang dalam pola *server-client*
   terpusat dengan satu *server aggregator* dan K = 4 *client node*.
   YOLOv11n dipilih sebagai model dasar karena menyatukan tugas lokalisasi
   dan klasifikasi pada satu *inference pipeline* dan ringan untuk
   kebutuhan komputasi terbatas.

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

5. **Pelatihan Federasi (B2, FedAvg).** Pada setiap ronde, tiap klien
   menjalankan *fine-tuning* lokal atas bobot global ronde tersebut;
   *server* merata-ratakan bobot klien secara berbobot ukuran data lokal
   (FedAvg). Diulang pada tiga *seed* pelatihan independen untuk menguji
   stabilitas konvergensi di bawah ketimpangan data antar klien.

6. **Evaluasi dan Perbandingan.** Kedua model dievaluasi dengan metrik
   deteksi standar (mAP@0.5, mAP@0.5:0.95, *Precision*, *Recall*), lalu
   selisih utilitas B1-versus-B2 dianalisis sebagai titik rujukan
   non-privat untuk perluasan DP-SGD pada tahap penelitian selanjutnya.

## 1.6 Hipotesis

Mengacu pada kerangka teoretis dan rancangan metodologi yang telah
diuraikan, penelitian ini menetapkan satu hipotesis kerja utama.

**H1 -- Kelayakan Deteksi YOLOv11n pada Skenario Federasi Non-IID.**
Pelatihan kolaboratif YOLOv11n dengan algoritma *Federated Averaging*
(FedAvg) tanpa DP di atas *client node* berdistribusi Non-IID (K = 4,
Dirichlet α = 0,5) diperkirakan mampu menghasilkan model global dengan
*mean Average Precision* (mAP@0.5) yang mendekati performa pelatihan
tersentral (B1), dengan selisih utilitas (*FL-cost*) yang kecil dan
konvergensi yang stabil di seluruh *seed* pelatihan yang diuji, meski
ketimpangan proporsi data antar klien substansial.

Hipotesis terkait bentuk *trade-off* privasi-utilitas DP-SGD (pengaruh σ
dan K terhadap ε dan utilitas) serta validitas interpretasi visual XAI --
yang menjadi bagian kerangka FedX-Palm secara lebih luas -- akan dirumuskan
dan diuji pada tahap penelitian lanjutan, setelah B1/B2 di sini menetapkan
titik rujukan non-privatnya.
