<!--
CATATAN PENULISAN (hapus komentar ini sebelum submit):
Bab ini adalah rombak total Bab 1 mengikuti kerangka FedX-Palm, tetapi
dijalankan ulang di atas dataset Roboflow baru ("palm-fruit-ripeness-
detection-f6sac-ccb2z" v2, workspace "dydy-worker") dan konfigurasi
eksperimen yang diperbarui (YOLOv11n, optimizer AdamW, K in
{2,4,8,12,16}, 40 ronde federasi, sigma in {0.5,1.0,1.5,2.0,3.0}).
Bagian yang ditandai [VERIFIKASI: ...] harus dicek terhadap data.yaml
hasil unduhan Roboflow yang sesungguhnya dan spesifikasi perangkat
keras aktual sebelum bab ini dianggap final -- lihat
docs/thesis/NOTES_FOR_RACHMA.md.
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

Pada sisi teknologi, model *object detection* mutakhir seperti YOLOv11
memungkinkan pengenalan enam tingkat kematangan TBS -- *Unripe*, *Underripe*,
*Ripe*, *Overripe*, *Empty Bunch*, dan *Abnormal* [VERIFIKASI: cocokkan nama
dan urutan enam kelas ini dengan `names` pada `data.yaml` hasil unduhan
proyek Roboflow *palm-fruit-ripeness-detection-f6sac-ccb2z* versi 2,
workspace *dydy-worker*] -- dilakukan dalam satu tahap inferensi yang
menggabungkan lokalisasi dan klasifikasi. Kemampuan tersebut menjanjikan,
namun bergantung pada ketersediaan data latih yang beragam. Di sinilah
persoalan privasi muncul. Citra perkebunan bukan sekadar gambar buah; di
dalamnya dapat tersingkap lokasi blok, pola budidaya, hingga volume produksi
yang oleh perusahaan kelapa sawit dipandang sebagai rahasia dagang.
Akibatnya, mengumpulkan citra dari banyak kebun ke satu server pusat menjadi
pilihan yang sulit diterima oleh pemilik data.

*Horizontal Federated Learning* (HFL) menawarkan jalan keluar yang lebih
alami untuk situasi seperti ini. Pada skema HFL, tiap *client node* (yang
merepresentasikan kebun berbeda) berbagi ruang fitur citra yang sama tetapi
memegang sampel data yang sepenuhnya terisolasi. Pelatihan berjalan lokal di
setiap node, dan yang dipertukarkan ke *server aggregator* hanya pembaruan
parameter model, bukan citra mentahnya. Penelitian ini juga merealisasikan
*deployment* inferensi berbasis Docker: model akhir dikemas ke dalam *image*
Docker dan dijalankan sebagai layanan inferensi nyata pada sebuah
[VERIFIKASI: GPU NVIDIA RTX -- isi tipe/VRAM perangkat pelatihan yang
sebenarnya dipakai] untuk pelatihan, dan CPU-only untuk *deployment*, guna
menjaga reprodusibilitas dan portabilitas antar lingkungan komputasi. Yang
berada di luar cakupan adalah pelatihan *Federated Learning* terdistribusi
lintas-host fisik, yang dijalankan sebagai simulasi ekuivalen pada satu GPU.

Memisahkan data mentah saja ternyata tidak cukup. Riset keamanan beberapa
tahun terakhir memperlihatkan bahwa parameter model yang dipertukarkan dapat
dimanfaatkan untuk serangan inferensi tingkat lanjut. Dua di antara yang
paling sering disebut adalah *Membership Inference Attack* dan *Model
Inversion Attack*, yang berpotensi memulihkan kembali citra-citra sensitif
hanya dari bobot yang disadap. Jaminan formal yang bersifat matematis
diperlukan di sini, dan *Differential Privacy* (DP) merupakan kerangka
teoretis yang paling matang untuk tujuan tersebut. Penelitian ini mewujudkan
DP melalui mekanisme *Differentially Private Stochastic Gradient Descent*
(DP-SGD) per-sampel, yaitu pembatasan norma gradien per-sampel (*gradient
clipping*) yang dipadukan dengan penambahan *noise* Gaussian terkalibrasi.
Pemilihan jalur per-sampel ini konsisten dengan temuan generasi pertama
kerangka FedX-Palm: penyuntikan *noise* pada level pembaruan bobot
teragregasi (DP-FedAvg tingkat-klien) cenderung menyebabkan *collapse* model
deteksi objek yang lebih mendadak, sedangkan *noise* pada level gradien
per-sampel berinteraksi lebih halus dengan dinamika SGD. Dua varian DP-SGD
dibandingkan pada penelitian ini: *full* DP-SGD (seluruh parameter) versus
*partial* DP-SGD (hanya kepala deteksi, *backbone* dibekukan), guna menguji
apakah pembekuan *backbone* pra-latih -- sebagaimana disarankan pada
literatur klasifikasi citra untuk domain yang jauh dari data pra-latihnya --
juga menguntungkan pada deteksi objek domain sawit. Tentu saja, pemberian
*noise* membawa konsekuensi pada konvergensi dan akurasi, fenomena yang
dikenal sebagai *privacy-utility trade-off*.

Kendala lain dalam adopsi model *Deep Learning* di sektor pertanian adalah
karakter *black-box*-nya: pengguna sulit memahami alasan model menghasilkan
keputusan tertentu. Dalam merespons keterbatasan tersebut, penelitian
menambahkan lapisan transparansi melalui pendekatan *Explainable AI* (XAI),
khususnya Grad-CAM++, pada model YOLOv11 hasil federasi. Tujuannya bukan
sekadar menampilkan peta panas, melainkan menunjukkan secara empiris bahwa
keputusan model bertumpu pada ciri morfologis buah seperti gradasi warna,
kepadatan brondolan, dan tekstur permukaan, bukan pada latar atau distorsi
akibat injeksi *noise* DP.

Dibandingkan dengan iterasi pertama kerangka ini, penelitian ini
menjalankan ulang keseluruhan pipeline pada **dataset publik baru** yang
diperoleh dari Roboflow (proyek *palm-fruit-ripeness-detection-f6sac-ccb2z*,
versi 2, workspace *dydy-worker*), dan memperbarui sejumlah pilihan
rancangan berdasarkan tinjauan pustaka atas penelitian sejenis: arsitektur
tetap YOLOv11 varian **nano** (bukan varian yang lebih besar), karena
mayoritas penelitian deteksi kematangan TBS sawit berbasis YOLOv11 yang
dijumpai pada tinjauan pustaka penelitian ini memilih varian nano justru
demi ruang gerak *deployment* CPU/edge -- konsisten dengan tujuan
*deployment* CPU-only pada Bab 4 -- dan *local optimizer* diganti dari SGD
ke **AdamW**, karena tiap ronde federasi hanya menjalankan sedikit *epoch*
lokal sebelum agregasi, sebuah rezim yang lebih diuntungkan oleh *learning
rate* adaptif AdamW dibandingkan SGD momentum yang biasanya memerlukan
penjadwalan *learning rate* lebih panjang. Berangkat dari keempat elemen di
atas, penelitian ini mengusulkan **FedX-Palm: A YOLOv11-Based Federated
Learning Framework with Differential Privacy and Explainable AI for Oil Palm
Ripeness Detection**. HFL menjadi tulang punggung komputasi terdistribusi,
DP-SGD per-sampel berfungsi sebagai lapisan proteksi matematis, dan
Grad-CAM++ menyediakan jendela interpretabilitas. FedX-Palm diharapkan
menjadi rujukan rancangan AI pertanian yang akurat, melindungi privasi data,
sekaligus keputusannya dapat dipertanggungjawabkan.

## 1.2 Rumusan Masalah

Praktik penilaian kematangan TBS yang berjalan saat ini meninggalkan tiga
persoalan yang saling berkelindan. Pertama, inspeksi visual manual bersifat
subjektif dan tidak konsisten antar pemanen, sehingga otomatisasi berbasis
*deep learning* menjadi kebutuhan. Kedua, otomatisasi itu sendiri menuntut
data citra dalam jumlah besar, sementara pengumpulan terpusat berisiko
menyingkap informasi operasional kebun dan rentan terhadap serangan
inferensi pada parameter model. Ketiga, model deteksi modern bersifat
*black-box*, sehingga keputusannya sulit dipercaya oleh agronom maupun
pemangku kepentingan. Ketiga persoalan inilah yang melatari kebutuhan akan
desain terpadu *Federated Learning*, *Differential Privacy*, dan *Explainable
AI*. Berdasarkan latar tersebut, rumusan masalah penelitian ini adalah:

1. Mengingat pendekatan pelatihan terpusat berisiko membocorkan data
   operasional perkebunan, bagaimana arsitektur *Federated Learning* (FL)
   dapat dirancang agar model deteksi objek YOLOv11 dapat dilatih secara
   kolaboratif tanpa harus memindahkan citra mentah dari masing-masing
   lokasi perkebunan?

2. Mengingat pertukaran parameter model masih rawan terhadap *membership
   inference* dan *model inversion*, sejauh mana mekanisme *Differential
   Privacy* berbasis DP-SGD per-sampel mampu memberikan proteksi formal, dan
   bagaimana dampaknya terhadap utilitas model seiring variasi anggaran
   privasi ε dan skala jaringan federasi (jumlah klien K)?

3. Mengingat karakter *black-box* model deteksi menghambat kepercayaan
   pengguna, bagaimana penerapan teknik *Explainable AI* (XAI) berbasis
   Grad-CAM++ dapat memberikan penjelasan visual yang dapat
   dipertanggungjawabkan atas keputusan deteksi yang dihasilkan oleh model
   YOLOv11 hasil federasi?

## 1.3 Tujuan Penelitian

Sejalan dengan rumusan masalah di atas, penelitian ini diarahkan untuk:

1. Membangun arsitektur *Federated Learning* yang terintegrasi dengan model
   YOLOv11n untuk klasifikasi enam tingkat kematangan TBS kelapa sawit dalam
   skema komputasi terdistribusi, dievaluasi pada dataset publik baru dari
   Roboflow.

2. Menyelidiki dan mengukur dampak penerapan *Differential Privacy*
   berbasis DP-SGD per-sampel terhadap utilitas deteksi melalui *sweep noise
   multiplier* (σ) dan jumlah klien (K); membandingkan strategi *full*
   DP-SGD dengan *partial* DP-SGD (*backbone* beku) untuk mengidentifikasi
   konfigurasi yang paling mempertahankan utilitas.

3. Mengembangkan komponen *Explainable AI* berbasis Grad-CAM++ yang
   menyediakan interpretasi visual yang dapat diverifikasi atas keputusan
   model YOLOv11 hasil federasi, divalidasi dengan metrik *Average Drop*
   dan *Focus Retention Rate*.

## 1.4 Batasan Masalah

Agar fokus dan hasilnya tetap terukur, penelitian dibatasi pada cakupan
berikut:

1. Subjek deteksi adalah citra Tandan Buah Segar (TBS) kelapa sawit yang
   dikategorikan ke dalam enam kelas kematangan [VERIFIKASI terhadap
   `data.yaml` dataset]: *Unripe*, *Underripe*, *Ripe*, *Overripe*, *Empty
   Bunch*, dan *Abnormal*.

2. Arsitektur deteksi yang dievaluasi adalah YOLOv11n (varian *nano*,
   ~2,6 juta parameter), dengan seluruh lapisan *Batch Normalization*
   dikonversi menjadi *Group Normalization*. Varian YOLOv11 lain (s/m/l/x)
   berada di luar cakupan utama, dipilih setelah tinjauan atas penelitian
   sejenis yang secara konsisten memakai varian nano untuk kebutuhan
   *deployment* CPU/edge.

3. Lingkungan FL dirancang sebagai sistem dengan satu *server aggregator*
   dan K *client node*, dengan K di-*sweep* pada {2, 4, 8, 12, 16}.
   Eksperimen pelatihan dijalankan sebagai *simulasi federated* yang
   ekuivalen pada satu GPU [VERIFIKASI: sebutkan tipe GPU] dengan skema
   pembagian data Non-IID (*Non-Independent and Identically Distributed*)
   berbasis distribusi Dirichlet. *Deployment* inferensi model akhir
   direalisasikan sebagai *image* Docker yang dibangun dan dijalankan pada
   satu VPS CPU-only; yang berada di luar cakupan hanyalah pelatihan FL
   terdistribusi lintas-host fisik dan orkestrasi multi-*container*.

4. Perlindungan privasi diwujudkan melalui DP-SGD per-sampel menggunakan
   *library* Opacus 1.5.4, yaitu *gradient clipping* per-sampel pada norma
   maksimum C = 1,0 yang dipadukan dengan *Gaussian noise* berskala σ pada
   gradien sebelum pembaruan bobot lokal. Konversi BatchNorm → GroupNorm
   dilakukan supaya gradien per-sampel terdefinisi dan model lolos
   *ModuleValidator* Opacus. Dua strategi DP-SGD dievaluasi sebagai
   eksperimen utama: *full* DP-SGD (E1, seluruh parameter) dan *partial*
   DP-SGD (E2, kepala deteksi saja, *backbone* dibekukan). Alternatif
   seperti DP-Adam, *secure aggregation*, maupun *homomorphic encryption*
   tidak diteliti.

5. *Sweep* parameter privasi mencakup σ ∈ {0,5; 1,0; 1,5; 2,0; 3,0} dengan
   *max_grad_norm* C = 1,0 tetap. Nilai *privacy budget* ε dihitung dari
   (σ, q, T, δ) menggunakan *accountant* PRV (*Privacy Random Variable*)
   Opacus pada δ = 1×10⁻⁵, bukan ditetapkan secara manual.

6. Teknik XAI yang digunakan terbatas pada Grad-CAM++ untuk menghasilkan
   peta atensi (*heatmap*) atas keputusan YOLOv11, dengan validasi
   kuantitatif memakai metrik *Average Drop* (AD, berbasis oklusi terhadap
   wilayah yang disorot penjelasan) dan *Focus Retention Rate* (FRR, rasio
   massa aktivasi peta panas yang jatuh di dalam kotak *ground-truth*
   dibanding total). Definisi operasional kedua metrik dijabarkan pada Bab
   2 dan Bab 3. Evaluasi *faithfulness* per-kelas dihitung langsung pada
   model federasi hasil agregasi.

## 1.5 Metode Penelitian

Penelitian ini menggunakan paradigma eksperimen kuantitatif yang
dikombinasikan dengan pendekatan *Research and Development* (R&D).
Eksperimen difokuskan pada pengembangan iteratif kerangka FedX-Palm dan
pengukuran dampak penyisipan komponen *Differential Privacy* (DP) serta
*Explainable AI* (XAI) terhadap performa deteksi YOLOv11n dalam lingkungan
komputasi terdistribusi.

### 1.5.1 Lingkungan dan Alat Implementasi

Rincian lingkungan pengembangan dan eksekusi eksperimen adalah:

1. **Platform Komputasi.** Seluruh pelatihan model YOLOv11n dijalankan pada
   [VERIFIKASI: workstation/server dengan akselerator GPU NVIDIA RTX --
   sebutkan tipe dan kapasitas VRAM]. Bobot hasil pelatihan disimpan dalam
   format `.pt` sebagai *checkpoint* untuk tahap evaluasi, eksplanasi XAI,
   dan penyiapan *blueprint deployment* Docker.

2. **Stack Perangkat Lunak.** Implementasi memanfaatkan Python 3.11 di atas
   *framework* PyTorch 2.5.1, dengan paket Ultralytics 8.4.51 untuk YOLOv11.
   Proteksi privasi DP-SGD per-sampel diterapkan melalui *library* Opacus
   1.5.4 dengan `PrivacyEngine`, setelah konversi BatchNorm → GroupNorm agar
   model lolos validasi per-sampel. Pelatihan federasi dijalankan sebagai
   simulasi ekuivalen menggunakan *loop* FedAvg *sequential* pada satu GPU.
   Model akhir di-*deploy* sebagai layanan inferensi dalam *image* Docker
   (Flask + Ultralytics CPU) yang dibangun dan dijalankan pada satu VPS;
   *Dockerfile* dan skrip inferensi disertakan untuk reprodusibilitas (lihat
   Bab 4).

3. **Dataset.** Citra TBS sawit enam kelas kematangan diperoleh dari
   platform Roboflow (proyek *palm-fruit-ripeness-detection-f6sac-ccb2z*,
   versi 2, workspace *dydy-worker*, format ekspor `yolov11`), lalu dibagi
   ke K *client node* mengikuti distribusi Dirichlet dengan parameter
   konsentrasi α = 0,5 untuk mensimulasikan kondisi Non-IID pada perkebunan
   nyata.

### 1.5.2 Tahapan Penelitian FedX-Palm

Realisasi sistem FedX-Palm dijalankan melalui delapan tahap berurutan
berikut.

1. **Studi Literatur.** Difokuskan pada empat bidang, yaitu *Federated
   Learning*, *Differential Privacy*, Grad-CAM++ sebagai metode XAI, dan
   arsitektur YOLOv11. Agenda utamanya adalah memetakan celah riset seputar
   keseimbangan antara perlindungan privasi dan keterjelasan model pada
   konteks deteksi kematangan buah sawit.

2. **Perancangan Sistem.** Topologi FL dirancang dalam pola *server-client*
   terpusat yang terdiri atas satu *server aggregator* dan K *client node*.
   YOLOv11n dipilih sebagai tulang punggung model karena menyatukan tugas
   lokalisasi dan klasifikasi pada satu *inference pipeline*, dan tetap
   ringan untuk *deployment* CPU pada tahap akhir.

3. **Penyiapan dan Distribusi Dataset.** Citra TBS sawit diunduh dari
   Roboflow, dipecah ulang menjadi *train/validation/test* berbasis
   identitas tandan (`bunch_id`) untuk mencegah kebocoran data (*leakage*)
   antar-*split*, lalu dipecah ke K klien melalui *sampling* Dirichlet,
   sehingga proporsi kelas kematangan tidak seragam antar klien dan
   mendekati heterogenitas data perkebunan di lapangan.

4. **Pelatihan Lokal (*Local Fine-tuning*).** Pada setiap ronde, klien
   menjalankan *fine-tuning* YOLOv11n atas porsi data lokalnya. *Optimizer*
   yang dipakai adalah AdamW, dengan *loss* *Complete IoU* (CIoU) untuk
   regresi *bounding box* yang dipadukan dengan komponen *loss* klasifikasi
   dan *Distribution Focal Loss* (DFL).

5. **Penyuntikan *Differential Privacy* pada Gradien Per-sampel.** Selama
   pelatihan lokal, mekanisme DP-SGD per-sampel diberlakukan melalui
   `PrivacyEngine` Opacus. Dua langkahnya: norma gradien per-sampel dibatasi
   melalui *clipping* C, lalu *Gaussian noise* berskala σ ditambahkan pada
   gradien yang sudah dijumlahkan per-*batch* sebelum pembaruan bobot.
   *Privacy budget* ε dihitung oleh *accountant* Opacus (PRV) dari
   (σ, q, T, δ), bukan ditetapkan manual.

6. **Agregasi Bobot Global (*Federated Averaging*).** *Server*
   mengonsolidasikan bobot dari semua klien menggunakan FedAvg untuk
   membentuk model global baru $\mathbf{w}_{t+1}$:

   $$\mathbf{w}_{t+1} = \sum_{k=1}^{K} \frac{n_k}{N} \mathbf{w}_k$$

   dengan $\mathbf{w}_k$ adalah bobot lokal klien ke-$k$, $n_k$ ukuran data
   lokalnya, dan $N$ jumlah total sampel di semua klien. Bobot global
   kemudian dikirim kembali ke setiap klien untuk siklus pelatihan
   berikutnya hingga 40 ronde federasi tercapai.

7. **Pengujian Transparansi XAI.** Model global hasil agregasi diperiksa
   kualitas interpretasinya menggunakan Grad-CAM++. Validasinya tidak
   berhenti pada inspeksi visual; dikuatkan oleh dua indikator kuantitatif,
   yaitu *Average Drop* dan *Focus Retention Rate* (FRR), yang definisi
   operasionalnya dijabarkan pada Bab 2 dan Bab 3.

8. **Evaluasi Akhir dan Sintesis Hasil.** Evaluasi penutup memetakan kinerja
   sistem dalam tiga dimensi sekaligus: (a) metrik deteksi standar
   (mAP@0.5, mAP@0.5:0.95, *Precision*, *Recall*, dan F1-Score); (b)
   sensitivitas model terhadap variasi *privacy budget* ε dan jumlah klien
   K; serta (c) kualitas penjelasan visual yang diukur dengan *Average
   Drop* dan FRR.

## 1.6 Hipotesis

Mengacu pada kerangka teoretis dan rancangan metodologi yang telah
diuraikan, penelitian ini menetapkan empat hipotesis kerja berikut.

1. **H1 -- Kelayakan Deteksi YOLOv11n pada Skenario Non-IID.** Pelatihan
   kolaboratif YOLOv11n dengan algoritma *Federated Averaging* (FedAvg)
   tanpa DP di atas *client node* berdistribusi Non-IID diperkirakan mampu
   menghasilkan model global dengan *mean Average Precision* (mAP@0.5)
   dalam kategori layak (*acceptable*, ≥ 0,70) dan mendekati performa
   pelatihan terpusat, dengan selisih utilitas (*FL-cost*) yang kecil.

2. **H2 -- Bentuk *Trade-off* Privasi versus Utilitas.** Penerapan DP-SGD
   per-sampel pada YOLOv11n diasumsikan menurunkan akurasi deteksi seiring
   penguatan privasi (semakin kecil ε, semakin besar penurunan), namun
   degradasinya diperkirakan bersifat bertahap (*gradual*) di sepanjang
   rentang ε yang diuji.

3. **H2-K -- Pengaruh Skala Jaringan pada DP-SGD Per-sampel.** Arah
   pengaruh jumlah klien K pada DP-SGD per-sampel diperkirakan berkebalikan
   dengan intuisi DP-FedAvg tingkat-klien: semakin besar K, semakin sedikit
   sampel per klien, sehingga rasio *sub-sampling* membesar, *noise*
   per-sampel makin mendominasi sinyal, dan utilitas menurun -- sementara
   anggaran privasi ε justru membengkak. Hipotesis ini diuji secara empiris
   dan dibahas pada Bab 4.

4. **H3 -- Validitas Interpretasi Visual Model.** Integrasi Grad-CAM++ pada
   model hasil federasi diprediksi menghasilkan peta panas yang konsisten
   dengan fitur morfologis buah sawit. Hal ini ditunjukkan oleh nilai
   *Average Drop* dan *Focus Retention Rate* yang mengindikasikan bahwa
   keputusan model bertumpu pada wilayah objek (TBS), bukan pada latar.
