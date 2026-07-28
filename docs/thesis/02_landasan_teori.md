<!--
CATATAN PENULISAN (hapus komentar ini sebelum submit):
Bab ini dipersempit selaras dengan penyempitan cakupan Bab 1: materi DP-SGD
formal (definisi (ε,δ)-DP, gradient clipping, Gaussian mechanism, akuntansi
privasi, Opacus) dan Explainable AI (Grad-CAM++, Average Drop, Focus
Retention Rate) serta containerization/deployment DIHAPUS dari landasan
teori inti karena bukan bagian dari rumusan masalah/tujuan/hipotesis tahap
ini (lihat docs/thesis/01_pendahuluan.md). Risiko privasi FL tetap dibahas
secara konseptual singkat (Subbab 2.4) karena itu tetap jadi motivasi utama
pemilihan FL, tanpa masuk ke matematika DP formal. Rujukan bernomor [n]
adalah placeholder -- lengkapi dengan sitasi pustaka aktual sebelum submit;
lihat docs/thesis/NOTES_FOR_RACHMA.md.
-->

# CHAPTER 2 -- LANDASAN TEORI

Bab ini membahas dasar-dasar teori yang menopang penelitian tahap ini,
dimulai dari domain aplikasi (kematangan TBS sawit), arsitektur detektor
(YOLOv11), kerangka pembelajaran terdistribusi (*Federated Learning*),
risiko privasi yang memotivasi penggunaannya, modifikasi arsitektural yang
dibutuhkan (BatchNorm ke GroupNorm), serta posisi penelitian ini terhadap
penelitian sejenis.

## 2.1 Klasifikasi Kematangan Kelapa Sawit

### 2.1.1 Tandan Buah Segar (TBS)

Kelapa sawit (*Elaeis guineensis* Jacq.) merupakan penghasil minyak nabati
dengan produktivitas tertinggi per hektar di antara komoditas serupa.
Buahnya tumbuh dalam Tandan Buah Segar (TBS) yang dapat memuat ribuan
brondolan. Kualitas *Crude Palm Oil* (CPO) yang dihasilkan sangat ditentukan
oleh tingkat kematangan TBS saat dipanen [n]. Tandan yang dipanen terlalu
dini, yang dikategorikan sebagai *Unripe* atau *Underripe*, memiliki
kandungan minyak rendah karena pembentukan minyak di mesokarp belum optimal.
Sebaliknya, tandan yang terlambat dipanen (*Overripe*) mengalami peningkatan
kadar Asam Lemak Bebas (*Free Fatty Acid*, FFA) yang menurunkan mutu CPO
[n].

### 2.1.2 Kriteria Kematangan Berdasarkan Perubahan Warna

Standar industri membagi kematangan TBS ke dalam enam kelas: *Unripe*,
*Underripe*, *Ripe*, *Overripe*, *Empty Bunch*, dan *Abnormal*. Enam kelas
ini mencerminkan protokol penilaian operasional, bukan skala kematangan
fisiologis yang murni linear: *Unripe* dan *Underripe* menangkap tahap awal
perubahan warna dengan brondolan lepas yang sangat sedikit atau tidak ada;
*Ripe* dan *Overripe* adalah tahap optimal-panen dan pasca-optimal, dibedakan
terutama oleh proporsi brondolan lepas dan tingkat perkembangan warna;
sedangkan *Empty Bunch* dan *Abnormal* merepresentasikan kondisi tandan
non-standar -- masing-masing tandan yang sebagian besar buahnya sudah lepas,
dan tandan yang bentuk/kepadatan/perkembangannya menyimpang dari pola normal
-- bukan titik pada kontinum kematangan. Struktur kelas ini relevan untuk
menafsirkan hasil per-kelas pada Subbab 4.1, karena *Ripe* dan *Overripe*
secara visual berdekatan dengan tahap tetangganya, sedangkan *Empty Bunch*
dan *Abnormal* adalah kondisi luar (*outlier*) yang secara visual lebih
khas.

1. ***Unripe* (Mentah).** Buah berwarna hitam pekat atau hijau tua dengan
   kandungan minyak yang masih sangat minim.
2. ***Underripe* (Kurang Matang).** Buah mulai berubah warna dari hitam ke
   jingga, namun warna hitam masih terlihat mendominasi.
3. ***Ripe* (Matang).** Buah berwarna jingga kemerahan. Kondisi ini paling
   optimal untuk dipanen karena memiliki rendemen minyak tertinggi.
4. ***Overripe* (Lewat Matang).** Buah berwarna merah tua keunguan dan
   banyak brondolan yang sudah lepas secara alami. Kadar Asam Lemak Bebas
   biasanya mulai meningkat.
5. ***Abnormal*.** Buah mengalami gangguan pertumbuhan sehingga bentuknya
   tidak mengikuti standardisasi buah normal.
6. ***Empty Bunch* (Tandan Kosong).** Tandan yang sebagian besar buahnya
   sudah lepas atau tidak berkembang, sehingga tidak memiliki nilai
   komersial untuk pengambilan minyak.

Pada praktiknya, penilaian dilakukan oleh mandor panen secara manual. Cara
tersebut subjektif, sangat bergantung pada pengalaman individu, dan rentan
terhadap kelelahan operator. Persoalan konsistensi inilah yang membuka jalan
bagi otomasi berbasis visi komputer untuk perkebunan skala besar.

## 2.2 Deteksi Objek dan YOLOv11

Deteksi objek (*object detection*) adalah tugas visi komputer yang menuntut
model tidak hanya mengklasifikasikan objek, tetapi juga melokalisasi
posisinya melalui *bounding box*. Berbeda dengan klasifikasi citra yang
menghasilkan satu label per gambar, deteksi objek menghasilkan himpunan
(kelas, kotak, skor) untuk setiap instans. Pada konteks TBS, kemampuan ini
memungkinkan deteksi beberapa tandan sekaligus dengan tingkat kematangan
berbeda dalam satu bingkai kamera. Kualitas deteksi diukur dengan
*Intersection over Union* (IoU) antara kotak prediksi $b$ dan kotak
kebenaran-dasar $b^{gt}$:

$$\text{IoU}(b, b^{gt}) = \frac{|b \cap b^{gt}|}{|b \cup b^{gt}|}. \tag{2.1}$$

### 2.2.1 YOLOv11 (*You Only Look Once* v11)

Keluarga model YOLO (*You Only Look Once*) memformulasikan deteksi sebagai
persoalan regresi tunggal yang dipetakan langsung dari piksel ke koordinat
kotak dan probabilitas kelas dalam satu *forward pass*, berbeda dengan
detektor dua tahap (mis. Faster R-CNN) yang memisahkan pengusulan wilayah
dan klasifikasi. YOLOv11 melanjutkan garis evolusi ini dengan tiga komponen
utama: **backbone**, **neck**, dan **head**.

*Backbone* bertugas mengekstraksi fitur multi-skala dari citra masukan.
Pada YOLOv11, *backbone* tersusun dari blok konvolusi (`Conv`), blok C3k2
(varian dari blok CSP -- *Cross Stage Partial* -- yang membagi peta fitur
untuk mengurangi biaya komputasi tanpa mengorbankan kekayaan fitur), blok
*Spatial Pyramid Pooling - Fast* (SPPF) untuk memperbesar *receptive field*
secara efisien, dan blok C2PSA (*Cross Stage Partial with Spatial
Attention*) yang menambahkan mekanisme atensi ruang pada level fitur
terdalam. *Neck* berupa arsitektur PAN-FPN (*Path Aggregation Network -
Feature Pyramid Network*) yang menggabungkan fitur dari berbagai skala
melalui operasi *upsample* dan *concatenate*, sehingga objek berukuran
kecil maupun besar sama-sama terwakili sebelum diteruskan ke *head*.
*Head* YOLOv11 bersifat *decoupled* dan *anchor-free*: cabang regresi kotak
dan cabang klasifikasi kelas dipisah menjadi dua jalur konvolusi yang
berbeda, tanpa bergantung pada *anchor box* yang telah ditentukan sebelumnya
seperti pada generasi YOLO lebih awal. Penelitian ini memakai varian
**nano** (YOLOv11n, 2.591.010 parameter setelah konversi GroupNorm --
Subbab 2.5) karena mayoritas penelitian sejenis pada domain kematangan TBS
sawit yang ditinjau memilih varian ini demi ruang gerak *deployment*
CPU/*edge* [n][n].

### 2.2.2 Fungsi Kerugian YOLOv11

Fungsi kerugian (*loss function*) YOLOv11 tersusun atas tiga komponen yang
dijumlahkan berbobot:

$$\mathcal{L} = \lambda_{box}\,\mathcal{L}_{CIoU} + \lambda_{cls}\,\mathcal{L}_{cls} + \lambda_{dfl}\,\mathcal{L}_{DFL} \tag{2.2}$$

1. ***Complete IoU* (CIoU) Loss** untuk regresi kotak, memperluas IoU biasa
   dengan mempertimbangkan jarak titik pusat kotak prediksi dan
   *ground-truth*, serta konsistensi rasio aspek keduanya, sehingga
   konvergensi regresi kotak lebih stabil dibanding IoU polos.
2. ***Classification Loss*** berupa *Binary Cross-Entropy* per kelas, karena
   YOLOv11 memperlakukan klasifikasi multi-kelas sebagai serangkaian
   keputusan biner independen per kelas pada tiap kandidat deteksi.
3. ***Distribution Focal Loss* (DFL)** yang memodelkan setiap sisi kotak
   sebagai distribusi probabilitas diskret atas kemungkinan posisi (bukan
   nilai regresi tunggal), sehingga batas kotak yang ambigu pada objek
   dengan tepi kabur -- relevan untuk brondolan TBS yang saling tumpang
   tindih -- dapat direpresentasikan lebih baik.

Sebelum ketiga komponen kerugian ini dihitung, setiap prediksi perlu
dipasangkan (*assigned*) dengan target *ground-truth* yang sesuai. YOLOv11
memakai **Task-Aligned Assigner**, yang memilih kandidat positif berdasarkan
skor keselarasan antara kualitas klasifikasi dan kualitas lokalisasi
sekaligus, alih-alih hanya berbasis IoU seperti pada skema *assignment*
generasi awal YOLO.

## 2.3 Federated Learning

*Federated Learning* (FL) adalah paradigma pembelajaran mesin terdistribusi
yang memungkinkan banyak pihak melatih satu model bersama tanpa
mempertukarkan data mentah mereka [n]. Setiap partisipan (*client*) melatih
model secara lokal di atas datanya sendiri, dan hanya pembaruan
parameter -- bukan data -- yang dikirim ke *server* pengagregasi.

### 2.3.1 Algoritma FedAvg

*Federated Averaging* (FedAvg), diperkenalkan oleh McMahan dkk. (2017) [n],
adalah algoritma agregasi FL paling mendasar dan menjadi tulang punggung
penelitian ini. Pada setiap ronde komunikasi $t$, *server* mengirim model
global $\mathbf{w}_t$ ke seluruh (atau subset) klien; setiap klien $k$
menjalankan beberapa langkah SGD lokal di atas datanya sendiri untuk
menghasilkan $\mathbf{w}_k$; *server* kemudian merata-ratakan pembaruan
tersebut secara berbobot berdasarkan ukuran data lokal:

$$\mathbf{w}_{t+1} = \sum_{k=1}^{K} \frac{n_k}{N}\,\mathbf{w}_k, \qquad N = \sum_{k=1}^{K} n_k. \tag{2.3}$$

Pembobotan $n_k/N$ ini penting pada skenario Non-IID: klien dengan lebih
banyak sampel diberi pengaruh lebih besar terhadap model global, mencegah
klien bersampel sedikit mendominasi arah pembaruan secara tidak
proporsional -- relevan langsung bagi penelitian ini, karena keempat klien
yang dipakai (Subbab 3.4) memegang porsi data yang jauh dari seimbang
(8,67% hingga 59,36% dari *split train*).

### 2.3.2 *Cross-silo* dan *Cross-device*

Literatur FL membedakan dua skenario deployment berdasarkan karakteristik
klien [n]. **Cross-device** melibatkan jutaan klien tak-andal (mis.
ponsel), masing-masing dengan data sangat sedikit dan konektivitas tidak
stabil. **Cross-silo** melibatkan sedikit klien (puluhan hingga ratusan)
yang relatif andal dan bertahan sepanjang pelatihan -- misalnya organisasi,
rumah sakit, atau, pada konteks penelitian ini, kebun/afdeling kelapa sawit
yang masing-masing memiliki infrastruktur komputasi sendiri. Penelitian ini
beroperasi pada rezim *cross-silo* dengan K = 4 klien tersimulasi,
merepresentasikan gambaran awal jumlah kebun/afdeling yang berpartisipasi
dalam federasi -- bukan perangkat individu. Klien-klien ini adalah pecahan
data (*data shard*) tersimulasi pada satu GPU, bukan representasi empat
perkebunan, afdeling, organisasi, atau perangkat fisik yang berbeda (lihat
Subbab 4.6 untuk keterbatasan ini).

### 2.3.3 Heterogenitas Non-IID dan Partisi Dirichlet

Asumsi data *identically and independently distributed* (IID) antar klien
jarang berlaku pada kondisi lapangan: komposisi kelas kematangan TBS antara
satu kebun dan kebun lain dapat berbeda jauh, dipengaruhi oleh jadwal panen,
varietas, dan kondisi agronomis setempat. Untuk mensimulasikan heterogenitas
ini secara terkendali, penelitian menggunakan **partisi Dirichlet**
berbasis skema *latent Dirichlet allocation* [n]: untuk setiap kelas $c$,
proporsi sampel yang jatuh ke tiap klien ditarik dari distribusi Dirichlet
$\mathrm{Dir}(\alpha, \ldots, \alpha)$ atas $K$ klien. Parameter konsentrasi
$\alpha$ mengendalikan derajat heterogenitas: $\alpha \to 0$ menghasilkan
partisi yang sangat timpang (satu klien mendominasi satu kelas), sedangkan
$\alpha \to \infty$ mendekati pembagian IID seragam. Nilai konsentrasi ini
dipilih langsung karena secara langsung mengendalikan derajat *skew*
distribusi kelas antar klien tanpa memerlukan aturan alokasi heuristik
tambahan [16]. Penelitian ini memakai $\alpha = 0{,}5$ dengan K = 4 klien
dan *partition seed* = 42, nilai moderat yang umum dipakai pada literatur FL
untuk mensimulasikan heterogenitas realistis tanpa membuat sebagian klien
kehabisan sampel kelas tertentu sama sekali; hasil partisi konkretnya
dilaporkan pada Tabel 3.2.

## 2.4 Risiko Privasi sebagai Motivasi *Federated Learning*

Meski FL menghindari pengumpulan data mentah ke satu server, hal ini tidak
serta-merta menghilangkan seluruh risiko privasi -- pertukaran parameter
model tetap membuka permukaan serangan. Dua ancaman inferensi yang paling
sering disebut pada literatur:

1. ***Membership Inference Attack* (MIA).** Penyerang mencoba menebak
   apakah suatu sampel data tertentu pernah dipakai dalam pelatihan model,
   dengan mengamati pola keluaran atau gradien model terhadap sampel
   tersebut [n].
2. ***Gradient Inversion Attack*.** Penyerang mencoba merekonstruksi
   kembali citra masukan asli dari gradien yang teramati, dengan
   mengoptimalkan citra sintetis sedemikian rupa hingga gradiennya
   menyerupai gradien yang disadap [n]. Serangan ini terbukti efektif
   terutama pada *batch* berukuran kecil.

Kedua risiko ini adalah alasan mengapa "menghindari pengiriman citra
mentah" saja belum menjadi jaminan privasi yang lengkap, dan mengapa
*Differential Privacy* (DP) -- khususnya DP-SGD per-sampel -- relevan
sebagai lapisan proteksi formal tambahan di atas arsitektur FL. Kajian
formal DP-SGD (definisi $(\varepsilon,\delta)$-DP, *gradient clipping*,
mekanisme Gaussian, akuntansi privasi) **tidak** dibahas pada bab ini karena
berada di luar cakupan penelitian tahap ini (Subbab 1.4); kajian tersebut
akan menjadi materi landasan teori pada laporan lanjutan yang membangun di
atas titik rujukan B1/B2 di sini. Yang tetap relevan pada tahap ini adalah
prasyarat arsitekturalnya: DP-SGD per-sampel mensyaratkan gradien tiap
sampel terdefinisi secara independen, sebuah sifat yang tidak dimiliki
*Batch Normalization* (Subbab 2.5) -- sehingga substitusi BatchNorm ->
GroupNorm sudah diterapkan sejak tahap B1/B2 ini, agar tahap lanjutan tidak
perlu melatih ulang model dari arsitektur yang berbeda.

## 2.5 BatchNorm versus GroupNorm

*Batch Normalization* (BatchNorm) menormalisasi aktivasi suatu lapisan
dengan menghitung rata-rata dan varians **lintas seluruh sampel dalam satu
*batch***. Pada federasi Non-IID dengan *batch* kecil per klien, statistik
*batch* ini menjadi tidak stabil: sedikit sampel per klien membuat estimasi
rata-rata/varians *batch* memiliki varians tinggi dan tidak representatif
terhadap distribusi data klien tersebut secara keseluruhan, terlebih ketika
komposisi kelas antar klien timpang (Subbab 2.3.3). BatchNorm juga secara
fundamental tidak kompatibel dengan gradien per-sampel yang dibutuhkan
DP-SGD pada tahap penelitian lanjutan (Subbab 2.4): statistik *batch*-nya
membuat gradien satu sampel bergantung pada sampel lain dalam *batch* yang
sama, sehingga konsep "gradien per-sampel yang independen" tidak
terdefinisi dengan bersih.

*Group Normalization* (GroupNorm) [n] menjadi pengganti yang mengatasi
kedua persoalan di atas sekaligus: alih-alih menormalisasi lintas *batch*,
GroupNorm membagi kanal suatu lapisan menjadi beberapa grup dan
menormalisasi **di dalam satu sampel saja** (lintas grup kanal, bukan
lintas sampel), sehingga tidak bergantung pada ukuran atau komposisi
*batch*. Penelitian ini mengonversi seluruh **81 lapisan BatchNorm2d** pada
YOLOv11n menjadi GroupNorm -- audit arsitektur akhir mengonfirmasi 0 modul
BatchNorm tersisa dan 81 modul GroupNorm -- dengan jumlah grup pada tiap
lapisan dipilih sebagai pembagi terbesar dari jumlah kanal lapisan tersebut
yang tidak melebihi 32, mengikuti rekomendasi umum literatur GroupNorm.
Konversi ini diterapkan secara **seragam pada B1 dan B2**, bukan hanya pada
model yang nantinya dipakai DP-SGD, agar selisih performa B1-versus-B2 pada
Bab 4 murni mencerminkan efek federasi, bukan tercampur dengan efek
perbedaan arsitektur normalisasi.

## 2.6 Penelitian Terkait dan Posisi Penelitian Ini

Deteksi objek berbasis *deep learning* untuk kematangan TBS sawit
sebelumnya telah diteliti memakai detektor satu-tahap, khususnya arsitektur
berbasis YOLO dengan *backbone* konvolusional seperti CSPNet [9] dan
jaringan residual [8], karena performa *real-time*-nya [3]-[7]. Sebagai
contoh, Lai dkk. menerapkan YOLOv4 untuk deteksi tandan matang secara
*real-time* di lingkungan perkebunan [3], sementara Suharjito dkk.
menyasar *deployment* pada perangkat *mobile* untuk klasifikasi kematangan
TBS [4]; sebuah tinjauan terbaru atas metode klasifikasi kematangan TBS
mencatat bahwa mayoritas sistem yang dilaporkan masih berupa model
tersentral pada satu lokasi, dengan mekanisme tata kelola data lintas-lokasi
seperti federasi baru jarang dibahas [7]. *Federated Learning* secara
terpisah telah diteliti untuk tugas klasifikasi dan deteksi visual di bawah
distribusi data Non-IID [12]-[14], [16], termasuk skema federasi yang
menormalisasi aktivasi tanpa bergantung pada statistik *batch* yang tidak
stabil di bawah heterogenitas klien [11], [15].

Kombinasi spesifik antara (i) deteksi kematangan TBS multi-kelas berbasis
YOLO11, (ii) pengaturan federasi Non-IID (partisi Dirichlet, K = 4 klien)
yang dievaluasi di bawah pembagian data bebas-kebocoran pada level kelompok
sumber, dan (iii) adopsi Group Normalization yang secara khusus dipilih
untuk menghilangkan ketergantungan statistik *batch* di bawah pelatihan
federasi Non-IID, sejauh penelusuran yang dilakukan belum ditemukan
dibahas bersamaan untuk tugas ini. Tabel 2.1 merangkum posisi penelitian
ini terhadap penelitian deteksi kematangan TBS sawit berbasis YOLO yang
paling relevan.

Tabel 2.1. Posisi penelitian ini terhadap penelitian sejenis

| Penelitian | Model | FL | Pembagian bebas-kebocoran | Catatan |
|---|---|---|---|---|
| Lai dkk. [3] | YOLOv4 | Tidak | Tidak dilaporkan | Deteksi tandan matang *real-time* di perkebunan |
| Suharjito dkk. [4] | Model klasifikasi kematangan (varian *mobile*) | Tidak | Tidak dilaporkan | Target *deployment* perangkat *mobile* |
| Asrol dkk. [6] | YOLOv4 (modifikasi) | Tidak | Tidak dilaporkan | Sistem *grading* *real-time* via *smartphone* |
| Goh dkk. (tinjauan) [7] | Beragam | Mayoritas tidak | Jarang dibahas eksplisit | Tinjauan metode klasifikasi kematangan TBS |
| **B1/B2 (penelitian ini)** | YOLOv11n + GroupNorm | **Ya** (FedAvg, K = 4, Non-IID) | **Ya**, diaudit eksplisit | Titik rujukan tersentral-versus-federasi non-privat; dasar bagi perluasan DP-SGD tahap lanjutan |

Penelitian ini tidak mengklaim sebagai penerapan pertama *Federated
Learning* pada citra sawit atau penggunaan pertama YOLO untuk klasifikasi
kematangan TBS; kontribusinya adalah menetapkan titik rujukan empiris
tersentral-versus-federasi untuk tugas, arsitektur, dan pengaturan federasi
spesifik ini, sebagai titik rujukan non-privat bagi program riset
*Differentially Private Federated Learning* yang lebih luas (di luar
cakupan bab ini, lihat Subbab 1.4 dan 5.3).
