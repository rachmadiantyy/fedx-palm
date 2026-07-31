<!--
CATATAN PENULISAN (hapus komentar ini sebelum submit):
Bab ini diperluas agar konsisten dengan cakupan penuh Bab 1 dan Bab 4:
mencakup teori formal DP-SGD ((ε,δ)-DP, gradient clipping, mekanisme
Gaussian, akuntansi privasi, Opacus), Explainable AI (Grad-CAM++, Average
Drop, Focus Retention Rate), dan containerization/deployment -- yang pada
draf sebelumnya sengaja dihapus dari landasan teori inti (lihat riwayat
pada docs/thesis/NOTES_FOR_RACHMA.md), namun sekarang relevan penuh
karena E1/E2/XAI/deployment sudah dikerjakan dan dilaporkan di Bab 4.
Rujukan bernomor [n] adalah placeholder -- lengkapi dengan sitasi pustaka
aktual sebelum submit.
-->

# CHAPTER 2 -- LANDASAN TEORI

Bab ini membahas dasar-dasar teori yang menopang penelitian ini, dimulai
dari domain aplikasi (kematangan TBS sawit), arsitektur detektor (YOLOv11),
kerangka pembelajaran terdistribusi (*Federated Learning*), risiko privasi
dan *Differential Privacy* (DP-SGD) sebagai lapisan proteksi formalnya,
modifikasi arsitektural yang dibutuhkan (BatchNorm ke GroupNorm),
*Explainable AI* berbasis Grad-CAM++, dasar *containerization*/*deployment*
model, serta posisi penelitian ini terhadap penelitian sejenis.

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
menafsirkan hasil per-kelas pada Subbab 4.2, karena *Ripe* dan *Overripe*
secara visual berdekatan dengan tahap tetangganya, sedangkan *Empty Bunch*
dan *Abnormal* adalah kondisi luar (*outlier*) yang secara visual lebih
khas; struktur ini juga relevan untuk memahami mengapa *Empty Bunch* menjadi
*failure case* ekstrem di bawah DP-SGD (Subbab 4.5.6).

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
CPU/*edge* [n][n], sekaligus lapisan konvolusi terakhir pada blok `C3k2`
(indeks tahap 22) yang memberi masukan ke kepala `Detect` menjadi
*target layer* analisis Grad-CAM++ pada Subbab 2.6.

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
generasi awal YOLO. Kontrak numerik fungsi kerugian ini -- khususnya konvensi
reduksi (`loss_reduction`) yang dipakai Ultralytics saat mengembalikan nilai
*loss* -- menjadi relevan langsung bagi implementasi DP-SGD per-sampel pada
Subbab 2.4.5, karena Opacus mengasumsikan konvensi reduksi tertentu untuk
merekonstruksi gradien per-sampel dengan benar (lihat Subbab 4.3).

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
(8,67% hingga 59,36% dari *split train*). Implementasi FedAvg pada
`fedxpalm/federated/fedavg.py` merata-ratakan seluruh kunci `state_dict`
klien tanpa memfilter berdasarkan status *trainable*/beku -- properti yang
menjadi relevan langsung bagi analisis biaya komunikasi E2 pada Subbab
4.8.3, karena parameter yang dibekukan pada skenario *partial* DP (Subbab
2.4.5) tetap ditransmisikan penuh oleh mekanisme agregasi ini.

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
Subbab 4.10.3 untuk keterbatasan ini).

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
dilaporkan pada Tabel 3.2. Ketimpangan ukuran data antar klien ini juga
langsung memengaruhi laju *sampling* $q$ yang dipakai akuntansi privasi
DP-SGD per klien (Subbab 2.4.4, Subbab 4.6.2).

## 2.4 *Differential Privacy* dan DP-SGD

### 2.4.1 Ancaman Inferensi pada *Federated Learning*

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
sebagai lapisan proteksi formal tambahan di atas arsitektur FL.

### 2.4.2 Definisi $(\varepsilon,\delta)$-*Differential Privacy*

*Differential Privacy* menyediakan jaminan matematis bahwa keluaran suatu
mekanisme (di sini, model yang dilatih) hampir tidak dapat dibedakan
apakah sampel data tertentu disertakan atau tidak dalam himpunan latihnya.
Secara formal, mekanisme $\mathcal{M}$ dikatakan $(\varepsilon,\delta)$-DP
jika untuk setiap dua *dataset* yang bertetangga $D, D'$ (berbeda tepat satu
sampel) dan setiap himpunan keluaran $S$:

$$\Pr[\mathcal{M}(D) \in S] \le e^{\varepsilon}\,\Pr[\mathcal{M}(D') \in S] + \delta. \tag{2.4}$$

Parameter $\varepsilon$ (anggaran privasi, *privacy budget*) mengukur
seberapa besar keluaran mekanisme boleh berbeda antara dua *dataset* yang
bertetangga -- semakin kecil $\varepsilon$, semakin ketat jaminan
privasinya, namun semakin besar pula *noise* yang harus disuntikkan
(sehingga utilitas model cenderung menurun). Parameter $\delta$ adalah
probabilitas kecil kegagalan jaminan tersebut (biasanya diset $\ll 1/N$,
dengan $N$ jumlah sampel).

### 2.4.3 DP-SGD: *Gradient Clipping* dan Mekanisme Gaussian

DP-SGD [n] mencapai jaminan $(\varepsilon,\delta)$-DP dengan memodifikasi
setiap langkah SGD lewat dua operasi tambahan pada gradien **per-sampel**
$g_i$ (bukan gradien rata-rata *batch*):

1. ***Gradient Clipping*.** Setiap gradien per-sampel dipotong normanya
   agar tidak melebihi ambang $C$:
   $$\bar{g}_i = g_i \big/ \max\!\left(1,\, \frac{\lVert g_i \rVert_2}{C}\right), \tag{2.5}$$
   membatasi pengaruh maksimum satu sampel terhadap pembaruan parameter.
   Ambang $C$ dapat diterapkan sebagai satu nilai global (*flat clipping*)
   atau diturunkan per-tensor/per-lapisan (*per-layer clipping*), dengan
   trade-off yang dianalisis pada Subbab 4.4.
2. **Penambahan *Noise* Gaussian.** *Noise* Gaussian berskala $\sigma C$
   ditambahkan pada rata-rata gradien yang sudah dipotong:
   $$\tilde{g} = \frac{1}{n}\left(\sum_{i=1}^{n} \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I})\right), \tag{2.6}$$
   dengan $\sigma$ (*noise multiplier*) mengendalikan besarnya *noise*
   relatif terhadap ambang *clipping* $C$ -- semakin besar $\sigma$,
   semakin ketat privasinya namun semakin besar pula degradasi utilitas.

Karena kedua operasi ini bekerja pada gradien **per-sampel**, bukan gradien
rata-rata *batch*, DP-SGD mensyaratkan lapisan normalisasi yang tidak
bergantung pada statistik lintas-sampel dalam satu *batch* -- prasyarat
inilah yang melatarbelakangi konversi BatchNorm ke GroupNorm pada Subbab
2.5.

### 2.4.4 Akuntansi Privasi dan *Accountant* PRV

Anggaran privasi $\varepsilon$ terakumulasi seiring bertambahnya langkah
optimisasi, dan besarnya bergantung pada tiga faktor: laju *sampling* $q$
(proporsi data yang disampel tiap langkah, relevan langsung bagi
ketimpangan antar klien pada Subbab 2.3.3), *noise multiplier* $\sigma$,
dan jumlah total langkah optimisasi privat. *Accountant* privasi menghitung
$\varepsilon$ terakumulasi ini secara ketat (bukan estimasi kasar); penelitian
ini memakai *accountant* **PRV** (*Privacy loss Random Variable*), yang
mengakumulasi $\varepsilon$ berdasarkan jumlah langkah optimisasi per klien
per ronde -- **bukan** berdasarkan jumlah atau proporsi parameter model yang
diperbarui secara *trainable*. Implikasi langsung dari sifat ini adalah
bahwa membatasi cakupan parameter yang dilatih secara privat (*partial* DP)
tidak serta-merta memberikan anggaran privasi yang "lebih murah" dibanding
melatih seluruh parameter (*full* DP) -- dianalisis empiris pada Subbab
4.6.3.

### 2.4.5 Opacus sebagai Pustaka Implementasi

Penelitian ini mengimplementasikan DP-SGD melalui **Opacus**, pustaka DP
untuk PyTorch yang menyediakan `PrivacyEngine` untuk membungkus model,
*optimizer*, dan `DataLoader` standar menjadi versi yang menghitung gradien
per-sampel, menerapkan *clipping* dan penambahan *noise* (Persamaan 2.5-2.6),
serta melacak anggaran $\varepsilon$ terakumulasi lewat *accountant* PRV.
Opacus mengasumsikan nilai *loss* yang diteruskan ke `backward()` berbentuk
rata-rata murni per-sampel (`loss_reduction="mean"`) untuk merekonstruksi
gradien per-sampel (`grad_sample`) dengan benar -- asumsi ini perlu
diverifikasi ulang terhadap konvensi *loss* kerangka kerja deteksi objek
yang dipakai (Ultralytics), karena ketidaksesuaian kontrak ini dapat secara
diam-diam menginflasi `grad_sample` tanpa memicu galat (*error*) apapun
(diaudit dan ditangani pada Subbab 4.3). Penelitian ini juga membedakan
konfigurasi **E1** (DP-SGD diterapkan pada seluruh parameter *trainable*)
dan **E2** (DP-SGD diterapkan hanya pada sebagian parameter, dengan
*backbone* dibekukan/*frozen* dan tidak ikut serta dalam mekanisme
`PrivacyEngine`), sebagai dua desain independen yang dibandingkan pada
Subbab 4.5 dan 4.6.3.

## 2.5 BatchNorm versus GroupNorm

*Batch Normalization* (BatchNorm) menormalisasi aktivasi suatu lapisan
dengan menghitung rata-rata dan varians **lintas seluruh sampel dalam satu
*batch***. Pada federasi Non-IID dengan *batch* kecil per klien, statistik
*batch* ini menjadi tidak stabil: sedikit sampel per klien membuat estimasi
rata-rata/varians *batch* memiliki varians tinggi dan tidak representatif
terhadap distribusi data klien tersebut secara keseluruhan, terlebih ketika
komposisi kelas antar klien timpang (Subbab 2.3.3). BatchNorm juga secara
fundamental tidak kompatibel dengan gradien per-sampel yang dibutuhkan
DP-SGD (Subbab 2.4.3): statistik *batch*-nya membuat gradien satu sampel
bergantung pada sampel lain dalam *batch* yang sama, sehingga konsep
"gradien per-sampel yang independen" tidak terdefinisi dengan bersih.

*Group Normalization* (GroupNorm) [n] menjadi pengganti yang mengatasi
kedua persoalan di atas sekaligus: alih-alih menormalisasi lintas *batch*,
GroupNorm membagi kanal suatu lapisan menjadi beberapa grup dan
menormalisasi **di dalam satu sampel saja** (lintas grup kanal, bukan
lintas sampel), sehingga tidak bergantung pada ukuran atau komposisi
*batch*, dan gradien per-sampel yang dibutuhkan DP-SGD terdefinisi dengan
bersih. Penelitian ini mengonversi seluruh **81 lapisan BatchNorm2d** pada
YOLOv11n menjadi GroupNorm -- audit arsitektur akhir mengonfirmasi 0 modul
BatchNorm tersisa dan 81 modul GroupNorm -- dengan jumlah grup pada tiap
lapisan dipilih sebagai pembagi terbesar dari jumlah kanal lapisan tersebut
yang tidak melebihi 32, mengikuti rekomendasi umum literatur GroupNorm.
Konversi ini diterapkan secara **seragam pada seluruh blok eksperimen (B1,
B2, E1, E2)**, agar selisih performa antar blok pada Bab 4 murni
mencerminkan efek federasi dan DP-SGD, bukan tercampur dengan efek
perbedaan arsitektur normalisasi.

## 2.6 *Explainable AI*: Grad-CAM++

### 2.6.1 *Class Activation Mapping* dan Grad-CAM++

Model *deep learning* -- termasuk detektor objek berbasis YOLO -- pada
umumnya bersifat *black-box*: keputusan klasifikasi/lokalisasinya sulit
ditelusuri secara langsung dari aktivasi internal jaringan. *Class
Activation Mapping* (CAM) dan variannya menjawab persoalan ini dengan
menghasilkan peta panas (*heatmap*) yang menunjukkan wilayah citra masukan
yang paling berkontribusi terhadap skor suatu kelas. **Grad-CAM++** [n]
memperluas Grad-CAM dengan memakai kombinasi berbobot dari turunan parsial
positif orde lebih tinggi pada peta fitur lapisan konvolusi terakhir
terhadap skor kelas target, menghasilkan lokalisasi yang lebih baik
dibanding Grad-CAM standar terutama ketika satu citra memuat beberapa
instans dari kelas yang sama -- relevan langsung bagi citra TBS yang sering
memuat beberapa tandan sekaligus. Penelitian ini menerapkan Grad-CAM++ pada
lapisan konvolusi terakhir blok `C3k2` (indeks tahap 22) sebelum kepala
`Detect` (Subbab 2.2.1), dengan resolusi masukan 960×960, seragam untuk
seluruh model yang dibandingkan (B2, E1, E2).

### 2.6.2 Metrik *Faithfulness*: *Average Drop* dan *Focus Retention Rate*

Visualisasi CAM semata tidak membuktikan bahwa wilayah yang disorot benar-
benar dipakai model untuk membuat keputusan (*faithfulness*); dua metrik
kuantitatif dipakai untuk mengukur properti ini:

1. ***Average Drop* (AD).** Mengukur seberapa besar keyakinan (*confidence*)
   model terhadap kelas target turun ketika wilayah dengan aktivasi CAM
   tertinggi (mis. 20% teratas) di-*occlude* (ditutup) dari citra masukan:
   $$AD = \frac{\max(0,\; Y_c - O_c)}{Y_c}, \tag{2.7}$$
   dengan $Y_c$ skor kelas $c$ pada citra asli dan $O_c$ skor kelas $c$
   setelah *occlusion*. AD tinggi mengindikasikan wilayah yang disorot
   benar-benar memuat bukti yang dipakai model. Metrik ini memiliki
   keterbatasan yang telah diketahui: metrik ini bersifat bias terhadap
   model dengan keyakinan dasar ($Y_c$) yang rendah dan rapuh, karena
   *occlusion* apapun -- bukan hanya pada wilayah relevan -- cenderung
   menjatuhkan keyakinan yang sudah rapuh tersebut mendekati nol, sehingga
   AD mendekati nilai maksimumnya (1,0) bukan karena penjelasannya lebih
   akurat (dibahas lebih lanjut pada Subbab 4.7.4).
2. ***Focus Retention Rate* (FRR).** Mengukur fraksi massa aktivasi CAM
   yang jatuh **di dalam** kotak *ground-truth* objek, versus yang bocor
   ke latar belakang atau konteks di luar kotak. FRR tinggi mengindikasikan
   perhatian model terkonsentrasi pada objek yang relevan, bukan menyebar
   ke elemen latar yang tidak relevan.

Kedua metrik ini dihitung secara *paired* (tersandingkan) pada sampel yang
identik lintas model yang dibandingkan (Subbab 3.9, 4.7), sehingga selisih
antar model dapat diinterpretasikan sebagai perbandingan langsung, bukan
dipengaruhi oleh perbedaan populasi sampel.

## 2.7 *Containerization* dan *Deployment* Model

Model yang sudah dilatih perlu dikemas menjadi layanan yang dapat diakses
dan diuji di luar lingkungan pelatihan untuk menunjukkan kelayakan
operasionalnya. **Docker** adalah platform *containerization* yang
mengemas aplikasi beserta seluruh dependensinya (pustaka, *runtime*,
konfigurasi) ke dalam satu *image* yang dapat dijalankan secara konsisten
di berbagai lingkungan (*host*), tanpa bergantung pada instalasi manual
pustaka pada mesin target. *Image* Docker didefinisikan lewat `Dockerfile`
yang menspesifikasikan sistem dasar (*base image*), langkah instalasi
dependensi, dan perintah yang dijalankan saat *container* aktif. Model dan
*checkpoint* yang berukuran besar umumnya tidak di-*bake* langsung ke dalam
*image*, melainkan dipasang (*mount*) sebagai *volume* terpisah dari *host*,
sehingga *image* aplikasi dapat diperbarui secara independen dari
*checkpoint* model yang dipakainya. Penelitian ini mendemonstrasikan
*deployment* model operasional (checkpoint B2, Subbab 4.9.1) sebagai
layanan inferensi berbasis Docker pada sebuah *Virtual Private Server*
(VPS), sebagai bukti kelayakan operasional model di luar lingkungan
pelatihan GPU (Subbab 4.9).

## 2.8 Penelitian Terkait dan Posisi Penelitian Ini

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
stabil di bawah heterogenitas klien [11], [15]. Penerapan DP-SGD pada
*deep learning* umum telah mapan [n], namun karakterisasi *trade-off*
privasi-utilitasnya secara tersandingkan penuh untuk detektor objek
federasi pada domain pertanian -- dilengkapi analisis interpretasi visual
(XAI) yang membandingkan model dengan dan tanpa DP secara langsung --
sejauh penelusuran yang dilakukan masih jarang dibahas bersamaan.

Kombinasi spesifik antara (i) deteksi kematangan TBS multi-kelas berbasis
YOLO11, (ii) pengaturan federasi Non-IID (partisi Dirichlet, K = 4 klien)
yang dievaluasi di bawah pembagian data bebas-kebocoran pada level kelompok
sumber, (iii) adopsi Group Normalization yang secara khusus dipilih untuk
menghilangkan ketergantungan statistik *batch* sekaligus mengaktifkan
kompatibilitas DP-SGD, (iv) karakterisasi *trade-off* privasi-utilitas
DP-SGD *full* versus *partial* yang tersandingkan penuh pada *split
held-out test* yang sama, dan (v) analisis XAI *matched-sample* antara
model dengan dan tanpa DP, sejauh penelusuran yang dilakukan belum
ditemukan dibahas bersamaan untuk tugas ini. Tabel 2.1 merangkum posisi
penelitian ini terhadap penelitian deteksi kematangan TBS sawit berbasis
YOLO yang paling relevan.

Tabel 2.1. Posisi penelitian ini terhadap penelitian sejenis

| Penelitian | Model | FL | DP | XAI | Pembagian bebas-kebocoran | Catatan |
|---|---|---|---|---|---|---|
| Lai dkk. [3] | YOLOv4 | Tidak | Tidak | Tidak | Tidak dilaporkan | Deteksi tandan matang *real-time* di perkebunan |
| Suharjito dkk. [4] | Model klasifikasi kematangan (varian *mobile*) | Tidak | Tidak | Tidak | Tidak dilaporkan | Target *deployment* perangkat *mobile* |
| Asrol dkk. [6] | YOLOv4 (modifikasi) | Tidak | Tidak | Tidak | Tidak dilaporkan | Sistem *grading* *real-time* via *smartphone* |
| Goh dkk. (tinjauan) [7] | Beragam | Mayoritas tidak | Tidak | Tidak | Jarang dibahas eksplisit | Tinjauan metode klasifikasi kematangan TBS |
| **Penelitian ini (B1/B2/E1/E2)** | YOLOv11n + GroupNorm | **Ya** (FedAvg, K = 4, Non-IID) | **Ya** (DP-SGD *full*/*partial*, akuntansi ε per klien) | **Ya** (Grad-CAM++ *matched-sample*) | **Ya**, diaudit eksplisit | Titik rujukan tersentral-versus-federasi, karakterisasi *trade-off* privasi-utilitas, dan validasi XAI tersandingkan; ditutup demonstrasi *deployment* |

Penelitian ini tidak mengklaim sebagai penerapan pertama *Federated
Learning* pada citra sawit, penggunaan pertama YOLO untuk klasifikasi
kematangan TBS, atau penerapan pertama DP-SGD pada visi komputer;
kontribusinya adalah karakterisasi empiris *trade-off* privasi-utilitas
dan interpretabilitas yang tersandingkan penuh (B1 vs B2 vs E1 vs E2) untuk
tugas, arsitektur, dan pengaturan federasi spesifik ini, dilengkapi
demonstrasi kelayakan operasionalnya.
