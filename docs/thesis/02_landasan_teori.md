<!--
CATATAN PENULISAN (hapus komentar ini sebelum submit):
Bab 2 dirombak total mengikuti kerangka FedX-Palm, dengan definisi
operasional AD/FRR dan detail arsitektural (mis. jumlah lapisan
BatchNorm, indeks lapisan target Grad-CAM++) disesuaikan persis
dengan implementasi kode pada repo ini (src/fedxpalm/). Rujukan
bernomor [n] adalah placeholder -- lengkapi dengan sitasi pustaka
aktual sebelum submit; lihat docs/thesis/NOTES_FOR_RACHMA.md.
-->

# CHAPTER 2 -- LANDASAN TEORI

Bab ini membahas dasar-dasar teori yang menopang penelitian, dimulai dari
domain aplikasi (kematangan TBS sawit), arsitektur detektor (YOLOv11),
kerangka pembelajaran terdistribusi (*Federated Learning*), jaminan privasi
formal (*Differential Privacy* dengan DP-SGD), model ancaman yang relevan,
modifikasi arsitektural yang dibutuhkan (BatchNorm ke GroupNorm dan pustaka
Opacus), metode penjelasan visual (Grad-CAM++), penelitian terkait, serta
konsep *containerization* untuk *deployment*.

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
*Underripe*, *Ripe*, *Overripe*, *Empty Bunch*, dan *Abnormal*. Kelas *Empty
Bunch* merepresentasikan tandan yang sebagian besar brondolannya sudah
lepas, sedangkan *Abnormal* mencakup tandan dengan kelainan pertumbuhan atau
penyakit. Pembedaan antarkelas terutama bertumpu pada isyarat visual berupa
warna kulit buah, jumlah brondolan yang lepas, dan tekstur permukaan tandan
[n]. Berdasarkan standar perkebunan, keenam kelas yang dipakai pada
penelitian ini (indeks 0-5 pada `data.yaml` dataset Roboflow berurutan
alfabetis: *Abnormal*, *Empty Bunch*, *Overripe*, *Ripe*, *Underripe*,
*Unripe* -- lihat `configs/dataset.yaml`) dijabarkan berikut ini menurut
urutan tingkat kematangan, bukan urutan indeks, agar lebih mudah dipahami.

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
terhadap kelelahan operator. Tingkat kesalahan klasifikasi manual dilaporkan
dapat mencapai 15-25% pada kondisi pencahayaan suboptimal [n]. Persoalan
konsistensi inilah yang membuka jalan bagi otomasi berbasis visi komputer
untuk perkebunan skala besar.

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
seperti pada generasi YOLO lebih awal. Struktur *backbone-neck-head* ini
konsisten di seluruh varian ukuran YOLOv11 (n/s/m/l/x) -- yang berbeda
hanyalah lebar (jumlah kanal) dan kedalaman (jumlah pengulangan blok) pada
tiap tahap, sehingga indeks tahap arsitektural (mis. tahap *backbone* ke-0
hingga ke-10, *neck* ke-11 hingga ke-22, *head* pada tahap ke-23) tetap sama
antar varian. Penelitian ini memakai varian **nano** (YOLOv11n, ~2,6 juta
parameter) karena mayoritas penelitian sejenis pada domain kematangan TBS
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
proporsional.

### 2.3.2 *Cross-silo* dan *Cross-device*

Literatur FL membedakan dua skenario deployment berdasarkan karakteristik
klien [n]. **Cross-device** melibatkan jutaan klien tak-andal (mis.
ponsel), masing-masing dengan data sangat sedikit dan konektivitas tidak
stabil. **Cross-silo** melibatkan sedikit klien (puluhan hingga ratusan)
yang relatif andal dan bertahan sepanjang pelatihan -- misalnya organisasi,
rumah sakit, atau, pada konteks penelitian ini, kebun/afdeling kelapa sawit
yang masing-masing memiliki infrastruktur komputasi sendiri. FedX-Palm
beroperasi pada rezim *cross-silo*: jumlah klien $K \in \{2,4,8,12,16\}$
merepresentasikan jumlah kebun atau afdeling yang berpartisipasi, bukan
perangkat individu.

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
$\alpha \to \infty$ mendekati pembagian IID seragam. Penelitian ini memakai
$\alpha = 0{,}5$, nilai moderat yang umum dipakai pada literatur FL untuk
mensimulasikan heterogenitas realistis tanpa membuat sebagian klien
kehabisan sampel kelas tertentu sama sekali.

## 2.4 *Differential Privacy* (DP)

*Differential Privacy* (DP) adalah kerangka matematis yang memberikan
jaminan privasi formal: keluaran suatu mekanisme (mis. model yang telah
dilatih) hampir tidak berubah probabilitasnya baik satu sampel data tertentu
disertakan dalam data latih atau tidak [n]. Berbeda dengan pendekatan
privasi heuristik (mis. anonimisasi), DP memberikan batas atas kuantitatif
atas kebocoran informasi.

### 2.4.1 Definisi (ε, δ)-DP

Suatu mekanisme *randomized* $\mathcal{M}$ dikatakan memenuhi $(\varepsilon,
\delta)$-*Differential Privacy* jika untuk setiap dua dataset $D, D'$ yang
bertetangga (berbeda tepat satu sampel), dan untuk setiap himpunan keluaran
$S$:

$$\Pr[\mathcal{M}(D) \in S] \le e^{\varepsilon}\Pr[\mathcal{M}(D') \in S] + \delta. \tag{2.4}$$

$\varepsilon$ (*privacy budget*) mengukur seberapa besar kemungkinan
keluaran boleh berbeda antara dua dataset bertetangga -- semakin kecil
$\varepsilon$, semakin ketat jaminan privasinya. $\delta$ adalah probabilitas
kecil kegagalan jaminan tersebut (pada penelitian ini ditetapkan $\delta =
10^{-5}$, mengikuti konvensi memilih $\delta \ll 1/N$ dengan $N$ ukuran
dataset).

### 2.4.2 *Gradient Clipping*

Karena gradien dari satu sampel data secara individual dapat memiliki norma
tak terbatas, DP-SGD terlebih dahulu **membatasi (*clip*) norma gradien
per-sampel** ke suatu ambang $C$:

$$\bar{g}_i = g_i \cdot \min\left(1, \frac{C}{\lVert g_i \rVert_2}\right), \tag{2.5}$$

dengan $g_i$ gradien mentah sampel ke-$i$. Langkah ini membatasi kontribusi
maksimum satu sampel terhadap pembaruan bobot, prasyarat agar penambahan
*noise* pada langkah berikutnya dapat memberikan jaminan privasi yang
terkalibrasi secara konsisten.

### 2.4.3 *Gaussian Mechanism*

Setelah *clipping*, *noise* Gaussian ditambahkan pada gradien yang sudah
dijumlahkan per-*batch*:

$$\tilde{g} = \frac{1}{B}\left(\sum_{i=1}^{B} \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 I)\right), \tag{2.6}$$

dengan $\sigma$ (*noise multiplier*) mengendalikan kekuatan privasi -- makin
besar $\sigma$, makin kuat jaminan privasi namun makin besar pula gangguan
terhadap sinyal gradien asli. Kombinasi *clipping* (2.5) dan mekanisme
Gaussian (2.6) inilah yang disebut **DP-SGD** [n], diterapkan *per-sampel*
pada penelitian ini (bukan pada level pembaruan bobot teragregasi klien --
lihat Subbab 2.4.5).

### 2.4.4 Akuntansi Privasi (*Privacy Accounting*)

Setiap langkah DP-SGD "membelanjakan" sebagian anggaran privasi; setelah $T$
langkah, anggaran total $\varepsilon$ harus diakumulasikan. Metode akumulasi
naif (penjumlahan langsung/*basic composition*) menghasilkan $\varepsilon$
yang jauh lebih besar (lebih longgar) dari kondisi sebenarnya. Penelitian
ini memakai ***accountant* PRV** (*Privacy Random Variable*) yang disediakan
Opacus, yang melacak distribusi kerugian privasi (*privacy loss random
variable*) secara numerik di sepanjang komposisi langkah, menghasilkan
estimasi $\varepsilon$ yang jauh lebih ketat (lebih kecil, lebih akurat)
dibanding *accountant* berbasis *moments accountant* generasi sebelumnya,
untuk kombinasi $(\sigma, q, T, \delta)$ tertentu -- dengan $q$ adalah
*sampling rate* (rasio ukuran *batch* terhadap ukuran dataset klien) dan
$T$ jumlah total langkah SGD yang dijalankan klien tersebut di sepanjang
seluruh ronde federasi.

### 2.4.5 Perbandingan Strategi Penyuntikan *Noise* pada *Federated Learning*

Terdapat dua jalur utama untuk mengintegrasikan DP ke dalam FL:

1. **DP-SGD per-sampel** (dipakai penelitian ini): *noise* disuntikkan pada
   gradien tiap sampel selama optimisasi lokal, sebelum agregasi FedAvg.
   Jaminan privasinya melekat pada tiap sampel data individual.
2. **DP-FedAvg tingkat-klien**: *noise* disuntikkan pada level pembaruan
   bobot (*delta*) tiap klien setelah pelatihan lokal selesai, sebelum
   dikirim ke *server*. Jaminan privasinya melekat pada partisipasi tiap
   klien (bukan tiap sampel).

Eksperimen pendahuluan pada penelitian ini menunjukkan bahwa penyuntikan
*noise* pada level pembaruan bobot teragregasi (opsi kedua) terhadap model
deteksi objek pra-latih cenderung menyebabkan *collapse* performa yang
mendadak begitu $\sigma$ melewati ambang tertentu -- kemungkinan karena
satu pembaruan bobot berdimensi tinggi jauh lebih sensitif terhadap
gangguan Gaussian berskala besar dibanding gradien per-langkah yang lebih
kecil dan lebih sering diperbarui. DP-SGD per-sampel, sebaliknya,
berinteraksi lebih halus dengan dinamika SGD karena *noise* disuntikkan
berulang kali dalam jumlah kecil di sepanjang optimisasi, bukan sekali dalam
jumlah besar di akhir. Atas dasar itu, DP-SGD per-sampel dipilih sebagai
mekanisme privasi utama pada penelitian ini, dengan DP-FedAvg tingkat-klien
disertakan sebagai pembanding pendahuluan pada Bab 4.

## 2.5 Model Ancaman pada *Federated Learning*

Meski FL menghindari pengumpulan data mentah, pertukaran parameter model
tetap membuka permukaan serangan. Dua ancaman inferensi yang paling relevan
bagi penelitian ini:

1. ***Membership Inference Attack* (MIA).** Penyerang mencoba menebak
   apakah suatu sampel data tertentu pernah dipakai dalam pelatihan model,
   dengan mengamati pola keluaran atau gradien model terhadap sampel
   tersebut [n]. Pada konteks FedX-Palm, MIA yang berhasil dapat
   menyingkap bahwa citra TBS tertentu (dan karenanya lokasi/waktu
   pengambilannya) berasal dari kebun tertentu.
2. ***Gradient Inversion Attack*** (kadang disebut *Model Inversion
   Attack*). Penyerang mencoba merekonstruksi kembali citra masukan asli
   dari gradien yang teramati, dengan mengoptimalkan citra sintetis
   sedemikian rupa hingga gradiennya menyerupai gradien yang disadap [n].
   Serangan ini terbukti efektif terutama pada *batch* berukuran kecil.

Selain dua ancaman inferensi di atas, jalur komunikasi *client-server* juga
rentan terhadap *Man-in-the-Middle Attack* (MitM) jika tidak diamankan
dengan *Transport Layer Security* (TLS); ancaman ini berada di luar cakupan
DP formal (yang melindungi *konten* pembaruan, bukan *transportnya*) dan
diasumsikan ditangani melalui praktik keamanan jaringan standar pada
*deployment* produksi. DP-SGD per-sampel, sebagaimana dijabarkan pada
Subbab 2.4, dirancang khusus untuk meredam MIA dan *gradient inversion*
dengan membatasi kontribusi informasi tiap sampel individual pada gradien
yang dipertukarkan.

## 2.6 BatchNorm versus GroupNorm

*Batch Normalization* (BatchNorm) menormalisasi aktivasi suatu lapisan
dengan menghitung rata-rata dan varians **lintas seluruh sampel dalam satu
*batch***. Karakteristik inilah yang membuat BatchNorm secara fundamental
tidak kompatibel dengan gradien per-sampel yang dibutuhkan DP-SGD:
statistik *batch*-nya membuat gradien satu sampel bergantung pada sampel
lain dalam *batch* yang sama, sehingga konsep "gradien per-sampel yang
independen" -- prasyarat *clipping* pada Persamaan (2.5) -- tidak
terdefinisi dengan bersih.

*Group Normalization* (GroupNorm) [n] menjadi pengganti standar pada
pelatihan DP: alih-alih menormalisasi lintas *batch*, GroupNorm membagi
kanal suatu lapisan menjadi beberapa grup dan menormalisasi **di dalam satu
sampel saja** (lintas grup kanal, bukan lintas sampel). Karena
komputasinya sepenuhnya independen per sampel, gradien per-sampel GroupNorm
terdefinisi dengan baik dan lolos validasi Opacus. Penelitian ini
mengonversi seluruh **81 lapisan BatchNorm2d** pada YOLOv11n menjadi
GroupNorm sebagai prasyarat arsitektural sebelum menjalankan DP-SGD --
jumlah grup pada tiap lapisan dipilih sebagai pembagi terbesar dari jumlah
kanal lapisan tersebut yang tidak melebihi 32, mengikuti rekomendasi umum
literatur GroupNorm. Konversi ini diterapkan secara **seragam pada seluruh
blok eksperimen** (B1, B2, E1, E2 -- lihat Bab 3), bukan hanya pada
eksperimen berDP, agar perbandingan performa antar blok tidak bercampur
dengan efek perbedaan arsitektur normalisasi.

## 2.7 Opacus

Opacus [n] adalah *library* PyTorch yang dikembangkan Meta AI untuk
melatih model dengan DP-SGD secara efisien. Tiga komponen utamanya:

1. **`GradSampleModule`** membungkus model dan memasang *hook* pada
   *forward* dan *backward pass* tiap lapisan berparameter, sehingga
   gradien per-sampel (bukan hanya gradien rata-rata *batch*) dapat
   dihitung tanpa perlu menjalankan *forward-backward* terpisah untuk tiap
   sampel satu per satu -- pendekatan ini jauh lebih efisien dibanding
   implementasi DP-SGD naif.
2. **`DPOptimizer`** membungkus *optimizer* PyTorch standar (pada
   penelitian ini, SGD momentum -- lihat Subbab 3.6.1 untuk alasan
   pemilihannya) untuk menerapkan *clipping* per-sampel (2.5) dan
   penambahan *noise* Gaussian (2.6) secara otomatis sebelum setiap langkah
   `step()`.
3. **`PrivacyEngine`** adalah antarmuka tingkat tinggi yang mengikat model,
   *optimizer*, dan `DataLoader` menjadi satu kesatuan yang taat-DP
   (`make_private`), serta menyediakan *accountant* PRV untuk menghitung
   $\varepsilon$ kumulatif kapan saja selama atau setelah pelatihan.

Opacus mensyaratkan setiap lapisan model didukung oleh mekanisme
`GradSampleModule`-nya (Conv2d, Linear, GroupNorm, dan beberapa jenis
lapisan lain) dan tidak mengandung operasi yang mencampur informasi
antar-sampel (seperti BatchNorm, dibahas pada Subbab 2.6). Selain itu,
`PrivacyEngine.make_private` mengganti strategi *sampling* `DataLoader`
menjadi *Poisson sampling* -- setiap sampel disertakan dalam suatu *batch*
secara independen dengan probabilitas tetap $q$, bukan pembagian *batch*
yang deterministik -- karena jaminan privasi formal DP-SGD (dan akurasi
*accountant* PRV) bergantung pada properti acak *Poisson sampling* ini.

## 2.8 *Explainable AI*: Grad-CAM dan Grad-CAM++

### 2.8.1 Grad-CAM dan Grad-CAM++

*Gradient-weighted Class Activation Mapping* (Grad-CAM) [n] menghasilkan
peta panas (*heatmap*) yang menyoroti wilayah citra yang paling berkontribusi
terhadap skor kelas tertentu, dengan menimbang peta fitur pada suatu lapisan
konvolusi target menggunakan gradien skor kelas terhadap peta fitur
tersebut. Grad-CAM++ [n] memperbaiki Grad-CAM dengan memakai kombinasi
turunan orde lebih tinggi (melalui pendekatan berbasis turunan pertama yang
dipangkatkan) sebagai bobot piksel-per-piksel pada tiap kanal peta fitur,
alih-alih rata-rata global gradien seperti Grad-CAM asli -- hasilnya adalah
lokalisasi yang lebih presisi terutama ketika suatu kelas muncul di beberapa
wilayah citra sekaligus, situasi yang umum pada citra TBS dengan banyak
brondolan.

Secara operasional pada penelitian ini, Grad-CAM++ dipasang pada **lapisan
neck bersama terakhir** (tahap indeks ke-22 pada graf arsitektur YOLOv11 --
blok C3k2 terakhir sebelum *head* Detect, lihat Subbab 2.2.1), karena
lapisan ini adalah representasi fitur skala-terdalam terakhir yang masih
diteruskan bersama ke seluruh cabang *head* deteksi, sehingga peta panas
yang dihasilkan merefleksikan fitur yang benar-benar dipakai keputusan
model, bukan fitur perantara yang idiosinkratik pada satu skala deteksi
saja. Untuk satu deteksi (kelas $c$, anchor tertentu), bobot Grad-CAM++
$\alpha_k^c$ untuk kanal fitur ke-$k$ dihitung dari gradien skor kelas
$S^c$ terhadap peta fitur $A^k$:

$$\alpha_k^c = \frac{\left(\partial S^c / \partial A^k\right)^2}{2\left(\partial S^c/\partial A^k\right)^2 + \sum_{a,b} A^k_{a,b}\left(\partial S^c/\partial A^k\right)^3}, \tag{2.7}$$

dan peta lokalisasi akhir:

$$L^c_{Grad-CAM++} = \text{ReLU}\left(\sum_k \left(\sum_{a,b}\alpha^c_{k,a,b}\cdot\text{ReLU}\left(\partial S^c/\partial A^k_{a,b}\right)\right) A^k\right). \tag{2.8}$$

$S^c$ diambil dari skor klasifikasi mentah (*pre-sigmoid logit*) *head*
Detect pada *anchor* dengan skor tertinggi untuk kelas $c$ yang dievaluasi,
konsisten dengan praktik umum Grad-CAM pada model klasifikasi tunggal-logit
yang diadaptasi ke detektor satu-tahap.

### 2.8.2 *Average Drop* (AD)

*Average Drop* mengukur *faithfulness* penjelasan melalui skema berbasis
oklusi: jika wilayah yang disorot peta panas benar-benar menjadi dasar
keputusan model, menutup (meng-oklusi) wilayah tersebut seharusnya
menurunkan keyakinan model secara nyata. Secara operasional pada penelitian
ini: wilayah citra dengan nilai aktivasi Grad-CAM++ pada persentil teratas
(20% piksel teratas) diganti dengan nilai rata-rata kanal citra tersebut
(*neutral fill*), lalu skor keyakinan kelas $c$ dihitung ulang pada citra
yang sudah dioklusi ($O_c$) dan dibandingkan dengan skor pada citra asli
($Y_c$):

$$\text{AD} = \frac{\max(0,\, Y_c - O_c)}{Y_c}. \tag{2.9}$$

Nilai AD dilaporkan sebagai rata-rata di seluruh sampel evaluasi, per kelas
maupun secara global. AD mendekati 0 mengindikasikan wilayah yang disorot
penjelasan tidak benar-benar dipakai model (penjelasan tidak *faithful*);
AD yang lebih besar mengindikasikan wilayah tersebut memang menjadi sumber
evidensi utama keputusan model.

### 2.8.3 *Focus Retention Rate* (FRR)

*Focus Retention Rate* mengukur seberapa besar proporsi "massa" aktivasi
peta panas yang terkonsentrasi di dalam kotak *ground-truth* objek,
dibandingkan bocor ke wilayah latar/konteks di luarnya:

$$\text{FRR} = \frac{\sum_{(x,y) \in \text{box}_{gt}} L^c_{Grad-CAM++}(x,y)}{\sum_{(x,y) \in \text{citra}} L^c_{Grad-CAM++}(x,y)}. \tag{2.10}$$

FRR yang tinggi mengindikasikan model "memusatkan perhatian" pada objek TBS
itu sendiri (warna kulit buah, tekstur brondolan) alih-alih pada latar
kebun atau artefak citra di sekelilingnya. FRR dihitung per instans
*ground-truth*, kemudian dirata-ratakan per kelas dan secara global,
sejalan dengan AD.

## 2.9 *Containerization* untuk *Deployment* (Docker)

Docker adalah platform *containerization* yang mengemas aplikasi beserta
seluruh dependensinya (interpreter Python, *library*, bobot model) ke dalam
satu *image* yang dapat dijalankan secara konsisten di lingkungan komputasi
manapun, tanpa persoalan "berjalan di mesin saya tapi tidak di mesin lain".
Pada penelitian ini, Docker berperan sebagai **cetak biru (*blueprint*)
arsitektur *deployment***: model YOLOv11n hasil federasi (format `.pt`)
dikemas bersama layanan inferensi berbasis Flask dan modul Grad-CAM++ ke
dalam satu *image*, dijalankan sebagai *container* CPU-*only* pada sebuah
*Virtual Private Server* (VPS). Pemilihan CPU-*only* untuk *deployment* --
berlawanan dengan GPU yang dipakai pada tahap pelatihan -- secara sengaja
menguji portabilitas praktis model: sebuah kerangka yang hasil akhirnya
hanya bisa dijalankan pada perangkat keras khusus kurang relevan bagi
perkebunan skala kecil-menengah yang umumnya tidak memiliki infrastruktur
GPU produksi.

## 2.10 *State of the Art*

<!-- Isi tabel di bawah mengikuti gaya Tabel 2.1 pada kerangka FedX-Palm
generasi pertama. Sesuaikan/tambahkan baris berdasarkan tinjauan pustaka
lengkap yang dilakukan penulis; baris berikut disusun berdasarkan
penelusuran web singkat selama penulisan ulang bab ini dan PERLU
diverifikasi terhadap makalah aslinya sebelum disitasi secara formal. -->

Tabel 2.1 merangkum posisi penelitian ini terhadap penelitian deteksi
kematangan TBS sawit berbasis YOLO yang paling relevan.

| Penelitian | Model | FL | DP | XAI | Catatan |
|---|---|---|---|---|---|
| Anniswa dkk. [n] | YOLOv11 (varian belum terverifikasi) | Tidak | Tidak | Tidak | Optimisasi inferensi CPU via ONNX Runtime; mAP@0.5 dilaporkan 90,2% |
| [n] (Jurnal Sisfokom) | YOLOv11-nano | Tidak | Tidak | Tidak | *Class-Balanced Focal Loss* untuk kelas minoritas; mAP@0.5 0,783 |
| [n] (JAIC) | YOLO12S | Tidak | Tidak | Tidak | Deteksi kematangan berbasis YOLOv12S |
| Perbandingan YOLOv5 n/s/m [n] | YOLOv5m (terbaik) | Tidak | Tidak | Tidak | mAP@0.5:0.95 0,842 pada varian *medium* |
| **FedX-Palm v2 (penelitian ini)** | YOLOv11n | **Ya** (FedAvg, K=2..16) | **Ya** (DP-SGD per-sampel, *full* & *partial*) | **Ya** (Grad-CAM++, AD & FRR) | Satu-satunya yang mengintegrasikan ketiga komponen sekaligus pada domain TBS sawit |

Sejauh penelusuran yang dilakukan, belum dijumpai penelitian deteksi
kematangan TBS sawit berbasis YOLOv11 yang secara eksplisit mengintegrasikan
*Federated Learning*, *Differential Privacy* formal berbasis DP-SGD, dan
*Explainable AI* sekaligus dalam satu kerangka evaluasi -- inilah celah
riset yang diisi oleh FedX-Palm.
