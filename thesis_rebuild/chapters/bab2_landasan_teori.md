# BAB 2 LANDASAN TEORI

Bab ini menguraikan landasan teoretis yang menopang penelitian, mulai dari
domain aplikasi (kematangan tandan buah segar kelapa sawit), arsitektur
deteksi objek yang digunakan (YOLOv11), kerangka pembelajaran terdistribusi
(*Federated Learning*), mekanisme jaminan privasi formal (*Differential
Privacy* melalui DP-SGD), model ancaman yang relevan, modifikasi arsitektural
yang diperlukan agar privasi per-sampel dapat diterapkan (GroupNorm dan
pustaka Opacus), metode penjelasan model (*Explainable AI*), serta konsep
*containerization* untuk deployment.

## 2.1 Kelapa Sawit dan Kematangan Tandan Buah Segar

Kelapa sawit (*Elaeis guineensis*) merupakan komoditas penghasil minyak
nabati dengan produktivitas per hektar tertinggi dibanding tanaman minyak
lain. Buah sawit tumbuh dalam tandan (Tandan Buah Segar/TBS) yang masing-masing
dapat memuat ribuan brondolan. Kualitas dan rendemen minyak yang diekstraksi
sangat ditentukan oleh tingkat kematangan TBS pada saat panen [Junos et al.,
2022]. Tandan yang dipanen terlalu dini (*unripe*/*underripe*) memiliki
kandungan minyak rendah karena pembentukan minyak dalam mesokarp belum optimal,
sedangkan tandan yang terlambat dipanen (*overripe*) mengalami hidrolisis
trigliserida yang meningkatkan kadar Asam Lemak Bebas (*Free Fatty Acid*, FFA)
dan menurunkan mutu *Crude Palm Oil* (CPO).

Standar industri mengklasifikasikan kematangan TBS ke dalam beberapa kelas
yang relevan secara komersial. Penelitian ini mengikuti taksonomi enam kelas:
*Unripe*, *Underripe*, *Ripe*, *Overripe*, *Empty Bunch*, dan *Abnormal*.
Kelas *Empty Bunch* merepresentasikan tandan yang telah kehilangan sebagian
besar brondolan, sementara *Abnormal* mencakup tandan dengan kelainan
pertumbuhan atau penyakit. Pembedaan antarkelas terutama bertumpu pada isyarat
visual: warna kulit buah (dari hitam-keunguan pada *unripe* menjadi
oranye-kemerahan pada *ripe*), jumlah brondolan yang lepas, serta tekstur
permukaan tandan [Junos et al., 2022].

Dalam praktik tradisional, penilaian kematangan dilakukan secara manual oleh
mandor panen berdasarkan pengamatan visual dan jumlah brondolan jatuh.
Pendekatan ini bersifat subjektif, bergantung pada pengalaman individu, dan
rentan terhadap inkonsistensi serta kelelahan operator; tingkat kesalahan
klasifikasi manual dilaporkan dapat mencapai 15–25% pada kondisi pencahayaan
suboptimal [Sabri et al., 2017]. Keterbatasan ini mendorong penerapan visi
komputer untuk mengotomasi dan menyeragamkan keputusan panen. Sejumlah
penelitian terdahulu telah mengeksplorasi klasifikasi dan deteksi kematangan
sawit menggunakan *convolutional neural network* (CNN) maupun arsitektur
deteksi objek, namun mayoritas mengasumsikan pelatihan terpusat yang
mengharuskan pengumpulan seluruh citra ke satu lokasi — asumsi yang
bertentangan dengan kebutuhan kerahasiaan data operasional perkebunan yang
menjadi motivasi penelitian ini.

## 2.2 Deteksi Objek dan YOLOv11

Deteksi objek (*object detection*) adalah tugas visi komputer yang
mengharuskan model tidak hanya mengklasifikasikan objek tetapi juga melokalisasi
posisinya melalui *bounding box*. Berbeda dengan klasifikasi citra yang
menghasilkan satu label per gambar, deteksi objek menghasilkan himpunan
(kelas, kotak, skor) untuk setiap instans objek. Dalam konteks TBS, hal ini
memungkinkan deteksi beberapa tandan dengan tingkat kematangan berbeda dalam
satu bingkai kamera.

Keluarga model YOLO (*You Only Look Once*) memformulasikan deteksi sebagai
masalah regresi tunggal yang dipetakan langsung dari piksel citra ke koordinat
kotak dan probabilitas kelas dalam sekali jalan jaringan (*single-stage*),
berbeda dengan pendekatan dua tahap seperti Faster R-CNN. Karakteristik ini
menjadikan YOLO unggul dalam kecepatan inferensi sehingga cocok untuk aplikasi
*real-time* dan *edge* [Redmon et al., 2016]. Sejak versi pertama, arsitektur
YOLO terus berevolusi: pengenalan *anchor boxes*, *feature pyramid network*,
*path aggregation*, hingga desain *anchor-free* pada generasi terbaru.

YOLOv11 [Khanam & Hussain, 2024] merupakan iterasi mutakhir yang menyempurnakan
keseimbangan akurasi-efisiensi. Arsitekturnya terdiri atas tiga komponen utama:
(1) *backbone* untuk ekstraksi fitur berhierarki, yang pada YOLOv11 memanfaatkan
blok C3k2 (varian *cross-stage partial* yang efisien) dan modul SPPF (*Spatial
Pyramid Pooling - Fast*); (2) *neck* yang mengagregasi fitur multi-skala,
diperkuat modul atensi C2PSA (*Convolutional block with Parallel Spatial
Attention*); dan (3) *head* yang memprediksi kotak dan kelas secara
*anchor-free*. Varian *nano* (YOLOv11n) yang digunakan dalam penelitian ini
hanya memiliki sekitar 2,6 juta parameter, menjadikannya kandidat ideal untuk
deployment pada perangkat dengan sumber daya terbatas.

Fungsi kerugian (*loss*) YOLOv11 menggabungkan tiga komponen. Untuk regresi
kotak digunakan **CIoU** (*Complete Intersection over Union*) yang
memperhitungkan tumpang-tindih area, jarak titik pusat, serta konsistensi rasio
aspek antara kotak prediksi dan kotak acuan. Untuk lokalisasi yang lebih
presisi digunakan **Distribution Focal Loss** (DFL) yang memodelkan koordinat
kotak sebagai distribusi diskret alih-alih nilai tunggal. Komponen klasifikasi
menggunakan *binary cross-entropy* pada setiap kelas. Pemilihan YOLOv11n dalam
penelitian ini didasari pertimbangan: bobotnya yang ringan memudahkan
pertukaran parameter antarklien dalam *Federated Learning*, dan jumlah
parameternya yang kecil meringankan beban *noise* DP-SGD (lihat 2.4).

## 2.3 Federated Learning

*Federated Learning* (FL) adalah paradigma pelatihan model di mana sejumlah
entitas (klien) berkolaborasi melatih model bersama tanpa memindahkan data
mentah keluar dari batas masing-masing entitas [McMahan et al., 2017]. Alih-alih
mengirim data ke server, setiap klien melatih model pada data lokalnya dan hanya
mengirimkan pembaruan parameter ke server agregator. Paradigma ini secara
arsitektural menjawab kendala kerahasiaan data: citra perkebunan yang bersifat
rahasia dagang tidak pernah berpindah lokasi.

### 2.3.1 Algoritma FedAvg

Algoritma kanonik FL adalah **Federated Averaging** (FedAvg) [McMahan et al.,
2017]. Pada setiap ronde komunikasi, server mendistribusikan parameter global
$w_t$ kepada klien terpilih. Setiap klien $k$ melakukan beberapa langkah
*stochastic gradient descent* pada data lokalnya, menghasilkan parameter lokal
$w_t^k$. Server kemudian mengagregasi pembaruan tersebut menjadi parameter
global baru melalui rata-rata berbobot proporsional terhadap ukuran data lokal:

$$
w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} \, w_t^k,
$$

dengan $n_k$ jumlah sampel klien $k$ dan $n = \sum_k n_k$. Proses diulang
hingga konvergen. Pembobotan berdasarkan $n_k$ memastikan klien dengan data
lebih banyak memberi kontribusi lebih besar terhadap model global.

### 2.3.2 Cross-silo dan Cross-device

Kairouz et al. [2021] membedakan dua rezim FL. Pada **cross-device FL**, peserta
berjumlah sangat banyak (jutaan), masing-masing memiliki sedikit data, dan
ketersediaannya tidak stabil (mis. ponsel pengguna). Pada **cross-silo FL**,
peserta berjumlah sedikit (puluhan), masing-masing memiliki data dalam jumlah
besar, dan ketersediaannya stabil — misalnya beberapa perusahaan atau institusi
yang berkolaborasi. Skenario perkebunan sawit dalam penelitian ini termasuk
*cross-silo*: sejumlah kecil perkebunan, masing-masing dengan koleksi citra
besar, berkolaborasi melatih model deteksi kematangan bersama.

### 2.3.3 Heterogenitas Data Non-IID dan Partisi Dirichlet

Tantangan mendasar FL adalah distribusi data antarklien yang tidak *independent
and identically distributed* (non-IID). Pada perkebunan, sebaran kelas
kematangan dapat sangat berbeda antarlokasi akibat perbedaan varietas, musim
panen, dan praktik pengelolaan. Heterogenitas ini memperlambat konvergensi dan
dapat menurunkan akurasi model global karena pembaruan lokal saling
"bertarik-tarikan" ke arah optimum lokal yang berbeda (*client drift*).

Untuk mensimulasikan kondisi non-IID secara terkontrol, penelitian FL umumnya
mempartisi dataset menggunakan **distribusi Dirichlet** [Hsu et al., 2019].
Untuk setiap kelas, proporsi sampel yang dialokasikan ke tiap klien ditarik dari
distribusi Dirichlet $\text{Dir}(\alpha)$. Parameter konsentrasi $\alpha$
mengendalikan derajat heterogenitas: nilai $\alpha$ kecil (mis. 0,1)
menghasilkan partisi sangat tidak seimbang (tiap klien didominasi sedikit
kelas), sedangkan $\alpha$ besar mendekati distribusi seragam (IID). Mekanisme
inilah yang dipakai dalam penelitian untuk membentuk partisi $K$ klien dengan
heterogenitas realistis pada berbagai nilai $K$.

## 2.4 Differential Privacy dan DP-SGD

Meskipun FL tidak memindahkan data mentah, pembaruan parameter yang
dipertukarkan masih dapat membocorkan informasi tentang data pelatihan
(lihat 2.5). **Differential Privacy** (DP) memberikan jaminan privasi formal
yang dapat dikuantifikasi terhadap kebocoran semacam ini.

### 2.4.1 Definisi Formal (ε, δ)-DP

Sebuah mekanisme acak $\mathcal{M}$ memenuhi $(\varepsilon, \delta)$-*differential
privacy* jika untuk setiap pasangan *dataset* bertetangga $D$ dan $D'$ (berbeda
satu sampel) dan setiap himpunan keluaran $S$ berlaku [Dwork & Roth, 2014]:

$$
\Pr[\mathcal{M}(D) \in S] \le e^{\varepsilon} \cdot \Pr[\mathcal{M}(D') \in S] + \delta.
$$

Parameter $\varepsilon$ (*privacy budget*) membatasi seberapa besar keluaran
mekanisme dapat berubah akibat penambahan/penghapusan satu sampel: $\varepsilon$
kecil berarti privasi kuat. Parameter $\delta$ menyatakan probabilitas
pelanggaran jaminan tersebut; konvensi umum memilih $\delta < 1/N$ dengan $N$
ukuran dataset, agar lebih kecil daripada peluang membocorkan satu sampel secara
kebetulan. Interpretasi intuitif: seorang pengamat yang melihat keluaran model
nyaris tidak dapat membedakan apakah suatu sampel tertentu ikut serta dalam
pelatihan atau tidak.

### 2.4.2 Algoritma DP-SGD

**DP-SGD** [Abadi et al., 2016] menanamkan jaminan DP ke dalam proses pelatihan
dengan memodifikasi langkah *gradient descent*. Pada setiap langkah, untuk
sebuah *minibatch*, algoritma melakukan tiga operasi kunci:

1. **Komputasi gradien per-sampel.** Berbeda dengan SGD biasa yang langsung
   merata-ratakan gradien seluruh batch, DP-SGD menghitung gradien $g_i$
   untuk **setiap** sampel $i$ secara terpisah.

2. **Pemotongan norma per-sampel (*clipping*).** Setiap gradien per-sampel
   dipotong agar norma-$\ell_2$-nya tidak melebihi ambang $C$:
   $$
   \bar{g}_i = g_i \big/ \max\!\left(1, \tfrac{\lVert g_i \rVert_2}{C}\right).
   $$
   Langkah ini membatasi sensitivitas — seberapa besar pengaruh satu sampel
   terhadap pembaruan.

3. **Penambahan *noise* Gaussian.** *Noise* acak ditambahkan pada jumlah
   gradien terpotong sebelum dirata-ratakan:
   $$
   \tilde{g} = \frac{1}{B}\left( \sum_{i} \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I}) \right),
   $$
   dengan $B$ ukuran batch dan $\sigma$ pengali *noise* (*noise multiplier*).
   Parameter $\tilde{g}$ inilah yang dipakai untuk memperbarui bobot.

Besarnya $\sigma$ mengendalikan trade-off privasi-utilitas: $\sigma$ besar
memberi privasi lebih kuat (ε lebih kecil) namun menambah *noise* yang dapat
menurunkan akurasi.

### 2.4.3 Privacy Accounting

*Privacy budget* $\varepsilon$ terakumulasi sepanjang banyak langkah pelatihan.
Menghitung akumulasi ini secara ketat memerlukan *privacy accountant*.
**Rényi Differential Privacy** (RDP) [Mironov, 2017] memberikan kerangka
komposisi yang lebih erat daripada teorema komposisi dasar, dengan memanfaatkan
divergensi Rényi; hasilnya kemudian dikonversi ke $(\varepsilon, \delta)$.
Konsep terkait **zero-Concentrated DP** (zCDP) [Bun & Steinke, 2016] menawarkan
analisis komposisi yang serupa eratnya. Akuntan modern berbasis **Privacy Loss
Random Variable** (PRV) memberikan batas yang lebih erat lagi dan menjadi
*default* pada pustaka Opacus versi terkini (lihat 2.7). Penelitian ini
melaporkan $\varepsilon$ dari akuntan PRV, bukan dari rumus tertutup.

Faktor penting yang menurunkan $\varepsilon$ adalah **amplifikasi privasi melalui
*subsampling***: karena setiap langkah hanya melihat sebagian acak data
(*minibatch*), informasi yang bocor per langkah berkurang secara proporsional
terhadap rasio pengambilan sampel $B/N$. Hal ini menjelaskan mengapa ukuran
batch dan jumlah sampel per klien memengaruhi anggaran privasi akhir — implikasi
yang relevan langsung terhadap pemilihan jumlah klien $K$ (lihat 2.4.4).

### 2.4.4 Mengapa Gradien Per-sampel Lebih Ramah Optimizer

Terdapat dua tempat untuk menanamkan *noise* DP dalam FL. Pendekatan pertama,
**DP level-klien** (mis. DP-FedAvg), menambahkan *noise* pada pembaruan
*agregat* tiap klien sebelum dikirim ke server. Pendekatan kedua, **DP-SGD
per-sampel** yang dipakai penelitian ini, menambahkan *noise* terkalibrasi pada
gradien di dalam proses optimisasi lokal.

Perbedaan ini berkonsekuensi besar terhadap stabilitas pelatihan. Pada DP
level-klien, *noise* dikenakan pada vektor berdimensi tinggi (seluruh bobot
model) sekali per ronde; bila skala *noise* tinggi, pembaruan agregat dapat
"meledak" dan menyebabkan *collapse* — model gagal belajar sama sekali. Sebaliknya,
DP-SGD per-sampel menyebar *noise* ke banyak langkah kecil yang masing-masing
sudah ternormalisasi oleh *clipping*, sehingga optimizer dapat tetap mengikuti
arah gradien rata-rata yang bermakna. Dengan kata lain, *noise* per-sampel
"larut" ke dalam dinamika SGD yang sudah stokastik secara alami, menghasilkan
degradasi yang lebih *gradual* dibanding *cliff* yang teramati pada DP
level-klien.

Namun, sifat per-sampel ini juga memunculkan ketergantungan pada jumlah sampel
per klien. Ketika jumlah klien $K$ bertambah pada dataset berukuran tetap,
sampel per klien menyusut; pada batas tertentu, *noise* mulai mendominasi sinyal
gradien lokal sebelum agregasi global sempat memperbaikinya. Konsekuensinya,
arah pengaruh $K$ pada DP-SGD per-sampel **berkebalikan** dengan intuisi DP
level-klien: $K$ besar justru cenderung memperburuk utilitas. Ini menjadi salah
satu hipotesis yang diuji secara empiris dalam penelitian.

## 2.5 Model Ancaman pada Federated Learning

Jaminan DP menjadi relevan justru karena pertukaran parameter dalam FL membuka
sejumlah celah serangan inferensi. Tiga ancaman utama menjadi acuan.

**Membership Inference Attack** (MIA) [Shokri et al., 2017] berupaya menentukan
apakah suatu sampel tertentu termasuk dalam data pelatihan, dengan mengeksploitasi
perbedaan perilaku model (mis. tingkat keyakinan) terhadap sampel yang pernah
dilihat versus yang belum. Dalam konteks perkebunan, keberhasilan MIA dapat
mengonfirmasi bahwa citra dari blok kebun tertentu ikut melatih model — bocornya
informasi kepemilikan/operasional.

**Model Inversion** [Fredrikson et al., 2015] berupaya merekonstruksi
karakteristik data pelatihan dari akses terhadap model, berpotensi memulihkan
fitur citra representatif suatu kelas. **Gradient Leakage** [Zhu et al., 2019]
menunjukkan bahwa pada kondisi tertentu, citra pelatihan beserta labelnya dapat
direkonstruksi nyaris sempurna hanya dari gradien yang dipertukarkan — ancaman
yang sangat relevan bagi FL karena gradien/pembaruan parameter adalah objek yang
justru dikomunikasikan.

DP-SGD **membatasi** celah-celah ini secara formal (bukan menutupnya sepenuhnya):
dengan membatasi sensitivitas (melalui *clipping*) dan mengaburkan kontribusi tiap
sampel (melalui *noise*), keluaran model menjadi nyaris tak terbedakan terhadap
ada-tidaknya satu sampel dalam batas anggaran $\varepsilon$.
Akibatnya, keunggulan penyerang MIA dibatasi secara matematis oleh $\varepsilon$,
dan rekonstruksi via *gradient leakage* maupun *model inversion* menjadi tidak
andal karena gradien yang dipertukarkan telah ternormalisasi dan dikaburkan.

## 2.6 BatchNorm versus GroupNorm

Penerapan DP-SGD per-sampel memunculkan kendala arsitektural yang sering
luput diperhatikan: lapisan normalisasi. **Batch Normalization** (BatchNorm)
menormalkan aktivasi menggunakan statistik (rata-rata dan varians) yang dihitung
*lintas sampel dalam satu batch*. Akibatnya, keluaran untuk satu sampel
bergantung pada sampel-sampel lain di batch yang sama. Ketergantungan
antar-sampel ini **bertentangan secara fundamental** dengan DP-SGD, yang
mensyaratkan gradien tiap sampel dapat dihitung dan dipotong secara independen —
sensitivitas per-sampel menjadi tak terdefinisi bila aktivasi satu sampel
mencampur informasi sampel lain.

**Group Normalization** (GroupNorm) [Wu & He, 2018] menyelesaikan masalah ini
dengan menormalkan aktivasi *di dalam tiap sampel*: kanal fitur dibagi menjadi
sejumlah grup, dan normalisasi dihitung per-grup pada satu sampel saja, tanpa
melibatkan sampel lain. Karena komputasinya independen antar-sampel, GroupNorm
**kompatibel** dengan DP-SGD. Sebagai bonus, GroupNorm tidak bergantung pada
ukuran batch sehingga stabil bahkan pada batch kecil — kondisi yang lazim pada
pelatihan terbatas memori. Oleh karena itu, penelitian ini mengganti seluruh
lapisan BatchNorm pada YOLOv11n dengan GroupNorm sebagai prasyarat (bukan
eksperimen terpisah) agar Opacus dapat diterapkan.

## 2.7 Opacus

**Opacus** adalah pustaka resmi untuk melatih model PyTorch dengan DP-SGD secara
efisien. Komponen intinya adalah komputasi **gradien per-sampel** yang efisien
melalui mekanisme *hook* dan, pada versi mutakhir, *backend* berbasis functorch
(`GradSampleModule`) — menghindari kebutuhan menghitung gradien per sampel satu
per satu yang mahal. Opacus membungkus model, optimizer, dan *data loader* ke
dalam sebuah `PrivacyEngine` yang secara otomatis melakukan *clipping* per-sampel,
penambahan *noise* Gaussian, serta pelacakan anggaran privasi $\varepsilon$
melalui *accountant* (PRV sebagai *default* pada versi terkini).

Opacus menegakkan sejumlah batasan melalui `ModuleValidator`. Model harus bebas
dari lapisan yang melanggar asumsi per-sampel — terutama **BatchNorm** (lihat
2.6) dan operasi *in-place* tertentu yang merusak pelacakan gradien per-sampel.
Validator ini sekaligus menyediakan utilitas untuk mengganti modul yang tidak
kompatibel (mis. BatchNorm → GroupNorm) secara otomatis. Dalam penelitian ini,
Opacus 1.5.4 dipakai untuk membungkus YOLOv11n-GN, dengan ambang *clipping* $C$
dan pengali *noise* $\sigma$ sebagai parameter eksperimen utama.

## 2.8 Explainable AI: Grad-CAM dan Grad-CAM++

Agar keputusan model dapat dipercaya oleh praktisi lapangan, model perlu
disertai penjelasan visual mengenai dasar prediksinya. **Grad-CAM** (*Gradient-
weighted Class Activation Mapping*) [Selvaraju et al., 2017] menghasilkan peta
panas (*heatmap*) yang menyoroti wilayah citra paling berpengaruh terhadap
prediksi suatu kelas. Caranya, gradien skor kelas terhadap peta fitur konvolusi
terakhir dirata-ratakan secara spasial menjadi bobot tiap kanal, lalu
dikombinasikan dengan peta fitur dan dilewatkan fungsi ReLU. Hasilnya
menunjukkan "di mana model melihat" saat membuat keputusan.

**Grad-CAM++** [Chattopadhyay et al., 2018] menyempurnakan Grad-CAM dengan
pembobotan piksel berbasis turunan orde lebih tinggi, sehingga lebih akurat
ketika terdapat **beberapa instans** objek dari kelas yang sama dalam satu citra
— kondisi yang lazim pada deteksi TBS (banyak tandan dalam satu bingkai). Karena
itu Grad-CAM++ dipilih sebagai metode XAI dalam penelitian ini.

Kualitas penjelasan dievaluasi secara kuantitatif melalui metrik *faithfulness*
(kesetiaan penjelasan terhadap perilaku model sesungguhnya):

- **Average Drop (AD)** mengukur rata-rata penurunan keyakinan model ketika
  wilayah yang disorot *heatmap* (intensitas di atas ambang) **ditutup** dari
  citra masukan. Nilai AD yang **lebih tinggi lebih baik**: menutup wilayah
  sorotan menjatuhkan kepercayaan model secara berarti, membuktikan wilayah
  itu memang menjadi dasar prediksi (penjelasan *faithful*).
- **Focus Retention Rate (FRR)** mengukur fraksi total intensitas *heatmap*
  Grad-CAM++ yang jatuh **di dalam kotak pembatas (ROI) deteksi**. FRR yang
  lebih tinggi menandakan atensi model terkonsentrasi pada objek (buah),
  bukan pada latar belakang.

Dalam penelitian ini, metrik XAI dipakai untuk memverifikasi bahwa penerapan
DP-SGD tidak hanya memengaruhi akurasi numerik tetapi juga *kualitas alasan*
model — apakah model yang dilatih dengan privasi tetap "melihat" buah pada lokasi
yang benar.

## 2.9 Penelitian Terkait

Bagian ini meninjau penelitian terdahulu pada lima untai yang menjadi
fondasi FedX-Palm, sekaligus memetakan posisi penelitian ini terhadapnya.

**(a) Deteksi/klasifikasi kematangan TBS sawit berbasis** ***deep learning***.
Septiarini et al. [2020] mengklasifikasikan kematangan tandan buah segar
(TBS) sawit menggunakan fitur warna dan tekstur dengan pembelajaran mesin
klasik, sementara Suharjito et al. [2021] menyusun dataset citra TBS
beranotasi untuk pembelajaran mesin dan mengevaluasi beberapa arsitektur
CNN. Mansour et al. [2018] dan Saleh & Liansitim [2020] menunjukkan *deep
learning* mengungguli pendekatan fitur-tangan untuk pemeringkatan
kematangan buah. Mayoritas karya ini berhenti pada **klasifikasi citra
ter-*crop*** dan dilatih secara **tersentral** pada satu dataset, belum
menyentuh deteksi multi-objek pada citra lapangan utuh maupun aspek privasi
data lintas-perkebunan.

**(b) YOLO untuk deteksi objek pertanian.** Sejak Redmon et al. [2016]
memperkenalkan YOLO sebagai detektor satu tahap *real-time*, keluarga YOLO
banyak diadopsi untuk deteksi buah dan tanaman karena keseimbangan
kecepatan-akurasinya. Varian mutakhir YOLOv11 [Khanam & Hussain, 2024]
memperbaiki *backbone* dan kepala deteksi dengan jumlah parameter yang
ringkas (varian nano ~2,6 juta), cocok untuk *inference edge*. Penelitian
ini memanfaatkan YOLOv11n tetapi memodifikasinya (BatchNorm → GroupNorm)
agar kompatibel dengan akuntansi gradien per-sampel — modifikasi yang tidak
dibahas pada literatur deteksi pertanian arus utama.

**(c)** ***Federated Learning*** **pada visi komputer.** FedAvg [McMahan
et al., 2017] menjadi algoritma agregasi rujukan; survei Kairouz et al.
[2021], Yang et al. [2019], dan Li et al. [2020] memetakan tantangan FL
termasuk heterogenitas data Non-IID, yang Hsu et al. [2019] formalkan
melalui partisi Dirichlet. Penerapan FL pada domain sensitif telah
ditunjukkan pada citra medis [Rieke et al., 2020] dan teks seluler [Hard
et al., 2018]. Namun penerapan FL pada **deteksi objek** (bukan
klasifikasi) masih jarang, dan hampir tidak ada pada domain agrikultur
sawit — celah yang diisi penelitian ini.

**(d)** ***Differential Privacy*** **dalam pembelajaran mendalam dan FL.**
DP-SGD [Abadi et al., 2016] menjadi mekanisme standar privasi tingkat
sampel, dengan akuntansi anggaran privasi yang diperketat oleh Rényi-DP
[Mironov, 2017] dan *concentrated DP* [Bun & Steinke, 2016; Dwork et
al., 2010]. Pada konteks FL, DP dapat diterapkan di tingkat klien
(DP-FedAvg [McMahan et al., 2018]) maupun tingkat sampel; Wei et al. [2020]
dan Truex et al. [2020] mengkaji trade-off privasi-utilitas keduanya.
Tramèr & Boneh [2021] berargumen bahwa membatasi parameter yang dilatih
(*partial fine-tuning*) memperbaiki efisiensi DP-SGD pada **klasifikasi
citra**. Penelitian ini menguji klaim tersebut pada **deteksi objek domain
baru** dan — sebagaimana dilaporkan Bab 4.5 — menemukan hasil yang berbeda.

**(e) XAI untuk evaluasi model visual.** Grad-CAM [Selvaraju et al., 2017]
dan penyempurnaannya Grad-CAM++ [Chattopadhyay et al., 2018] adalah metode
atribusi visual yang dominan untuk CNN, sementara SHAP [Lundberg & Lee,
2017] menawarkan atribusi agnostik-model. Kebutuhan privasi yang
memotivasi DP berakar pada serangan inferensi nyata — *membership
inference* [Shokri et al., 2017], *model inversion* [Fredrikson et al.,
2015], dan *deep leakage from gradients* [Zhu et al., 2019]. Sebagian besar
studi XAI mengevaluasi model **non-privat dan tersentral**; pengaruh
pelatihan DP-federated terhadap *faithfulness* penjelasan belum banyak
dikuantifikasi.

**Posisi penelitian ini.** Tabel 2.1 merangkum cakupan karya terdahulu.
Sepanjang penelusuran penulis, belum ada karya yang **menyatukan** deteksi
objek YOLOv11 + FL + DP-SGD per-sampel + evaluasi XAI kuantitatif +
demonstrasi *deployment* dalam satu kerangka untuk domain kematangan TBS
sawit. FedX-Palm mengisi celah integratif tersebut.

| Karya | Deteksi (bukan klasifikasi) | Federated | Differential Privacy | XAI kuantitatif | Deployment |
|---|:--:|:--:|:--:|:--:|:--:|
| Septiarini et al. [2020]; Suharjito et al. [2021] | – | – | – | – | – |
| Redmon et al. [2016]; Khanam & Hussain [2024] | ✓ | – | – | – | – |
| McMahan et al. [2017]; Rieke et al. [2020] | – | ✓ | – | – | – |
| Abadi et al. [2016]; Wei et al. [2020] | – | ✓ | ✓ | – | – |
| Tramèr & Boneh [2021] | – | – | ✓ | – | – |
| Selvaraju et al. [2017]; Chattopadhyay et al. [2018] | – | – | – | ✓ | – |
| **FedX-Palm (penelitian ini)** | **✓** | **✓** | **✓** | **✓** | **✓** |

## 2.10 Containerization untuk Deployment (Docker)

Tahap akhir siklus penelitian adalah menyajikan model terlatih agar dapat
digunakan di lapangan. **Docker** adalah teknologi *containerization* yang
mengemas aplikasi beserta seluruh dependensinya ke dalam unit terisolasi yang
portabel. Tiga konsep perlu dibedakan: **Dockerfile** adalah *cetak biru*
(*blueprint*) berupa berkas teks yang mendeskripsikan langkah membangun
lingkungan; **image** adalah paket hasil "build" dari Dockerfile; dan
**container** adalah image yang sedang berjalan. Analoginya, Dockerfile bagaikan
gambar arsitek, image adalah rumah prefabrikasi, dan container adalah rumah yang
dihuni.

Penting ditegaskan ruang lingkup pemanfaatan Docker dalam penelitian ini.
Docker digunakan **semata-mata untuk deployment inference** model akhir
(`best.pt`), **bukan** untuk pelatihan maupun untuk merealisasikan arsitektur
*Federated Learning*. Pelatihan FL dilakukan melalui **simulasi sequential pada
satu GPU** (lihat Bab 3), yang menghasilkan nilai numerik identik dengan
deployment terdistribusi namun jauh lebih hemat sumber daya; privasi dijamin oleh
mekanisme DP-SGD per-sampel, bukan oleh isolasi jaringan antarkontainer. Dengan
demikian, *containerization* hanya relevan pada saat menyajikan model.

Nilai utama Docker di sini adalah **reproducibility dan portabilitas**:
Dockerfile berlaku sebagai cetak biru yang menjamin model dapat dijalankan ulang
di lingkungan mana pun — mini-PC pabrik, VPS, maupun server — tanpa konflik
dependensi. Image dirancang *CPU-only* karena target deployment lapangan umumnya
tanpa GPU, ukuran image menjadi jauh lebih kecil, dan YOLOv11n cukup ringan untuk
inferensi pada CPU. Dalam penelitian ini *image* Docker tidak berhenti sebagai
cetak biru: image **dibangun dan dijalankan sebagai layanan inferensi nyata pada
sebuah VPS** (Bab 4.10), sehingga kesiapan deployment (*deployment-readiness*)
dibuktikan secara operasional, bukan sekadar di atas kertas.
