# Deteksi Kematangan Tandan Buah Segar Kelapa Sawit Enam Kelas Berbasis *Federated Learning* dengan Penjelasan Visual Grad-CAM++ Tervalidasi untuk Pendukung Keputusan Panen

> **Target**: JUTI (Jurnal Teknik Informatika dan Sistem Informasi), ITS — SINTA 3.
> **Status**: draf IMRaD. `{TBD: ...}` = angka menunggu full grid + script XAI per-kelas.
> Judul alternatif (pilih satu saat submit):
> - "Evaluasi Keterjelasan Deteksi Kematangan Kelapa Sawit pada *Federated Learning* Menggunakan Grad-CAM++"
> - "Deteksi Kematangan Kelapa Sawit yang Akurat dan Dapat Dijelaskan dengan *Federated Learning* dan Grad-CAM++"

---

## Abstrak

Penentuan waktu panen tandan buah segar (TBS) kelapa sawit yang tepat sangat
menentukan rendemen dan mutu minyak, namun penilaian manual bersifat subjektif
dan tidak konsisten. Otomasi berbasis visi komputer menjanjikan keseragaman,
tetapi pengumpulan citra perkebunan secara terpusat terbentur kerahasiaan data
operasional. Penelitian ini mengevaluasi apakah pendekatan *Federated Learning*
(FL) — yang melatih model bersama tanpa memindahkan citra mentah antar-kebun —
dapat menghasilkan deteksi kematangan TBS enam kelas yang **akurat sekaligus
dapat dijelaskan**, sehingga layak menjadi pendukung keputusan panen yang
dipercaya operator. Detektor YOLOv11n dilatih secara federasi (FedAvg) pada
data Non-IID (partisi Dirichlet) hasil pembagian anti-kebocoran berbasis
identitas tandan (*bunch_id*), lalu kualitas penjelasannya dievaluasi
menggunakan Grad-CAM++ melalui metrik *faithfulness*: *Average Drop* (AD) dan
*Faithfulness/Region Retention Rate* (FRR), dilengkapi kurva *Insertion/Deletion*
dan validasi agronomis kesesuaian sorotan *heatmap* terhadap isyarat visual
kematangan. Model federasi mencapai mAP@0.5 = `{TBD: B2 best mAP50}`
(pembanding tersentral mAP@0.5 = 0,787), dengan AD = `{TBD}` dan
FRR = `{TBD}`. Analisis per-kelas menunjukkan `{TBD: kelas paling mudah/sulit
dijelaskan}`. Hasil ini mengindikasikan bahwa FL dapat menyediakan deteksi
kematangan sawit yang menjaga privasi data antar-kebun tanpa mengorbankan
keterjelasan keputusan model. Aspek privasi formal (*Differential Privacy*)
dibahas secara ringkas dan dirujuk ke publikasi pendahuluan penulis.

**Kata kunci**: kelapa sawit, deteksi kematangan, *federated learning*,
YOLOv11, *explainable AI*, Grad-CAM++, pertanian presisi.

## Abstract

Timely harvesting of oil palm fresh fruit bunches (FFB) is decisive for oil
yield and quality, yet manual ripeness assessment is subjective and
inconsistent. Computer-vision automation promises consistency, but centrally
aggregating plantation imagery conflicts with the confidentiality of
operational data. This study evaluates whether a Federated Learning (FL)
approach — training a shared model without moving raw imagery across estates —
can deliver six-class FFB ripeness detection that is **both accurate and
explainable**, making it a trustworthy harvest decision-support tool. A
YOLOv11n detector is trained federatively (FedAvg) on Non-IID (Dirichlet)
partitions derived from a leakage-free bunch-identity split, and its
explanation quality is assessed with Grad-CAM++ via faithfulness metrics —
Average Drop (AD) and Faithfulness/Region Retention Rate (FRR) — complemented
by Insertion/Deletion curves and an agronomic validation of heatmap–cue
alignment. The federated model attains mAP@0.5 = `{TBD}` (centralized
reference 0.787), with AD = `{TBD}` and FRR = `{TBD}`. Per-class analysis
reveals `{TBD}`. The results indicate FL can provide privacy-preserving palm
ripeness detection without sacrificing decision explainability. Formal privacy
(Differential Privacy) is discussed briefly and referred to the authors' prior
work.

**Keywords**: oil palm, ripeness detection, federated learning, YOLOv11,
explainable AI, Grad-CAM++, precision agriculture.

---

## I. PENDAHULUAN

Industri kelapa sawit merupakan tulang punggung agribisnis Indonesia dan
Malaysia yang bersama-sama menyumbang lebih dari 85% produksi minyak sawit
dunia [1]. Pada tingkat operasional, keputusan paling menentukan rendemen
adalah ketepatan waktu panen TBS: tandan yang dipanen terlalu dini berkadar
minyak rendah, sedangkan yang terlambat meningkatkan kadar Asam Lemak Bebas
dan menurunkan mutu *Crude Palm Oil* [2]. Penilaian manual oleh mandor
bersifat subjektif dengan tingkat kesalahan 15–25% pada pencahayaan suboptimal
[3], sehingga otomasi berbasis visi komputer menjadi kebutuhan nyata.

Detektor objek modern seperti YOLOv11 mampu melokalisasi dan mengklasifikasi
beberapa tandan dalam satu bingkai secara *real-time* [4], namun pelatihannya
memerlukan banyak citra beragam. Pengumpulan terpusat terbentur kenyataan
bahwa citra perkebunan adalah **informasi bisnis sensitif** (lokasi blok,
volume produksi) yang enggan dibagikan antar-perusahaan. *Federated Learning*
[5] menjawab kendala ini dengan melatih model bersama tanpa memindahkan data
mentah; dalam taksonomi [6], skenario perkebunan termasuk *cross-silo*.

Namun, akurasi saja tidak cukup agar sebuah model dipercaya dan diadopsi
operator di lapangan. Sistem pendukung keputusan pertanian memerlukan
**keterjelasan** (*explainability*): operator perlu memahami *mengapa* model
mengklasifikasikan sebuah tandan sebagai "matang" atau "terlalu matang".
Tanpa penjelasan yang dapat divalidasi, prediksi model berisiko ditolak atau,
lebih buruk, dipercaya secara membabi buta. Metode *Explainable AI* (XAI)
seperti Grad-CAM++ [7] menghasilkan peta panas yang menyoroti wilayah citra
penentu prediksi, dan kualitasnya dapat diukur kuantitatif melalui metrik
*faithfulness*.

Berbeda dengan studi pendahuluan penulis yang berfokus pada perilaku
*Differential Privacy* pada FL [8], **penelitian ini berfokus pada pertanyaan
aplikatif**: *Apakah deteksi kematangan TBS berbasis FL dapat sekaligus akurat
dan dapat dijelaskan dengan andal sehingga layak menjadi pendukung keputusan
panen?* Kontribusi penelitian ini ada tiga:

1. **Evaluasi keterjelasan mendalam** deteksi kematangan sawit federasi —
   melampaui satu angka global — melalui analisis Grad-CAM++ **per-kelas
   kematangan**, kurva *Insertion/Deletion*, serta validasi agronomis
   kesesuaian sorotan *heatmap* terhadap isyarat visual kematangan.
2. **Protokol pembagian data anti-kebocoran berbasis *bunch_id*** untuk deteksi
   sawit, mencegah kebocoran tingkat-tandan yang menggelembungkan metrik secara
   semu.
3. **Bukti empiris** bahwa federasi tidak menurunkan kualitas penjelasan secara
   signifikan dibanding pelatihan tersentral, memperkuat kelayakan FL untuk
   pendukung keputusan pertanian yang menjaga privasi.

## II. TINJAUAN PUSTAKA

### A. Deteksi Kematangan Kelapa Sawit

`{TBD: 1 paragraf — ringkas 3-4 studi deteksi/klasifikasi kematangan sawit
sebelumnya (Junos 2022, dll), tekankan bahwa mayoritas mengasumsikan
pelatihan terpusat dan tidak mengevaluasi keterjelasan secara kuantitatif —
celah yang diisi paper ini.}`

### B. Federated Learning untuk Pertanian

Singkat: FedAvg [5], cross-silo [6], tantangan Non-IID dan partisi Dirichlet
[9]. `{TBD: tambah 1-2 referensi FL di domain pertanian/agriculture jika ada,
untuk menegaskan kebaruan domain.}`

### C. Explainable AI dan Metrik Faithfulness

Grad-CAM [10] dan penyempurnaannya Grad-CAM++ [7] yang lebih akurat saat
terdapat banyak instans objek sejenis — kondisi lazim pada citra TBS. Kualitas
penjelasan diukur via *Average Drop* dan *Region Retention*, serta kurva
*Insertion/Deletion* [11] yang mengukur perubahan keyakinan saat piksel penting
ditambahkan/dihapus bertahap.

## III. METODOLOGI

### A. Dataset dan Pembagian Anti-Kebocoran

Dataset terdiri atas 10.814 citra dari 89 tandan unik dengan anotasi enam kelas
kematangan: *Abnormal*, *Empty Bunch*, *Overripe*, *Ripe*, *Underripe*,
*Unripe*. Karena beberapa citra berasal dari tandan yang sama, pembagian acak
biasa berisiko membocorkan tandan ke *train* dan *test* sekaligus
(menggelembungkan metrik). Penelitian ini menerapkan **pembagian terstratifikasi
berbasis *bunch_id*** pada tingkat tandan: Train 9.094 citra (72 tandan), Valid
769 (8 tandan), Test 951 (9 tandan), dengan audit memastikan nol irisan tandan
antar-split.

### B. Partisi Federasi Non-IID

Data *train* dipartisi ke `{TBD: jumlah K yang dilaporkan, mis. K=4}` klien
menggunakan distribusi Dirichlet untuk mensimulasikan heterogenitas antar-kebun
[9]. Himpunan validasi dan uji bersifat global.

### C. Arsitektur dan Pelatihan

Detektor YOLOv11n (~2,6 juta parameter) diinisialisasi dari bobot pra-latih
COCO. Seluruh lapisan BatchNorm dikonversi menjadi GroupNorm agar konsisten
dengan jalur eksperimen privasi penulis [8]; rincian justifikasi dirujuk ke
[8]. Pelatihan federasi memakai FedAvg, lima ronde komunikasi, dua epoch lokal
per ronde (SGD, *lr* 0,01, batch 16, citra 640×640). Sebagai pembanding,
*baseline* tersentral dilatih 50 epoch pada seluruh data.

> **Catatan privasi (ringkas).** Varian penjaga-privasi formal melalui DP-SGD
> per-sampel diuji pada studi pendahuluan penulis [8]; di sini privasi
> ditangani pada tingkat arsitektur (data tak berpindah antar-klien) dan detail
> *Differential Privacy* tidak diulang demi fokus pada keterjelasan.

### D. Evaluasi Keterjelasan (XAI)

Grad-CAM++ diterapkan pada lapisan konvolusi terakhir sebelum kepala deteksi.
Kualitas penjelasan diukur pada `{TBD: jumlah}` citra uji menggunakan:

- **Average Drop (AD)**: rata-rata penurunan keyakinan saat masukan dibatasi
  pada wilayah sorotan (semakin tepat → wilayah benar-benar penting).
- **Faithfulness/Region Retention Rate (FRR)**: proporsi keyakinan yang
  dipertahankan saat hanya wilayah penting yang disisakan.
- **Kurva Insertion/Deletion** [11]: AUC keyakinan terhadap fraksi piksel
  penting yang ditambahkan (Insertion, tinggi=baik) / dihapus (Deletion,
  rendah=baik).
- **Validasi agronomis**: IoU antara wilayah sorotan *heatmap* (di atas ambang)
  dengan *bounding box* buah, sebagai proksi apakah model "melihat" buah dan
  bukan latar.

Metrik dihitung **per-kelas kematangan** untuk mengungkap kelas yang paling
mudah/sulit dijelaskan.

## IV. HASIL DAN PEMBAHASAN

### A. Akurasi Deteksi

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| Tersentral (acuan) | 0,787 | 0,672 | 0,815 | 0,833 |
| Federasi (FedAvg) | `{TBD: B2 mAP50}` | `{TBD}` | `{TBD}` | `{TBD}` |

`{TBD: 1 paragraf — berapa selisih federasi vs tersentral (biaya FL)? Apakah
masih dalam rentang berguna untuk pendukung keputusan?}`

### B. Kualitas Penjelasan Global

| Metrik | Federasi | Interpretasi |
|---|---|---|
| Average Drop | `{TBD}` | `{TBD}` |
| FRR | `{TBD}` | `{TBD}` |
| Insertion AUC | `{TBD}` | `{TBD}` |
| Deletion AUC | `{TBD}` | `{TBD}` |

### C. Analisis Per-Kelas (kontribusi inti)

| Kelas | AP@0.5 | AD | FRR | IoU heatmap–buah |
|---|---|---|---|---|
| Unripe | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` |
| Underripe | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` |
| Ripe | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` |
| Overripe | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` |
| Empty Bunch | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` |
| Abnormal | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` |

`{TBD: 2 paragraf — kelas mana paling andal dijelaskan (mis. Ripe/Overripe
karena isyarat warna kuat) vs paling lemah (mis. Abnormal/Empty Bunch karena
variasi visual tinggi)? Kaitkan dengan isyarat agronomis: warna kulit,
brondolan, tekstur. Inilah nilai aplikatif: operator bisa lebih waspada pada
kelas yang penjelasannya lemah.}`

### D. Pengaruh Federasi terhadap Keterjelasan

`{TBD: 1 paragraf — bandingkan AD/FRR model federasi vs tersentral. Jika
selisih kecil → federasi tidak merusak keterjelasan → temuan utama yang
memperkuat kelayakan FL untuk pendukung keputusan.}`

### E. Analisis Kualitatif

`{TBD: Gambar — tampilkan heatmap Grad-CAM++ untuk 1 citra representatif per
kelas; diskusikan apakah sorotan jatuh pada permukaan buah/brondolan (benar)
atau latar (salah).}` (`figures/xai_per_class.png`)

## V. KESIMPULAN

`{TBD: 1 paragraf}` Penelitian ini menunjukkan bahwa deteksi kematangan TBS
kelapa sawit enam kelas berbasis *Federated Learning* dapat mencapai akurasi
`{TBD: kategori}` sekaligus mempertahankan keterjelasan yang
`{TBD: tinggi/memadai}` menurut metrik *faithfulness* Grad-CAM++. Analisis
per-kelas mengungkap bahwa `{TBD}`, memberi panduan praktis bagi operator.
Federasi `{TBD: tidak/sedikit}` menurunkan kualitas penjelasan dibanding
pelatihan tersentral, sehingga FL layak menjadi fondasi pendukung keputusan
panen yang menjaga privasi data antar-kebun. Penelitian lanjut mencakup
`{TBD: integrasi DP-SGD penuh (lihat [8]), validasi multi-kebun nyata,
deployment edge}`.

## UCAPAN TERIMA KASIH

`{TBD}`

## DAFTAR PUSTAKA

[1] USDA FAS, *Oilseeds: World Markets and Trade*, 2024.
[2] M. H. Junos et al., "Automatic detection of oil palm fruits from UAV images
using an improved YOLO model," *The Visual Computer*, vol. 38, 2022.
[3] N. Sabri et al., "Palm oil FFB ripeness grading using color features,"
*J. Fundam. Appl. Sci.*, 2017.
[4] R. Khanam and M. Hussain, "YOLOv11: An Overview of the Key Architectural
Enhancements," arXiv:2410.17725, 2024.
[5] B. McMahan et al., "Communication-Efficient Learning of Deep Networks from
Decentralized Data," in *AISTATS*, 2017.
[6] P. Kairouz et al., "Advances and Open Problems in Federated Learning,"
*Found. Trends Mach. Learn.*, 2021.
[7] A. Chattopadhyay et al., "Grad-CAM++: Generalized Gradient-Based Visual
Explanations for Deep CNNs," in *WACV*, 2018.
[8] [Author], "FedX-Palm: An Explainable Federated Framework for Oil Palm
Ripeness Detection — A Characterized Failure Mode and Mitigation for
Client-Level Differential Privacy on YOLOv11," `{TBD: venue/status Paper A}`.
[9] T.-M. H. Hsu et al., "Measuring the Effects of Non-Identical Data
Distribution for Federated Visual Classification," arXiv:1909.06335, 2019.
[10] R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization," in *ICCV*, 2017.
[11] V. Petsiuk et al., "RISE: Randomized Input Sampling for Explanation of
Black-box Models," in *BMVC*, 2018.
