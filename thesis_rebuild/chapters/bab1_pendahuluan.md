# BAB 1 PENDAHULUAN

## 1.1 Latar Belakang

Industri kelapa sawit merupakan salah satu sektor agribisnis terbesar di
Indonesia dan Malaysia, dengan kedua negara tersebut menyumbang lebih dari
85% produksi minyak sawit dunia [USDA, 2024]. Pada tingkat operasional
perkebunan, keputusan kunci yang menentukan rendemen minyak adalah penentuan
waktu panen Tandan Buah Segar (TBS). Tandan yang dipanen terlalu dini
menghasilkan kandungan minyak yang rendah, sementara tandan yang dipanen
terlambat mengalami peningkatan kadar Asam Lemak Bebas (Free Fatty Acid)
sehingga menurunkan mutu Crude Palm Oil (CPO). Standar industri membagi
kematangan TBS menjadi enam kelas yang relevan secara komersial: *Unripe*,
*Underripe*, *Ripe*, *Overripe*, *Empty Bunch*, dan *Abnormal*
[Junos et al., 2022].

Penilaian kematangan TBS dalam praktik tradisional dilakukan secara visual
oleh mandor panen berdasarkan warna kulit buah dan jumlah brondolan yang
jatuh. Pendekatan ini bersifat subjektif, sangat bergantung pada pengalaman
individu, dan rentan terhadap kelelahan operator. Studi pendahuluan
menunjukkan tingkat kesalahan klasifikasi manual dapat mencapai 15-25%
pada kondisi pencahayaan suboptimal [Sabri et al., 2017]. Otomasi penilaian
kematangan melalui visi komputer karenanya menjadi kebutuhan yang
mendesak, terutama untuk perkebunan skala besar yang memerlukan
keseragaman keputusan panen lintas blok kebun.

Perkembangan deep learning untuk object detection menghadirkan
keluarga model YOLO (You Only Look Once) sebagai standar de facto dalam
deteksi objek real-time. Versi terbaru, YOLOv11, menawarkan arsitektur
yang ringan (~2.6 juta parameter pada varian *nano*) dengan kecepatan
inferensi yang memungkinkan deployment pada perangkat *edge* seperti
kamera CCTV pabrik atau drone surveillance [Khanam & Hussain, 2024].
YOLOv11 mengintegrasikan loss CIoU untuk regresi *bounding box*,
Distribution Focal Loss untuk klasifikasi, serta arsitektur *anchor-free*
yang menyederhanakan post-processing.

Meskipun YOLOv11 menjanjikan akurasi tinggi pada deteksi TBS, pelatihan
model yang andal membutuhkan dataset citra dalam jumlah besar dan
beragam—mencakup variasi musim, pencahayaan, dan kondisi geografis.
Pengumpulan dataset terpusat menghadapi dua kendala utama. Pertama,
citra perkebunan bersifat **informasi bisnis sensitif**: lokasi blok,
volume produksi, dan kondisi tanaman dapat diinferensi dari koleksi
citra dan dianggap rahasia dagang oleh perusahaan perkebunan. Kedua,
regulasi seperti Permentan No. 11/2020 tentang sertifikasi ISPO dan
prinsip RSPO mengharuskan kerahasiaan data operasional antar entitas.
Akibatnya, perusahaan enggan mengirimkan citra mentah ke server pusat
walaupun untuk tujuan pelatihan model bersama.

**Federated Learning (FL)** memberikan jawaban arsitektural terhadap
kendala ini. Diperkenalkan oleh McMahan et al. [2017], FL memungkinkan
beberapa entitas (dalam konteks ini, beberapa perkebunan) untuk
mempelajari model bersama tanpa pernah memindahkan data mentah keluar
dari batas masing-masing entitas. Hanya parameter model (atau update
parameter) yang dipertukarkan dengan server agregator melalui algoritma
seperti FedAvg. Dalam taksonomi Kairouz et al. [2021], skenario
perkebunan termasuk *cross-silo FL*: jumlah peserta sedikit (puluhan,
bukan jutaan), setiap peserta memiliki data dalam jumlah besar, dan
ketersediaan partisipan stabil—berbeda dengan *cross-device FL* pada
ponsel pengguna.

Namun riset terkini menunjukkan bahwa FL saja tidak cukup memberikan
jaminan privasi formal. Parameter atau update yang dipertukarkan
selama pelatihan dapat dimanfaatkan oleh adversary untuk
merekonstruksi data latih melalui serangan seperti *membership inference*
[Shokri et al., 2017], *model inversion* [Fredrikson et al., 2015], dan
*gradient leakage* [Zhu et al., 2019]. Untuk skenario perkebunan
sawit, kebocoran semacam ini dapat memungkinkan kompetitor
menginferensi karakteristik kebun—misalnya distribusi tingkat kematangan
yang merepresentasikan strategi panen—dari pembaruan model yang
terobservasi.

**Differential Privacy (DP)** [Dwork et al., 2006] adalah kerangka
matematis yang memberikan jaminan formal terhadap inferensi data
individu. Sebuah algoritma dikatakan memenuhi (ε,δ)-DP apabila perubahan
satu sampel dalam dataset hanya mengubah distribusi keluaran sebesar
faktor `exp(ε)` (dengan probabilitas kegagalan δ). Implementasi praktis
DP pada deep learning diperkenalkan oleh Abadi et al. [2016] sebagai
**DP-SGD** (Differentially Private Stochastic Gradient Descent), yang
menambahkan noise Gaussian terkalibrasi pada *per-sample gradient* yang
telah di-clip ke norma maksimum tertentu.

Penggabungan FL dan DP secara naif—yaitu menambahkan noise pada
*aggregated delta* di tingkat klien (sering disebut *DP-FedAvg* level
klien)—telah ditunjukkan menyebabkan kegagalan konvergensi pada model
object detection dengan parameter pretrained [Hu et al., 2024]. Hipotesis
yang dikemukakan adalah noise pada bobot teragregasi bersifat
*persistent*: tidak dapat di-dampen oleh momentum optimizer dan
berakumulasi sepanjang ronde komunikasi. Sebaliknya, DP-SGD per-sampel
menambahkan noise pada level gradient sehingga interaksi dengan
optimizer SGD-momentum lebih *graceful*.

Aspek tambahan yang relevan untuk kepercayaan stakeholder adalah
**Explainable AI (XAI)**. Model deep learning bersifat black-box;
mandor panen dan auditor perusahaan tidak dapat menelusuri alasan model
mengklasifikasi suatu TBS sebagai *Ripe* atau *Overripe*. Teknik
visualisasi seperti **Grad-CAM++** [Chattopadhyay et al., 2018]
menghasilkan peta panas yang menunjukkan area citra yang paling
berkontribusi terhadap keputusan model, sehingga memungkinkan validasi
visual dan akuntabilitas keputusan.

Penelitian ini mengintegrasikan keempat pilar tersebut—**YOLOv11 untuk
deteksi, Federated Learning untuk kolaborasi lintas-perkebunan,
Differential Privacy (DP-SGD) untuk proteksi formal, dan Grad-CAM++
untuk validasi visual**—menjadi sebuah kerangka utuh untuk deteksi
kematangan TBS yang akurat, kolaboratif, terverifikasi privasinya, dan
dapat dijelaskan. Penekanan utama riset adalah kuantifikasi
*privacy-utility trade-off* pada DP-SGD untuk object detection, sebuah
celah yang relatif kurang dieksplorasi dibandingkan kasus klasifikasi
citra.


## 1.2 Rumusan Masalah

Berdasarkan latar belakang di atas, penelitian ini merumuskan tiga
pertanyaan utama:

**RQ1** Bagaimana arsitektur Horizontal Federated Learning dengan
backbone YOLOv11 dapat mencapai utilitas (mAP@0.5) yang setara dengan
pelatihan terpusat pada dataset kematangan TBS yang terdistribusi
secara *Non-Independent and Identically Distributed* (Non-IID) antar
perkebunan?

**RQ2** Bagaimana penerapan Differential Privacy DP-SGD secara
per-sampel pada klien lokal memengaruhi *privacy-utility trade-off*
sistem? Secara spesifik: (a) bagaimana kurva mAP terhadap ε untuk
strategi *full DP-SGD* (seluruh parameter dilindungi) dibandingkan
*partial DP-SGD* (hanya detection head dilindungi); dan (b) bagaimana
pengaruh skala jaringan federasi (jumlah klien K) terhadap stabilitas
training dan privasi tercapai?

**RQ3** Bagaimana visualisasi Grad-CAM++ dapat memvalidasi bahwa
model yang dilatih dengan DP-SGD tetap memfokuskan perhatian pada
area buah yang relevan secara semantik, diukur menggunakan metrik
*Average Drop* dan *Focus Retention Rate*?


## 1.3 Tujuan Penelitian

Tujuan umum penelitian ini adalah membangun, mengevaluasi, dan
mendokumentasikan kerangka HFL-YOLOv11 dengan proteksi DP-SGD dan
validasi XAI untuk deteksi kematangan TBS kelapa sawit. Tujuan khusus
dijabarkan sebagai berikut:

1. **Merancang dan mengimplementasikan arsitektur federated learning
   ter-Dockerisasi** dengan satu server agregator dan K klien
   (K ∈ {2, 4, 8, 12, 16}) yang masing-masing memuat shard data Non-IID
   hasil partisi Dirichlet.

2. **Mengonversi backbone YOLOv11n** dari BatchNorm ke GroupNorm agar
   kompatibel dengan kerangka per-sample gradient pada Opacus, dan
   memverifikasi bahwa konversi tidak menurunkan utilitas baseline
   secara signifikan.

3. **Menerapkan DP-SGD per-sampel** pada pelatihan lokal di setiap
   klien dengan dua strategi: *full* (seluruh ~2,6 juta parameter
   dilindungi) dan *partial* (hanya ~0,2 juta parameter detection head
   yang dilindungi, backbone dibekukan).

4. **Mengkuantifikasi trade-off privacy-utility** melalui sweep
   noise multiplier σ ∈ {0.5, 1.0, 1.5, 2.0, 3.0} dan jumlah klien
   K ∈ {2, 4, 8, 12, 16}, dengan ε dihitung menggunakan *Privacy Random
   Variable* (PRV) accountant pada δ = 1×10⁻⁵.

5. **Membandingkan secara empiris** mekanisme DP-SGD per-sampel
   dengan DP-FedAvg level-klien yang dilaporkan mengalami *collapse*
   pada eksperimen pendahuluan, sebagai justifikasi pemilihan
   mekanisme DP yang tepat untuk object detection pretrained.

6. **Mengevaluasi interpretabilitas** model terdistribusi-terenkripsi
   menggunakan Grad-CAM++ dengan metrik kuantitatif Average Drop dan
   Focus Retention Rate pada subset validasi.

7. **Membangun dan menjalankan layanan inferensi berbasis Docker**:
   mengemas model akhir ke dalam *image* Docker dan men-*deploy*-nya
   sebagai layanan inferensi nyata pada sebuah VPS (Bab 4.10), sekaligus
   menyediakan *blueprint* yang dapat direplikasi oleh kelompok riset
   atau perusahaan perkebunan tanpa ketergantungan pada infrastruktur
   khusus.


## 1.4 Batasan Penelitian

Untuk menjaga fokus dan reproducibility, penelitian ini dibatasi pada:

1. **Dataset**: Roboflow `palm-fruit-ripeness-detection-f6sac-ccb2z`
   versi 2, dengan enam kelas (urutan alfabet sesuai sumber).
   Penelitian tidak mengumpulkan dataset baru maupun memvalidasi pada
   dataset eksternal lintas-geografi.

2. **Backbone**: YOLOv11n (varian nano) dengan GroupNorm sebagai
   pengganti BatchNorm. Varian YOLOv11 lain (s/m/l/x) di luar cakupan.

3. **Jumlah klien**: K ∈ {2, 4, 8, 12, 16} dalam mode *cross-silo*.
   Skenario *cross-device* dengan ribuan klien dan ketersediaan
   intermiten tidak diteliti.

4. **Mekanisme DP**: DP-SGD per-sampel via library Opacus 1.5.x.
   Alternatif seperti DP-Adam, *secure aggregation*, atau
   *homomorphic encryption* tidak diteliti.

5. **Sweep parameter privasi**: σ ∈ {0.5, 1.0, 1.5, 2.0, 3.0} dengan
   `max_grad_norm` C = 1.0 tetap. Ablation atas C terbatas pada satu
   konfigurasi K-σ kunci, kalau waktu memungkinkan.

6. **Komputasi**: pelatihan *Federated Learning* dijalankan sebagai
   simulasi *sequential* single-GPU (RTX 4080) tanpa replikasi
   multi-host — yang di luar cakupan adalah pelatihan FL terdistribusi
   lintas-host fisik, bukan deployment-nya. Adapun *deployment* inferensi
   berbasis Docker **dijalankan secara nyata** pada satu VPS CPU-only
   (Bab 4.10).

7. **Threat model**: *honest-but-curious server* dengan kemampuan
   melihat pembaruan model per ronde. Adversary aktif (Byzantine)
   maupun analisis serangan empiris (Membership Inference Attack)
   tidak dijalankan; klaim privasi bersandar pada jaminan formal
   (ε,δ)-DP via Opacus PRV accountant.


## 1.5 Metode Penelitian (Ringkas)

Metodologi mengikuti delapan tahap yang dijelaskan rinci di Bab 3:

1. **Studi literatur** atas YOLO, FL, DP-SGD, GroupNorm, dan XAI.
2. **Perancangan arsitektur** server-klien ter-Dockerisasi.
3. **Persiapan dataset** yang reproducible: download dari Roboflow,
   re-split berbasis *bunch_id* untuk anti-leakage, partisi Dirichlet
   per K ∈ {2, 4, 8, 12, 16}.
4. **Modifikasi backbone** YOLOv11n: BatchNorm → GroupNorm in-place.
5. **Pelatihan lokal** per klien dengan SGD-momentum, dan opsional
   pembungkusan Opacus PrivacyEngine untuk DP-SGD.
6. **Agregasi** FedAvg ter-bobot jumlah sampel per klien.
7. **Eksplanasi XAI** Grad-CAM++ pada model akhir.
8. **Evaluasi** dengan mAP@0.5, mAP@0.5:0.95, Precision, Recall, ε,
   Average Drop, dan Focus Retention Rate.

Eksperimen dirancang sebagai grid penuh: dua baseline (B1 centralized,
B2 federated tanpa DP), dan dua varian DP-SGD (E1 full, E2 partial)
masing-masing di-sweep atas K × σ.


## 1.6 Hipotesis

Penelitian ini menguji **empat hipotesis utama** (H1, H2, H2-K, H3),
ditambah satu **hipotesis pendukung** yang diadopsi dari literatur untuk
diuji secara empiris. Verifikasi seluruh hipotesis disajikan pada Bab 4.8.

**H1 (kelayakan baseline).** *Baseline* YOLOv11n dengan BatchNorm yang
dikonversi menjadi GroupNorm — prasyarat agar gradien per-sampel
terdefinisi — mencapai mAP@0.5 pada kategori *acceptable* (≥ 0,70) di set
validasi global anti-kebocoran, sehingga sah dijadikan *upper bound*
utilitas. Biaya federasi (FL-cost = mAP B1 − mAP B2) diharapkan kecil
(≤ 0,05) pada anggaran ronde komunikasi yang memadai.

**H2 (degradasi gradual).** *DP-SGD per-sampel menghasilkan trade-off
privasi-utilitas yang gradual*: penurunan mAP@0.5 bersifat monotonik
terhadap σ, bukan *cliff* / *collapse* mendadak seperti yang teramati
pada DP-FedAvg level-klien di eksperimen pendahuluan. Ambang operasional
*collapse*: mAP@0.5 < 0,05.

**H2-K (arah K terbalik).** *Arah pengaruh jumlah klien K pada DP-SGD
per-sampel berkebalikan dari DP-FedAvg level-klien*: K besar berarti
samples-per-klien kecil sehingga *noise* mendominasi sinyal lokal sebelum
agregasi global; mAP@0.5 diharapkan **menurun** seiring K naik pada σ
tetap. K = 16 dilaporkan sebagai *limit study* karena samples-per-klien
berada di ambang batas konvergensi DP-SGD.

**H3 (XAI tetap bermakna).** *Penjelasan Grad-CAM++ pada model operasional
tetap faithful secara kuantitatif*: Average Drop (penurunan kepercayaan
saat region salien ditutup, makin tinggi makin baik) dan Focus Retention
Rate menunjukkan atensi model terkonsentrasi pada region buah (ROI), bukan
latar, sehingga prediksi dapat diaudit secara visual.

**Hipotesis pendukung (Tramèr & Boneh, 2021).** Strategi *partial* DP-SGD
(hanya kepala deteksi yang dilatih, E2, ~0,2 juta parameter) diuji terhadap
*full* DP-SGD (E1, ~2,6 juta parameter): literatur memprediksi E2 lebih
efisien pada ε rendah karena dimensi gradien yang lebih kecil. Penelitian
ini menguji apakah prediksi tersebut berlaku pada deteksi objek di domain
baru (TBS sawit). Verdict empiris dilaporkan pada Bab 4.5.


## 1.7 Sistematika Penulisan

**Bab 1** memaparkan latar belakang, rumusan masalah, tujuan,
batasan, metode ringkas, hipotesis, dan sistematika.

**Bab 2** membahas landasan teori meliputi industri kelapa sawit
dan kematangan TBS, evolusi YOLO dan arsitektur YOLOv11, prinsip
Federated Learning dan FedAvg, definisi formal Differential Privacy
dan algoritma DP-SGD, mekanisme akuntansi PRV/RDP, kompatibilitas
GroupNorm dengan per-sample gradient, library Opacus, dan teknik
Explainable AI Grad-CAM++.

**Bab 3** menjabarkan metodologi penelitian: arsitektur sistem
Docker, persiapan dataset reproducible (00 → 02), modifikasi model
YOLOv11n-GN, algoritma DP-SGD lokal, agregasi FedAvg, skema
ablation K × σ, metrik evaluasi, dan lingkungan implementasi.

**Bab 4** menyajikan hasil dan pembahasan: validasi baseline
centralized B1, baseline federated B2 dengan sweep K, hasil
DP-SGD full E1 dan partial E2, perbandingan dengan DP-FedAvg,
analisis statistik multi-seed, dan validasi XAI.

**Bab 5** menyimpulkan kontribusi penelitian, mendiskusikan
keterbatasan, dan menyarankan arah riset lanjutan termasuk validasi
multi-dataset, skala cross-device, dan integrasi *secure aggregation*
sebagai komplemen DP.
