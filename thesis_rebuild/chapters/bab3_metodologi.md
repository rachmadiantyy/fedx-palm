# BAB 3 METODOLOGI PENELITIAN

Bab ini menguraikan rancangan penelitian secara menyeluruh: arsitektur sistem
dan strategi simulasi, dataset beserta prosedur partisi anti-kebocoran,
modifikasi model agar kompatibel dengan privasi per-sampel, prosedur pelatihan
lokal, mekanisme inti DP-SGD, agregasi *Federated Averaging*, metode penjelasan
XAI, metrik evaluasi, skenario eksperimen, lingkungan implementasi, serta
*blueprint* deployment. Seluruh prosedur dirancang agar dapat direproduksi dan
konsisten dengan implementasi perangkat lunak yang menyertainya.

## 3.1 Arsitektur Sistem dan Strategi Simulasi

Penelitian ini merealisasikan *Federated Learning* (FL) melalui **simulasi
sequential pada satu GPU**, bukan melalui sejumlah kontainer atau mesin fisik
terpisah. Pada setiap ronde komunikasi, sebuah proses Python tunggal
"mengaktifkan" $K$ klien secara berurutan: tiap klien menyalin parameter global,
melatih model pada *shard* data lokalnya, lalu mengembalikan parameter hasil
pelatihan; server kemudian mengagregasi seluruh parameter klien menjadi
parameter global baru (Algoritma FedAvg). Pseudokode tingkat tinggi yang
diimplementasikan pada modul `utils/fl_dp_loop.py` adalah:

```
inisialisasi parameter global w
untuk round = 1..T:
    untuk client = 1..K:
        salin w -> w_lokal
        (opsional) bungkus dengan Opacus PrivacyEngine (sigma, C)
        latih w_lokal selama E epoch pada shard client
    w = FedAvg(w_lokal dari semua client, bobot = ukuran shard)
    evaluasi w pada himpunan validasi global
    catat (round, sigma, K, mAP, epsilon)
```

Pemilihan simulasi sequential dilandasi tiga pertimbangan. **Pertama,
kesetaraan numerik**: karena FedAvg hanya bergantung pada parameter lokal dan
ukuran *shard* — bukan pada lokasi fisik komputasi — hasil numerik simulasi
identik dengan deployment terdistribusi. Mode simulasi ini merupakan praktik
baku pada riset FL (mis. *Flower simulation*, FedML) untuk eksperimen
terkendali. **Kedua, efisiensi sumber daya**: menjalankan $K$ proses paralel
berikut isolasi jaringan memerlukan sumber daya berlipat tanpa memberi nilai
ilmiah tambahan pada tahap riset. **Ketiga, kejelasan sumber jaminan privasi**:
dalam penelitian ini privasi dijamin oleh **mekanisme DP-SGD per-sampel**, bukan
oleh isolasi jaringan antarkontainer. Pemisahan data antar-klien tetap dijaga
secara nyata melalui partisi disk terpisah (`data/clients_K{K}/client_*/`),
sehingga tidak ada pencampuran data lintas-klien di dalam memori bersama.

Dengan demikian, *containerization* (Docker) tidak digunakan pada tahap
pelatihan, melainkan hanya pada tahap **deployment inference** model akhir
(lihat 3.11).

## 3.2 Dataset dan Partisi

### 3.2.1 Sumber dan Karakteristik

Dataset yang digunakan adalah *palm-fruit-ripeness-detection* (Roboflow, versi 2),
terdiri atas **10.814 citra** yang berasal dari **89 tandan unik** dengan anotasi
*bounding box* enam kelas kematangan berurutan alfabet: *Abnormal*, *Empty
Bunch*, *Overripe*, *Ripe*, *Underripe*, dan *Unripe*. Pengunduhan dan penyusunan
dilakukan oleh skrip `00_download_dataset.py`.

### 3.2.2 Re-split Anti-Kebocoran Berbasis bunch_id

Banyak dataset deteksi sawit berisi beberapa citra dari **tandan yang sama**
(diambil dari sudut/jarak berbeda). Jika citra dari satu tandan tersebar ke
*train* sekaligus *test*, model dapat "mengenali" tandan alih-alih belajar
fitur kematangan yang dapat digeneralisasi — sebuah **kebocoran data** (*data
leakage*) yang menggelembungkan metrik secara semu. Untuk mencegahnya, skrip
`01_resplit_bunch_id.py` melakukan **stratified group split berbasis bunch_id**:
pembagian dilakukan pada tingkat tandan (bukan citra), dengan stratifikasi agar
proporsi kelas tetap seimbang antar-split. Hasilnya:

| Split | Citra | Tandan |
|---|---|---|
| Train | 9.094 | 72 |
| Valid | 769 | 8 |
| Test | 951 | 9 |

Audit kebocoran mengonfirmasi tidak ada irisan bunch_id antar-split
(train↔valid, train↔test, valid↔test seluruhnya **bersih**). Himpunan *test*
bersifat *held-out* dan tidak pernah dilihat selama pelatihan maupun pemilihan
model.

### 3.2.3 Partisi Non-IID Dirichlet

Untuk mensimulasikan heterogenitas data antar-perkebunan, *train set*
dipartisi ke $K$ klien menggunakan distribusi Dirichlet (skrip
`02_dirichlet_partition.py`), dengan $K \in \{2, 4, 8, 12, 16\}$. Untuk tiap
klien diberikan parameter konsentrasi $\alpha$ yang di-*sweep* dari ekstrem
(0,1; satu klien sangat terdominasi sedikit kelas) hingga mendekati seragam
(0,8), sehingga setiap konfigurasi $K$ memiliki spektrum heterogenitas yang
terkontrol dan realistis. Himpunan validasi dan uji bersifat **global** (sama
untuk semua klien) agar metrik dapat dibandingkan lintas-konfigurasi.

Konsekuensi penting dari variasi $K$ adalah perubahan **jumlah sampel per
klien**, yang berdampak langsung pada DP-SGD (lihat 3.5.4):

| $K$ | Rentang sampel/klien |
|---|---|
| 2 | 1.156 – 7.938 |
| 4 | 1.267 – 3.393 |
| 8 | 607 – 1.816 |
| 12 | 419 – 1.079 |
| 16 | 270 – 819 |

Pada $K=16$, klien terkecil hanya memiliki 270 sampel — berada di ambang batas
konvergensi DP-SGD, sehingga $K=16$ diperlakukan sebagai *limit study*.

## 3.3 Model: YOLOv11n dengan GroupNorm

Model dasar adalah **YOLOv11n** (~2,6 juta parameter), diinisialisasi dari bobot
pra-latih COCO. Karena penerapan DP-SGD per-sampel mensyaratkan gradien tiap
sampel dapat dihitung independen, **seluruh lapisan BatchNorm diganti dengan
GroupNorm**. YOLOv11n memuat **81 lapisan BatchNorm2d**; modul
`utils/gn_convert.py` mengganti setiap lapisan tersebut dengan `GroupNorm`
secara *in-place*. Jumlah grup ditetapkan **8** secara *default*, dengan
mekanisme penyesuaian otomatis: bila 8 tidak membagi habis jumlah kanal pada
suatu lapisan, jumlah grup diturunkan hingga membagi habis. Parameter afin
($\gamma, \beta$) dari BatchNorm disalin ke GroupNorm untuk mempertahankan
manfaat inisialisasi pra-latih. Setelah konversi, validasi memastikan **nol**
lapisan BatchNorm tersisa (`count_bn_layers`), sebagai prasyarat agar Opacus
`GradSampleModule` dapat diterapkan.

Konversi GroupNorm berlaku pada **semua** skenario (B1, B2, E1, E2) — termasuk
*baseline* tanpa DP — agar perbandingan bersifat adil (perbedaan hasil murni
berasal dari mekanisme privasi, bukan dari perbedaan arsitektur normalisasi).

## 3.4 Prosedur Pelatihan Lokal

Setiap klien melatih model dengan *Stochastic Gradient Descent* (SGD)
menggunakan hiperparameter berikut (konsisten dengan `FedDPConfig`):

| Hiperparameter | Nilai |
|---|---|
| Optimizer | SGD |
| Learning rate ($lr_0$) | 0,01 |
| Momentum | 0,937 |
| Weight decay | 0,0005 |
| Batch size | 16 |
| Ukuran citra | 640 × 640 |
| Epoch lokal per ronde ($E$) | 2 |

Fungsi kerugian adalah `v8DetectionLoss` bawaan Ultralytics, yang menggabungkan
komponen kotak (CIoU, bobot 7,5), klasifikasi (bobot 0,5), dan Distribution
Focal Loss (bobot 1,5). Pada konfigurasi *partial* (E2), parameter *backbone*
(stage 0–9) **dibekukan** (`requires_grad = False`) sehingga hanya kepala
deteksi yang dilatih — relevan untuk menguji pengaruh dimensi parameter terhadap
DP-SGD (lihat 3.5.4 dan 3.9).

## 3.5 Mekanisme Inti: DP-SGD Per-sampel (Opacus)

Bagian ini merupakan inti metodologis penelitian. Privasi ditanamkan ke dalam
pelatihan lokal tiap klien melalui DP-SGD per-sampel yang diimplementasikan
dengan pustaka **Opacus 1.5.4**.

### 3.5.1 Pembungkusan dengan PrivacyEngine

Ketika DP diaktifkan (`use_dp = True`), model, optimizer, dan *data loader*
dibungkus oleh `PrivacyEngine.make_private(...)` dengan dua parameter privasi
utama: pengali *noise* `noise_multiplier` ($\sigma$) dan ambang pemotongan
`max_grad_norm` ($C$). Pembungkusan ini mengubah model menjadi `GradSampleModule`
yang otomatis menghitung gradien per-sampel.

### 3.5.2 Operasi per Langkah

Untuk setiap *minibatch*, Opacus melakukan tiga operasi sesuai algoritma DP-SGD
[Abadi et al., 2016]:

1. **Gradien per-sampel.** Gradien $g_i$ dihitung untuk setiap sampel $i$ secara
   terpisah (dimungkinkan karena seluruh normalisasi telah GroupNorm).
2. **Pemotongan norma per-sampel.** Tiap $g_i$ dipotong agar
   $\lVert g_i \rVert_2 \le C$, membatasi sensitivitas kontribusi satu sampel.
3. **Penambahan *noise* Gaussian.** *Noise* $\mathcal{N}(0, \sigma^2 C^2)$
   ditambahkan pada jumlah gradien terpotong sebelum dirata-ratakan dan
   dipakai memperbarui bobot.

### 3.5.3 Privacy Accounting

Anggaran privasi $\varepsilon$ dilacak otomatis oleh akuntan *default* Opacus
(**PRV accountant**, lebih erat dari RDP) dan dibaca melalui
`get_epsilon(delta)`. Penelitian melaporkan $\varepsilon$ pada
$\delta = 10^{-5}$ **tetap** untuk semua $K$. Pilihan ini valid dan konservatif:
klien terkecil (K=16) memiliki 270 sampel sehingga $\delta = 10^{-5} \ll 1/270
\approx 3{,}7\times 10^{-3}$, memenuhi syarat $\delta < 1/n_k$ pada seluruh
konfigurasi; sekaligus memudahkan perbandingan $\varepsilon$ lintas-$K$ karena
$\delta$ seragam. Ambang pemotongan awal ditetapkan $C = 1{,}0$.

### 3.5.4 Implikasi Jumlah Klien terhadap Privasi-Utilitas

Karena *noise* DP-SGD bersifat per-sampel dan ber-*subsampling*, jumlah sampel
per klien menentukan rasio sinyal terhadap *noise* pada pelatihan lokal. Saat
$K$ membesar pada dataset berukuran tetap, sampel per klien menyusut sehingga
*noise* dapat mendominasi sinyal gradien lokal sebelum agregasi global
memperbaikinya. Hal ini melandasi hipotesis **H2-K**: pada DP-SGD per-sampel,
$K$ besar cenderung **memperburuk** utilitas — arah yang berkebalikan dengan
DP-FedAvg level-klien. Konfigurasi *partial* (E2, hanya kepala deteksi yang
dilatih, ~0,2 juta parameter) menguji hipotesis terkait bahwa **dimensi
parameter** yang diberi *noise* memengaruhi utilitas [Tramèr & Boneh, 2021]:
semakin sedikit parameter yang di-*noise*, semakin kecil akumulasi *noise*
relatif terhadap sinyal.

## 3.6 Agregasi: Federated Averaging

Setelah seluruh klien menyelesaikan pelatihan lokal pada satu ronde, server
mengagregasi parameter melalui **FedAvg** terbobot ukuran *shard*
(`fedavg(...)`):

$$
w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n}\, w_t^k .
$$

Tensor bobot dijumlahkan secara berbobot lalu dikembalikan ke tipe data
aslinya; entri non-tensor (mis. *buffer* konfigurasi) diwariskan langsung dari
klien pertama. Proses pelatihan-agregasi diulang selama **$T = 5$ ronde
komunikasi**. Pada setiap ronde, model global dievaluasi pada himpunan validasi
global, dan parameter dengan **mAP@0.5 tertinggi** disimpan sebagai *best.pt*.

## 3.7 Explainable AI: Grad-CAM++

Untuk menilai apakah penerapan DP memengaruhi *kualitas alasan* model — bukan
sekadar akurasi numerik — digunakan **Grad-CAM++** [Chattopadhyay et al., 2018].
Lapisan target adalah lapisan konvolusi terakhir sebelum kepala deteksi, yang
memuat representasi semantik paling kaya. *Heatmap* yang dihasilkan menyoroti
wilayah citra yang paling memengaruhi prediksi. Kualitas penjelasan diukur
secara kuantitatif melalui:

- **Average Drop (AD)**: rata-rata penurunan keyakinan model ketika masukan
  dibatasi pada wilayah sorotan; nilai yang tepat menandakan wilayah tersebut
  memang penting.
- **Faithfulness/Region Retention (FRR)**: seberapa besar keyakinan
  dipertahankan ketika hanya wilayah penting yang disisakan.

Perbandingan metrik XAI antara model *baseline* dan model DP-SGD menunjukkan
apakah model privat tetap "melihat" buah pada lokasi yang benar.

## 3.8 Metrik Evaluasi

Evaluasi mencakup tiga dimensi:

**Utilitas (deteksi):** dievaluasi pada himpunan global menggunakan
`yolo.val(...)`:
- mAP@0.5 (metrik utama),
- mAP@0.5:0.95,
- Precision (mp) dan Recall (mr).

**Privasi:** $\varepsilon$ via PRV accountant pada $\delta = 10^{-5}$
(lihat 3.5.3).

**Penjelasan (XAI):** Average Drop dan FRR (lihat 3.7).

Untuk klasifikasi hasil yang konsisten lintas $K$ dan $\sigma$, digunakan
**ambang operasional** berikut:

| Kategori | mAP@0.5 |
|---|---|
| *collapsed* | < 0,05 |
| *degraded* | 0,05 – < 0,70 |
| *acceptable* | 0,70 – < 0,90 |
| *excellent* | ≥ 0,90 |

## 3.9 Skenario Eksperimen

Eksperimen disusun atas *baseline*, eksperimen DP-SGD utama, dan uji
ketangguhan. Demi efisiensi, digunakan **strategi dua fase**: Fase 1
mengeksplorasi seluruh *grid* dengan satu *seed* untuk menemukan zona menarik;
Fase 2 mengonfirmasi konfigurasi menjanjikan dengan tiga *seed* (mean ± std).

**Baseline (tanpa DP):**

| ID | Setup | Tujuan |
|---|---|---|
| **B1** | Centralized YOLOv11n-GN, tanpa DP | *Upper bound* utilitas |
| **B2** | Federated YOLOv11n-GN, tanpa DP, $K$ bervariasi, Non-IID | Mengukur "biaya FL" murni |
| **B3** | Federated + DP-FedAvg (reuse hasil terdahulu yang *collapse*) | Pembanding mekanisme DP |

**Eksperimen DP-SGD (per-sampel via Opacus):**

| ID | Setup | Sweep | Tujuan |
|---|---|---|---|
| **E1** | Full DP-SGD (semua ~2,6 juta param) | $\sigma \in \{0{,}5;\ 1{,}0;\ 1{,}5;\ 2{,}0;\ 3{,}0\}$, $K \in \{2,4,8,12,16\}$ | Kurva privasi-utilitas utama |
| **E2** | Partial DP-SGD (backbone beku, ~0,2 juta param kepala) | $\sigma$ & $K$ sama | Menunjukkan pengaruh dimensi parameter [Tramèr & Boneh, 2021] |

**Ketangguhan (robustness):**

| ID | Setup | Tujuan |
|---|---|---|
| **R1** | 3 *seed* × subset menjanjikan (B2, E1/E2 pada $\sigma$ terbaik & terburuk-layak) | Robustness statistik |

Rentang $\sigma$ dipilih untuk meng-*cover* $\varepsilon$ bermakna dari sekitar
1 (privasi ketat) hingga 10+ (longgar), dengan target *sweet spot*
$\varepsilon \approx 4$–8. *Baseline* B1 yang telah dijalankan menghasilkan
mAP@0.5 = 0,787 sebagai *upper bound* acuan Bab 4.

## 3.10 Lingkungan Implementasi

| Komponen | Spesifikasi |
|---|---|
| Hardware (training) | Workstation GPU NVIDIA RTX 4080 |
| Hardware (deployment) | VPS/edge CPU |
| OS | Windows 11 + miniconda env `fedx` |
| Deep learning | PyTorch 2.5.1 + CUDA 12.1 |
| Deteksi objek | Ultralytics 8.4.51 |
| Privasi | **Opacus 1.5.4** (PRV accountant, GradSampleModule via functorch) |

Orkestrasi seluruh *grid* dilakukan oleh `run_full_grid.py`, sementara skrip
per-skenario (`train_b1_centralized.py`, `train_b2_fl.py`,
`train_e1_fl_dp_sgd_full.py`, `train_e2_fl_dp_sgd_partial.py`) memanggil modul
inti `utils/fl_dp_loop.py`.

## 3.11 Deployment Inference (Docker Blueprint)

Tahap akhir menyajikan model terlatih (`best.pt`) untuk penggunaan lapangan.
Artefak deployment berada pada `thesis_rebuild/deploy/`, terdiri atas
`Dockerfile` (*image* CPU-only: Torch CPU + Ultralytics) dan `predict.py`
(inferensi *batch* via CLI maupun *endpoint* HTTP). Sesuai penegasan pada 3.1,
*containerization* hanya dipakai untuk **inference**, bukan pelatihan/arsitektur
FL.

`Dockerfile` berperan sebagai **cetak biru (*blueprint*) reproducible** yang
menjamin model dapat dijalankan ulang di lingkungan mana pun tanpa konflik
dependensi — bukti *deployment-readiness*. Membangun dan menjalankan *container*
bersifat opsional (nilai tambah demonstratif), sementara *Dockerfile* itu
sendiri telah sah sebagai cetak biru. *Image* dirancang **CPU-only** karena
target deployment lapangan (mini-PC pabrik, VPS murah) umumnya tanpa GPU,
ukuran *image* jauh lebih kecil, dan YOLOv11n cukup ringan untuk inferensi CPU
pada kebutuhan non-*realtime*.
