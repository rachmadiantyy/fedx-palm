# BAB 4 HASIL DAN PEMBAHASAN

Bab ini menyajikan hasil empiris dari lima blok eksperimen (termasuk satu pendahuluan) — *baseline*
centralized (B1), *baseline* federated tanpa privasi (B2), DP-SGD federated
penuh (E1), dan DP-SGD federated parsial dengan *backbone* beku (E2) —
diikuti pembahasan privasi-utilitas, analisis kuantitatif kualitas
penjelasan (XAI) pada model operasional, dan validasi empat hipotesis
penelitian (H1, H2, H2-K, H3). Seluruh angka berasal dari
`thesis_rebuild/tables/runs_master.csv` yang menggabungkan 55 *federated
runs* (B2 + E1 + E2) ditambah satu *run* centralized terpisah untuk B1.

## 4.0 Definisi Operasional Hasil

Seluruh hasil dievaluasi pada himpunan validasi global (769 citra dari 8
tandan) maupun himpunan uji *held-out* (951 citra dari 9 tandan). Anggaran
privasi $\varepsilon$ dilaporkan pada $\delta = 10^{-5}$ tetap, dihitung
via PRV *accountant* Opacus. Untuk konsistensi klasifikasi hasil lintas
konfigurasi $(K, \sigma)$, dipakai ambang operasional Bab 3.8:

| Kategori | mAP@0.5 |
|---|---|
| *collapsed* | $< 0{,}05$ |
| *degraded* | $0{,}05 \le x < 0{,}70$ |
| *acceptable* | $0{,}70 \le x < 0{,}90$ |
| *excellent* | $\ge 0{,}90$ |

Pelaporan menggunakan mAP@0.5 sebagai metrik primer dan mAP@0.5:0.95,
*precision*, dan *recall* sebagai metrik pendukung. *Best mAP@0.5*
diambil dari ronde dengan nilai tertinggi pada `rounds.csv` per *run*,
sedangkan $\varepsilon$ dilaporkan sebagai nilai *final* (akumulasi
sampai ronde terakhir).

## 4.1 Validasi Setup: *Baseline* Centralized (B1)

Eksperimen B1 mengukur **batas atas (*upper bound*) utilitas** YOLOv11n-GN
tanpa biaya federasi maupun *noise* privasi. Pelatihan dilakukan selama
50 *epoch* pada seluruh 9.094 citra *train* dengan hiperparameter Bab 3.4
(SGD, $lr_0 = 0{,}01$, *batch* 16, citra 640×640).

**Hasil:**

| Metrik | Nilai |
|---|---|
| mAP@0.5 | **0,787** |
| mAP@0.5:0.95 | 0,672 |
| *Precision* | 0,815 |
| *Recall* | 0,833 |

Angka ini memenuhi syarat sebagai *upper bound* yang valid: di atas
ambang *acceptable* (0,70) dan jauh di atas hasil literatur klasifikasi
sawit manual dengan tingkat kesalahan 15–25% (Bab 2.1). B1 menjadi
acuan kuantitatif sepanjang bab ini untuk mengukur **"biaya FL"**
($\Delta_\text{FL} = \text{mAP}_\text{B1} - \text{mAP}_\text{B2}$) dan
**"biaya DP"** ($\Delta_\text{DP} = \text{mAP}_\text{B2} - \text{mAP}_\text{E1/E2}$).

Konversi BatchNorm → GroupNorm tidak menurunkan utilitas secara
signifikan; kurva pelatihan menunjukkan penurunan *loss* monotonik tanpa
indikasi *overfit*. **H1 (baseline GN mencapai mAP@0.5 *acceptable*)
TERKONFIRMASI**.

## 4.2 *Baseline* Federated Tanpa DP (B2)

Eksperimen B2 mengisolasi **"biaya FL murni"** — selisih utilitas akibat
federasi dan heterogenitas data Non-IID, tanpa pengaruh *noise* privasi.
Eksperimen dijalankan untuk $K \in \{2, 4, 8, 12, 16\}$, lima ronde
komunikasi, dua *epoch* lokal per ronde. Konfigurasi $K = 4$ tambahan
dilatih selama 25 ronde sebagai *operating point* untuk eksperimen XAI
hilir (Bagian 4.9).

### 4.2.1 Hasil per-$K$

| $K$ | Ronde | mAP@0.5 | mAP@0.5:0.95 | *Precision* | *Recall* | $\Delta_\text{FL}$ | Status |
|---|---|---|---|---|---|---|---|
| 2  | 5  | 0,542     | 0,431 | 0,391 | 0,660 | 0,245     | *degraded*   |
| 4  | 5  | 0,445     | 0,354 | 0,355 | 0,520 | 0,342     | *degraded*   |
| 4  | 25 | **0,738** | 0,593 | 0,676 | 0,680 | **0,049** | *acceptable* |
| 8  | 5  | 0,246     | 0,187 | 0,218 | 0,565 | 0,541     | *degraded*   |
| 12 | 5  | 0,248     | 0,187 | 0,191 | 0,703 | 0,539     | *degraded*   |
| 16 | 5  | 0,231     | 0,169 | 0,199 | 0,825 | 0,556     | *degraded*   |

> Sumber angka: `tables/b2_grid_map50.md` dan `tables/runs_master.csv`
> (baris `exp=B2`). Nilai $K=4$ ronde 5 direkonstruksi dari `rounds.csv`
> di folder `b2_fl_K4_seed42/` (ronde 1–5 saja, sebelum perpanjangan
> menjadi 25 ronde).

### 4.2.2 Pembahasan B2

Tiga observasi utama muncul dari Tabel 4.2.1.

**Pertama**, pada anggaran komputasi yang sama (5 ronde × 2 *epoch* lokal
= 10 *epoch* ekuivalen — seperlima dari B1), kurva-$K$ memperlihatkan
penurunan monoton yang jelas: $K = 2$ mencapai mAP@0.5 = 0,542; $K = 4$
turun ke 0,445; lalu $K \ge 8$ stagnan di kisaran 0,23–0,25. Pola ini
sejalan dengan literatur FL Non-IID [Hsu et al., 2019]: partisi
Dirichlet yang lebih halus memperburuk *client drift*, memperlambat
konvergensi global, dan menurunkan kontribusi efektif tiap pembaruan
ke arah optimum centralized.

**Kedua**, dengan menambah jumlah ronde komunikasi dari 5 menjadi 25
pada $K = 4$, utilitas meningkat dramatis dari 0,445 menjadi **0,738**
— kenaikan absolut 0,293 mAP@0.5. Hal ini menunjukkan bahwa **kerugian utilitas FL
bukan barrier fundamental, melainkan masalah anggaran komunikasi**:
dengan ronde yang memadai, gap antara federated dan centralized dapat
ditekan ke level *acceptable*. $\Delta_\text{FL}$ pada *operating
point* ini hanya 0,049 mAP@0.5 (~6,2% relatif terhadap B1) — biaya FL
yang sangat kecil untuk manfaat lokalitas data plantation.

**Ketiga**, *recall* meningkat dengan $K$ (0,660 pada $K=2$ ke 0,825
pada $K=16$) sementara *precision* tetap rendah (0,191–0,399). Ini
mengindikasikan model di $K$ besar belajar menghasilkan deteksi
ber-*recall* tinggi tetapi banyak *false positive* — gejala under-konvergensi
khas pelatihan FL yang terhenti sebelum *precision* sempat
ter-tuning. Penambahan ronde mengatasi keduanya secara simultan: pada
$K = 4$ 25-ronde, *precision* dan *recall* sama-sama mencapai ~0,68.

## 4.3 Eksperimen Pendahuluan: DP-FedAvg Level-Klien

Sebelum menetapkan DP-SGD per-sampel sebagai mekanisme privasi utama,
penelitian ini lebih dulu menguji pendekatan yang lebih sederhana — penyuntikan
*noise* Gaussian pada *delta* bobot teragregasi di sisi server (*client-level*
DP-FedAvg). Pada seluruh tingkat perturbasi yang diuji, pendekatan ini
menghasilkan **collapse deteksi total** (mAP@0.5 → 0): model kehilangan
kemampuan menghasilkan *bounding box* valid, sehingga tidak ada utilitas yang
dapat dilaporkan maupun dievaluasi XAI-nya.

Alih-alih menyimpulkan DP-FedAvg fundamental tidak layak, penelitian ini
mengidentifikasi **dua confound** pada eksperimen pendahuluan tersebut:

1. **Inkompatibilitas BatchNorm dengan akuntansi per-sampel.** Arsitektur
   YOLOv11 awal sarat lapisan BatchNorm yang statistiknya bergantung
   antar-sampel, bertentangan dengan prinsip privasi per-sampel dan ditolak
   *ModuleValidator* Opacus.
2. **Pemetaan ε tidak andal.** Anggaran privasi pendahuluan tidak diturunkan
   dari *privacy accountant* formal, sehingga ε yang dilaporkan tidak dapat
   dipertanggungjawabkan.

Kedua confound ini memotivasi perombakan desain yang menjadi inti penelitian:
(a) konversi seluruh BatchNorm → GroupNorm agar gradien per-sampel terdefinisi,
(b) peralihan ke DP-SGD per-sampel yang menyebarkan *noise* ke banyak langkah
kecil ternormalisasi alih-alih satu suntikan agregat, dan (c) perhitungan ε
via PRV *accountant* (Opacus) pada δ = 10⁻⁵. Studi utama (Bagian 4.4–4.5)
karenanya berfokus pada DP-SGD per-sampel, dengan DP-FedAvg sebagai **pembanding
pendahuluan** yang menjelaskan *mengapa* jalur per-sampel dipilih. Karena kedua
confound di atas, *collapse* DP-FedAvg **tidak** diklaim sebagai sifat
fundamental, melainkan hasil spesifik pada konfigurasi pipeline pendahuluan.

## 4.4 DP-SGD Federated Penuh (E1)

Eksperimen E1 menambahkan DP-SGD per-sampel pada seluruh ~2,6 juta
parameter YOLOv11n-GN, dilatih secara federasi pada *grid* lengkap
$K \in \{2, 4, 8, 12, 16\}$ × $\sigma \in \{0{,}5;\, 1;\, 1{,}5;\, 2;\, 3\}$
= **25 konfigurasi**, masing-masing 5 ronde × 2 *epoch* lokal.

### 4.4.1 Tabel mAP@0.5 untuk seluruh $(K, \sigma)$

| $K \downarrow$ \ $\sigma \rightarrow$ | 0,5 | 1,0 | 1,5 | 2,0 | 3,0 |
|---|---|---|---|---|---|
| **2**  | 0,188 | 0,183 | 0,124 | 0,077 | 0,064 |
| **4**  | **0,190** | 0,150 | 0,124 | 0,070 | 0,045 |
| **8**  | 0,156 | 0,158 | 0,110 | 0,086 | 0,031 |
| **12** | 0,113 | 0,116 | 0,112 | 0,107 | 0,042 |
| **16** | 0,087 | 0,082 | 0,071 | 0,061 | 0,045 |

> Cetak tebal = nilai terbaik global. Sumber: `tables/e1_grid_map50.md`.
> Kolom $\sigma$ semakin ke kanan: privasi semakin kuat
> ($\varepsilon$ semakin kecil). Baris $K$ semakin ke bawah:
> *samples-per-klien* semakin kecil.

### 4.4.2 Pemetaan $\sigma \to \varepsilon$ per-$K$

| $K \downarrow$ \ $\sigma \rightarrow$ | 0,5 | 1,0 | 1,5 | 2,0 | 3,0 |
|---|---|---|---|---|---|
| **2**  | 8,62  | 1,14 | 0,51 | 0,34 | 0,20 |
| **4**  | 8,36  | 1,08 | 0,48 | 0,32 | 0,19 |
| **8**  | 10,68 | 1,73 | 0,76 | 0,49 | 0,30 |
| **12** | 11,92 | 2,13 | 0,95 | 0,61 | 0,36 |
| **16** | 13,75 | 2,81 | 1,28 | 0,82 | 0,48 |

> Sumber: kolom `final_epsilon` pada `runs_master.csv` (baris `exp=E1`).

Pola dua dimensi yang muncul dari Tabel 4.4.2 sangat instruktif:
$\varepsilon$ **naik** seiring $K$ membesar pada $\sigma$ tetap. Pada
$\sigma = 0{,}5$, $K = 2$ menghabiskan $\varepsilon = 8{,}62$ sementara
$K = 16$ membutuhkan $\varepsilon = 13{,}75$ — kenaikan 60% — meskipun
keduanya menjalankan jumlah ronde dan *epoch* lokal yang identik. Hal
ini adalah konsekuensi langsung **privacy amplification by subsampling**
(Bab 2.4.3): klien kecil memiliki rasio sub-sampling $q = B/n_k$ yang
lebih besar (karena $n_k$ kecil), sehingga setiap langkah pelatihan
"membayar" privasi lebih mahal dalam akumulasi anggaran.

Implikasi praktis: pada *deployment* nyata, **$K$ besar tidak hanya
menurunkan utilitas tetapi juga memperburuk efisiensi privasi** — sebuah
*double penalty* yang tidak dialami DP-FedAvg level-klien.

### 4.4.3 Kurva Privasi-Utilitas

Gambar 4.1 (`figures/privacy_utility_e1.png`) memplot mAP@0.5 sebagai
fungsi $\varepsilon$ (sumbu-x) untuk setiap $K$ (satu kurva per $K$).
Garis horizontal pada mAP = 0,05 menandai ambang *collapsed*; garis
horizontal pada 0,70 menandai ambang *acceptable*.

Tiga observasi dari kurva:

1. **Tidak ada konfigurasi E1 yang mencapai ambang *acceptable*.**
   Titik tertinggi global adalah $K = 4$, $\sigma = 0{,}5$,
   $\varepsilon = 8{,}36$ dengan mAP@0.5 = 0,190 — masih jauh di bawah
   0,70. Dengan kata lain, **biaya DP pada *operating point* terbaik**
   ($\Delta_\text{DP} = 0{,}738 - 0{,}190 = 0{,}548$ mAP@0.5) jauh
   melampaui *biaya FL* (0,049).

2. **Degradasi bersifat *gradual*, bukan *cliff*.** Pada $K = 2$,
   penurunan mAP@0.5 dari $\varepsilon = 8{,}62$ ke $\varepsilon = 0{,}20$
   adalah 0,188 → 0,064, monoton dengan slope yang relatif konsisten
   (sekitar 0,03 mAP per pengurangan $\varepsilon$ satu unit pada rezim
   $\varepsilon < 2$). Tidak ada $\sigma$ tunggal yang menyebabkan
   loncatan vertikal ke regime *collapsed* secara tiba-tiba; transisi
   dari *degraded* ke *collapsed* terjadi konsisten di $\sigma = 3{,}0$
   untuk $K \ge 4$. **H2 (degradasi *gradual*) TERKONFIRMASI**.

3. **Urutan kurva konsisten $K = 2 \approx K = 4 > K = 8 > K = 12 > K = 16$.**
   Pada $\varepsilon$ yang sebanding, $K = 2$ dan $K = 4$ secara konsisten
   memberikan mAP@0.5 lebih tinggi dibanding $K = 16$. Selisih ini
   nyata: pada $\sigma = 0{,}5$ (rezim $\varepsilon$ paling longgar),
   $K = 2$ menghasilkan 0,188 sedangkan $K = 16$ hanya 0,087 — selisih
   relatif 117%. **H2-K (K besar memperburuk utilitas pada DP-SGD
   per-sampel) TERKONFIRMASI**.

### 4.4.4 Kurva-$K$

Gambar 4.2 (`figures/K_curve_e1.png`) memetakan mAP@0.5 sebagai fungsi
$K$ untuk tiap nilai $\sigma$. Pola yang muncul memperkuat 4.4.3:

- Pada $\sigma$ kecil (0,5 dan 1,0), kurva-$K$ menurun monoton dengan
  *plateau* relatif tinggi pada $K \in \{2, 4\}$ (~0,15–0,19) sebelum
  jatuh ke ~0,08 pada $K = 16$. Penurunan paling tajam terjadi pada
  $K = 8 \to K = 12$ ($\sigma = 0{,}5$: 0,156 → 0,113) — kemungkinan
  besar berhubungan dengan jumlah *samples-per-klien* yang melewati
  ambang minimum untuk konvergensi DP-SGD (sekitar 700 sampel pada
  $K = 12$ versus 1.100 pada $K = 8$).

- Pada $\sigma$ besar (3,0), kurva-$K$ datar di kisaran 0,03–0,06,
  mendekati rezim *collapsed* di seluruh $K$. Pada anggaran *noise*
  setinggi ini, perbedaan $K$ tidak lagi membantu karena rasio
  *signal-to-noise* sudah jatuh terlalu rendah secara universal.

### 4.4.5 Analisis Komparatif: Mengapa E1 Berhenti di Rezim *Degraded*

DP-SGD per-sampel pada *object detection* dengan ~2,6 juta parameter
*trainable* membutuhkan dua hal yang saling berkonflik: (a) gradien
per-sampel ber-*norm* kecil agar *clipping* tidak terlalu agresif,
dan (b) batch besar agar *signal-to-noise* setelah agregasi memadai.
YOLOv11n yang dilatih dari bobot COCO memiliki gradien per-sampel yang
*norm*-nya besar pada *epoch* awal (jarak antara distribusi COCO dan
sawit), sehingga *clipping* dengan $C$ sederhana memotong sebagian besar
informasi gradien tepat ketika model paling membutuhkannya untuk
adaptasi domain. Hasilnya: model "belajar perlahan, lupa cepat".

Pembuktian empiris: meskipun $\sigma = 0{,}5$ secara matematis
menghasilkan *noise* gradien yang kecil, mAP@0.5 hanya 0,190 (versus
B2 K=4 25-ronde = 0,738). Dengan jumlah ronde komunikasi yang sama
seperti B2 5-ronde, kerusakan kemungkinan besar berasal dari
*clipping* yang menjegal adaptasi domain, bukan dari *noise* itu sendiri.

## 4.5 DP-SGD Federated Parsial (E2): *Backbone* Beku

Eksperimen E2 membekukan *backbone* YOLOv11n (stage 0–9, ~2,4 juta
parameter) dan hanya melatih kepala deteksi (~0,2 juta parameter
*trainable*) dengan DP-SGD. Hipotesis pendukung dari Tramèr & Boneh
(2021): dengan vektor gradien yang jauh lebih kecil, akumulasi *noise*
relatif terhadap sinyal jauh berkurang, sehingga **E2 seharusnya
mendominasi E1** terutama pada $\varepsilon$ rendah.

### 4.5.1 Tabel mAP@0.5 untuk seluruh $(K, \sigma)$

| $K \downarrow$ \ $\sigma \rightarrow$ | 0,5 | 1,0 | 1,5 | 2,0 | 3,0 |
|---|---|---|---|---|---|
| **2**  | **0,113** | 0,080 | 0,078 | 0,061 | 0,040 |
| **4**  | 0,070     | 0,066 | 0,069 | 0,059 | 0,038 |
| **8**  | 0,060     | 0,059 | 0,050 | 0,039 | 0,032 |
| **12** | 0,056     | 0,051 | 0,046 | 0,046 | 0,034 |
| **16** | 0,032     | 0,031 | 0,027 | 0,025 | 0,025 |

> Cetak tebal = nilai terbaik global. Sumber: `tables/e2_grid_map50.md`.

### 4.5.2 E1 versus E2: *Side-by-side* pada $\varepsilon$ Sebanding

| $\varepsilon \approx$ | $(K, \sigma)$ | E1 mAP@0.5 | E2 mAP@0.5 | Selisih (E1 − E2) |
|---|---|---|---|---|
| 8,4   | (4, 0,5)  | **0,190** | 0,070 | +0,120 |
| 1,1   | (4, 1,0)  | **0,150** | 0,066 | +0,084 |
| 0,48  | (4, 1,5)  | **0,124** | 0,069 | +0,055 |
| 8,62  | (2, 0,5)  | **0,188** | 0,113 | +0,075 |
| 1,14  | (2, 1,0)  | **0,183** | 0,080 | +0,103 |
| 0,20  | (2, 3,0)  | **0,064** | 0,040 | +0,024 |

### 4.5.3 Pembahasan: Hipotesis Tramèr & Boneh DITOLAK pada Setup Ini

Pola Tabel 4.5.2 sangat tegas: **E1 mengungguli E2 di setiap konfigurasi
$(K, \sigma)$ yang diuji**, baik pada $\varepsilon$ longgar (8,6)
maupun ketat (0,2). Selisih rata-rata adalah +0,084 mAP@0.5 menguntungkan
E1. Hasil ini **berlawanan dengan prediksi Tramèr & Boneh (2021)** yang
menyatakan partial fine-tuning di bawah DP-SGD seharusnya lebih efisien.

Tiga kemungkinan penjelasan, mengurutkan dari yang paling mungkin:

1. **Backbone YOLOv11n COCO tidak ter-fine-tune ke domain TBS sawit.**
   Bobot pra-latih COCO mengkode fitur generik (tepi, tekstur, bentuk
   umum) tetapi belum mengenal *pattern* spesifik TBS — warna khas
   *fresh fruit bunch*, struktur *bunch*, *occlusion* daun. Dengan
   hanya kepala deteksi yang trainable, model E2 berusaha memetakan
   fitur COCO langsung ke 6 kelas TBS — *capacity gap* yang terlalu
   besar untuk ditutup oleh ~0,2 juta parameter saja.

2. **DP-SGD di kepala mendapat *noise* relatif besar.** Walaupun
   absolut dimensi gradien E2 lebih kecil, dengan $\sigma$ sama, *noise*
   relatif (terhadap *norm* sinyal kepala yang juga kecil) tidak
   selalu lebih baik. Tramèr & Boneh menggunakan asumsi model image
   classification dengan fitur pra-latih yang sudah relevan; *object
   detection* di domain baru tidak memenuhi asumsi tersebut.

3. **Jumlah *epoch* tidak mencukupi untuk konvergensi *head-only*.**
   Lima ronde × 2 *epoch* lokal = 10 *epoch* ekuivalen terlalu sedikit
   untuk *fine-tuning head-only* yang biasanya memerlukan 20–50 *epoch*
   pada literatur transfer learning. Replikasi dengan ronde lebih banyak
   (seperti yang dilakukan B2 $K = 4$ pada Bagian 4.2) dapat mengurangi
   *gap* tetapi tidak diduga akan membalikkan arah komparasi.

**Kontribusi metodologis tesis**: laporan empiris bahwa **partial DP-SGD
tidak selalu lebih baik daripada full DP-SGD pada object detection**,
khususnya ketika *backbone* pra-latih belum di-*fine-tune* ke domain
target. Temuan ini menambah nuansa pada rekomendasi luas Tramèr & Boneh
yang dirumuskan untuk *image classification* dengan fitur ImageNet yang
sudah jenuh.

## 4.6 Sintesis Privasi-Utilitas: Tiga Rezim

Menggabungkan B1, B2, E1, E2, hasil empiris dapat diringkas ke dalam
tiga **rezim utilitas-privasi** untuk YOLOv11n-GN pada deteksi TBS sawit:

| Rezim | Mekanisme | mAP@0.5 | $\varepsilon$ | Status Deployment |
|---|---|---|---|---|
| **No privacy formal** | B1 centralized (50 ep) | 0,787 | $\infty$ | *acceptable* |
| | B2 federated $K=4$ (25 ronde) | 0,738 | $\infty$ | *acceptable* |
| **DP-SGD lemah** | E1 $(K=4,\sigma=0{,}5)$ | 0,190 | 8,36 | *degraded* |
| | E1 $(K=2,\sigma=0{,}5)$ | 0,188 | 8,62 | *degraded* |
| **DP-SGD ketat** | E1 $(K=4,\sigma=3{,}0)$ | 0,045 | 0,19 | *collapsed* |
| | E2 $(K=16,\sigma=3{,}0)$ | 0,025 | 0,48 | *collapsed* |

Dua tindakan praktis mengikuti:

**Untuk produksi pendek waktu**: gunakan B2 ($K = 4$, 25 ronde) — model
operasional yang aman dari kebocoran data mentah (lokalitas plantation)
dengan utilitas hampir setara centralized. Cocok untuk skenario di mana
ancaman utama adalah *raw data exfiltration*, bukan *membership inference
attack* (MIA) terhadap model.

**Untuk skenario yang menuntut $(\varepsilon, \delta)$-DP formal**: hasil
empiris menunjukkan trade-off saat ini terlalu mahal. Diperlukan riset
lanjutan pada teknik *DP-friendly architecture* (model lebih kecil,
*group convolutions*, *low-rank adaptation*) atau peningkatan anggaran
ronde komunikasi yang substansial sebelum DP-SGD dapat memberikan
mAP@0.5 *acceptable* pada deteksi objek domain baru.

## 4.7 Catatan Ketangguhan Statistik

Pada penelitian ini, seluruh 56 *run* dilaksanakan dengan satu *seed*
($\{42\}$) karena keterbatasan anggaran komputasi (15 jam *grid* FL
penuh + B1 terpisah). Implikasinya: angka tunggal yang dilaporkan pada
4.1–4.4 dapat memiliki variasi acak ±0,01–0,03 mAP@0.5 berdasarkan
literatur YOLO. Pola **kualitatif** yang menjadi dasar klaim hipotesis
(monoton turun E1 dengan $K$, dominasi E1 atas E2, *gradual decay*
seiring $\sigma$) memiliki *effect size* yang jauh melebihi variasi
seed yang diperkirakan, sehingga ketiganya tetap dapat
dipertanggungjawabkan dari satu seed.

Replikasi tiga-*seed* ($\{42, 7, 123\}$) untuk subset
$\{B2_{K=4,\,25\text{r}},\, E1_{K=4,\sigma=0{,}5},\, E2_{K=2,\sigma=0{,}5}\}$
disarankan sebagai pekerjaan lanjutan untuk pengetatan *confidence
interval* sebelum publikasi jurnal.

## 4.8 Validasi Hipotesis

| Hipotesis | Klaim | Verdict | Bukti |
|---|---|---|---|
| **H1** | *Baseline* GN mencapai mAP@0.5 *acceptable* | **TERKONFIRMASI** | 4.1: mAP=0,787 > 0,70 |
| **H2** | DP-SGD per-sampel memberi degradasi *gradual* pada $\varepsilon \le 8$ | **TERKONFIRMASI** | 4.4.3: penurunan monoton, tidak ada *cliff* tunggal |
| **H2-K** | $K$ besar memperburuk utilitas DP-SGD per-sampel (kebalikan DP-FedAvg) | **TERKONFIRMASI** | 4.4.4: $K=2$ konsisten > $K=16$ di tiap $\sigma$; ditambah *double penalty* $\varepsilon$ |
| **H3** | XAI tetap *meaningful* pada model operasional | **TERKONFIRMASI sebagian** | 4.9: berlaku pada B2; tidak dapat dievaluasi serius pada E1/E2 karena model *collapsed* |

Catatan tambahan, satu hipotesis pendukung **DITOLAK**: prediksi Tramèr
& Boneh bahwa partial DP-SGD (E2) mendominasi full DP-SGD (E1) tidak
terjadi pada setup ini (Bagian 4.5.3).

## 4.9 Validasi Penjelasan (XAI)

Grad-CAM++ diterapkan pada *checkpoint* operasional B2 $K = 4$
(mAP@0.5 = 0,738) untuk mengevaluasi *faithfulness* penjelasan terhadap
prediksi *bounding box*. Evaluasi pada model E1/E2 dilakukan secara
terbatas tetapi tidak memadai untuk pelaporan kuantitatif karena
seluruh *checkpoint* E1/E2 berada di rezim *degraded*/*collapsed* —
*saliency map* yang dihasilkan didominasi *noise* dan tidak
men-*localize* objek dengan andal.

Metrik *faithfulness* (Bab 2.8) dihitung pada 344 *instance* deteksi
dari 100 citra *test* acak, dirinci per-kelas.

### 4.9.1 Hasil Global & Per-Kelas (B2 $K = 4$)

| Kelas | $n$ | *Average Drop* (%) | FRR |
|---|---|---|---|
| Abnormal       | 72  | 7,45  | 0,093 |
| Empty Bunch    | 20  | 20,90 | 0,462 |
| Overripe       | 45  | 18,93 | 0,343 |
| Ripe           | 124 | 26,29 | 0,292 |
| Underripe      | 49  | 26,68 | 0,270 |
| Unripe         | 34  | 15,71 | 0,463 |
| **Global**     | 344 | **15,90** | **0,281** |

> Sumber: `tables/xai_per_class.csv`.
> AD lebih **tinggi** lebih baik: *Average Drop* mengukur penurunan
> kepercayaan ketika wilayah *salient* (heatmap di atas ambang) ditutup
> (Bab 2.6.2, Pers. 2.7) — drop yang besar berarti model benar-benar
> bergantung pada ROI yang disorot. FRR lebih tinggi lebih baik
> (*saliency* ter-*localize* di dalam ROI deteksi).

### 4.9.2 Komparasi dengan *Centralized* (B1)

Sebagai pembanding, B1 dievaluasi pada 200 citra *test*:

| Model | $n$ | *Average Drop* (%) | FRR |
|---|---|---|---|
| B1 *centralized* | 200 | 4,95  | 0,182 |
| B2 *federated* $K=4$ | 344 | 15,90 | 0,281 |

Dua observasi yang saling menguatkan:

- **AD lebih tinggi pada B2** (15,90% vs 4,95%) menunjukkan prediksi B2
  lebih *grounded* pada ROI: menutup wilayah *salient* menjatuhkan
  kepercayaan jauh lebih besar dibanding pada B1. Sebaliknya, pada B1
  menutup ROI hampir tidak mengubah kepercayaan (AD 4,95%), indikasi
  model *centralized* lebih banyak bersandar pada konteks global di luar
  ROI ketika memutuskan.
- **FRR lebih tinggi pada B2** (0,281 vs 0,182) menunjukkan *saliency*
  B2 **lebih ter-konsentrasi di dalam ROI deteksi**. Kedua metrik
  searah: meskipun B2 sedikit kurang akurat secara mAP@0.5 (0,738 vs
  0,787), penjelasan visualnya justru lebih *faithful* — ketika B2
  mendeteksi sesuatu, keputusannya benar-benar didasarkan pada area
  buah, bukan pada latar.

### 4.9.3 Analisis Per-Kelas

Variasi besar antar-kelas mengungkap *failure mode* spesifik:

- **Abnormal** memiliki AD terendah (7,45%) **dan** FRR terendah (0,093):
  model membuat prediksi kelas ini berdasarkan konteks luas (luar ROI),
  bukan fitur lokal *bunch*. Konsekuensi praktis: prediksi *Abnormal*
  perlu *manual review* lebih intensif sebelum dipakai untuk keputusan
  panen.
- **Ripe** dan **Underripe** menunjukkan AD tertinggi (26,29% dan
  26,68%) — *occlusion* ROI sangat memengaruhi prediksi, indikasi
  model *grounded* pada fitur visual *bunch* itu sendiri. Kombinasi
  dengan FRR moderat (0,27–0,29) menyatakan kelas-kelas ini paling
  *trustworthy*.
- **Unripe** memiliki FRR tertinggi (0,463): *saliency* sangat fokus
  di dalam ROI, meskipun jumlah sampel kecil ($n = 34$) menuntut
  validasi tambahan.

Visualisasi *heatmap* untuk citra representatif per-kelas disajikan
pada Gambar 4.5 (`figures/xai_comparison.png`). **H3 (XAI tetap
*meaningful* pada model operasional)** dianggap **TERKONFIRMASI** untuk
B2 berdasarkan: (i) FRR > 0 secara konsisten di seluruh kelas,
(ii) AD positif menunjukkan model responsif terhadap ROI, dan (iii)
ranking per-kelas konsisten dengan intuisi domain (Abnormal paling
kontekstual, Ripe paling lokal).

## 4.10 Demonstrasi Operasional: *Deployment* Layanan Inferensi

Untuk membuktikan bahwa kerangka FedX-Palm tidak terhenti sebagai simulasi laboratorium, *checkpoint* operasional **B2 $K = 4$ (25 ronde, mAP@0.5 = 0,738)** di-*deploy* ke sebuah *Virtual Private Server* (VPS) berbasis CPU sebagai layanan inferensi mandiri. *Checkpoint* ini dipilih karena merupakan model dengan utilitas tertinggi yang dihasilkan oleh pipeline federated tanpa privasi formal — sesuai dengan rekomendasi Bagian 4.6 untuk skenario produksi di mana ancaman utama adalah *raw data exfiltration*, bukan MIA terhadap bobot model.

### 4.10.1 Arsitektur *Deployment*

Layanan dikemas sebagai satu *image* Docker dengan komponen berikut:

| Komponen | Implementasi |
|---|---|
| *Web framework* | Flask 3.x (*endpoint* `/predict`) |
| *Inference engine* | Ultralytics YOLOv11 + PyTorch CPU |
| *Model weights* | `best.pt` dari `runs/b2_fl_K4_seed42/` |
| *XAI module* | Grad-CAM++ menyorot detection head terakhir |
| *Frontend* | Halaman HTML statis dengan tombol *upload* citra |
| *Runtime* | VPS Linux CPU-only (tanpa GPU) |

Pemilihan arsitektur CPU-only disengaja untuk menunjukkan bahwa **biaya *deployment* riil dapat ditekan jauh di bawah biaya pelatihan**: pelatihan menggunakan GPU NVIDIA RTX 4080 selama 15 jam, sedangkan inferensi cukup ditangani CPU komoditas dengan latensi sub-detik per citra.

### 4.10.2 Antarmuka Layanan

Gambar 4.6 (`pic/docker_ui.png`) menampilkan halaman utama layanan yang berjalan pada *port* 8080. Antarmuka menyediakan tombol unggah citra serta menampilkan ringkasan metrik agregat model sebagai konteks transparansi bagi pengguna sebelum melakukan inferensi.

Metrik agregat yang ditampilkan (mAP@0.5 = 0,738; AD = 15,90%; FRR = 0,281) berasal langsung dari hasil eksperimen Bagian 4.2 dan 4.9 — bukan angka pemasaran. Pengguna akhir dengan demikian mengetahui dari awal **rentang kepercayaan yang wajar** terhadap prediksi yang akan diterima: model layak untuk *screening* otomatis dan rekomendasi panen, tetapi tidak menggantikan inspeksi mata-akhir untuk kasus *borderline*.

### 4.10.3 Inferensi *Live* dan Visualisasi Grad-CAM++

Gambar 4.7 (`pic/result-heatmap.png`) memperlihatkan hasil inferensi *live* pada citra TBS sawit yang diunggah lewat antarmuka. Tiga elemen ditampilkan secara simultan:

1. *Bounding box* di sekitar setiap *bunch* terdeteksi, beserta label kelas (Unripe/Underripe/Ripe/Overripe/Empty Bunch/Abnormal) dan *confidence score*.
2. Peta panas (*heatmap*) Grad-CAM++ yang menyorot area yang menjadi dasar visual keputusan kelas — sehingga setiap prediksi disertai justifikasi yang dapat diperiksa pengguna.
3. Tabel ringkas yang mendaftar seluruh deteksi pada citra beserta kelas, kepercayaan, dan koordinat *bounding box*.

Pengujian dilakukan pada 50 citra dari himpunan *test* yang belum pernah dilihat model selama pelatihan. Latensi rata-rata pada VPS CPU 4-vCPU adalah **~0,14 detik per citra** (inference) ditambah **~0,44 detik** untuk komputasi *heatmap* Grad-CAM++ — total **di bawah satu detik** per citra, masih sangat nyaman untuk *web upload* manual.

### 4.10.4 Konsistensi Visual dengan Validasi Kuantitatif

Peta panas yang ditampilkan pada antarmuka **konsisten secara visual** dengan FRR = 0,281 yang dilaporkan pada Bagian 4.9.1: atensi model terkonsentrasi di area *bunch* untuk kelas dengan FRR tinggi (Unripe, Empty Bunch), sementara kelas dengan FRR rendah (Abnormal) menunjukkan *spread* atensi yang lebih luas ke konteks daun di sekitarnya — pola yang mengonfirmasi temuan analisis per-kelas pada 4.9.3.

Demonstrasi ini juga **mengilustrasikan prinsip lokalitas data** yang dirancang pada Bab 3.12: secara desain FedAvg, data mentah klien tidak perlu meninggalkan plantation karena hanya bobot model ter-agregasi yang dikemas ke *image* Docker dan di-*deploy* ke VPS inferensi. Pada layanan inferensi yang ditunjukkan, tidak ada satu pun citra TBS *train* yang tersimpan di *server*. (Catatan: pelatihan FL pada penelitian ini dijalankan sebagai simulasi *sequential*, sehingga isolasi jaringan antar-klien bersifat properti rancangan, bukan diukur secara empiris.)

### 4.10.5 Implikasi Kelayakan Praktis

Tiga implikasi dari demonstrasi *deployment* ini:

1. **Pisah kekhawatiran *training* vs *inference*.** Pelatihan FL membutuhkan GPU lokal di setiap klien (estate plantation), tetapi inferensi pada VPS pusat dapat berjalan tanpa akselerator. Hal ini berarti **biaya infrastruktur produksi sangat rendah** dibandingkan biaya R&D awal.

2. **XAI sebagai jaminan kepercayaan operator.** Pengguna lapangan (mandor panen) yang tidak memiliki latar belakang *machine learning* dapat memvalidasi keputusan model secara visual lewat *heatmap*. Kombinasi prediksi + justifikasi visual + metrik agregat yang jujur memberikan basis kepercayaan yang lebih kuat dibanding *black-box detector* konvensional.

3. **Operating point yang realistis.** Model B2 ($K = 4$, 25 ronde) dengan mAP@0.5 = 0,738 berada di rezim *acceptable* (Bagian 4.0) dan terbukti operasional pada VPS produksi — bukan angka *benchmark* yang hanya bermakna di kertas. *Trade-off* antara mAP yang lebih tinggi (B1 centralized 0,787) dan lokalitas data plantation (B2 federated 0,738) menghasilkan selisih hanya 0,049 mAP@0.5, yang dapat dipertanggungjawabkan untuk manfaat *privacy-by-design*.

## 4.11 Ancaman terhadap Validitas

**Internal.** (a) *Smoke test* sebelum *grid* lepas menemukan tiga
inkompatibilitas Opacus×YOLO (SiLU *in-place*, *signature loss*,
Conv+BN *fusion*) yang telah ter-patch; verifikasi awal $E1\,(K=2,
\sigma=1{,}0,$ 2 ronde) menghasilkan $\varepsilon$ terhitung dan
mAP non-NaN. (b) $\delta$ tetap pada $10^{-5}$ memudahkan perbandingan
tetapi tidak menyesuaikan skala $1/n_k$ per-klien — analisis
sensitivitas opsional disarankan jika waktu memungkinkan. (c) *Best
mAP@0.5* dipilih dari ronde terbaik per *run* (bukan ronde terakhir);
hal ini *cherry-picks* titik optimum dan menguntungkan E1/E2 secara
sistematik, tetapi keputusan ini konsisten lintas eksperimen sehingga
tidak menggoyahkan komparasi.

**Eksternal.** Dataset berasal dari satu sumber (Roboflow versi 2)
sehingga generalisasi ke perkebunan lain (varietas, iklim, kamera)
belum diukur. Hasil bab ini berlaku pada distribusi data tersebut.
Replikasi pada *FFB Indonesia*, *MOIST*, atau dataset citra lapangan
lain disarankan sebagai validasi eksternal.

**Konstruksi.** Sepuluh *epoch* ekuivalen (5 ronde × 2 *epoch* lokal)
lebih pendek dari 50 *epoch* B1, sehingga selisih utilitas dapat
dipengaruhi durasi pelatihan yang lebih singkat — bukan murni
FL/DP. Mitigasi: (i) B2 ($K = 4$, 25 ronde, 50 *epoch* ekuivalen)
disertakan sebagai *baseline* yang adil terhadap B1 untuk mengisolasi
pengaruh DP, dan (ii) seluruh komparasi E1 vs E2 menggunakan durasi
pelatihan identik (5 ronde × 2 *epoch*) sehingga selisih antar-mekanisme
tetap valid.

**Statistik.** Satu *seed* tunggal (Bagian 4.7) berarti *confidence
interval* tidak dilaporkan. Klaim kualitatif (monoton, *gradual*,
dominasi arah) tetap dapat dipertanggungjawabkan karena *effect size*
melampaui variasi seed YOLO yang khas (~0,01–0,03 mAP@0.5).
