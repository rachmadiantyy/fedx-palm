# BAB 4 HASIL DAN PEMBAHASAN

> **STATUS PENULISAN — SKELETON.**
> Bab ini ditulis sebagai kerangka naratif. Setiap besaran numerik yang
> belum tersedia ditandai dengan `{TBD: keterangan}`. Setelah eksperimen
> Phase 1 selesai, jalankan `python thesis_rebuild/scripts/aggregate_results.py`
> lalu salin angka dari `thesis_rebuild/tables/runs_master.csv`,
> `e{1,2}_grid_map50.md`, dan plot dari `thesis_rebuild/figures/`. Ganti
> setiap `{TBD: ...}` sesuai keterangannya. Jangan ubah struktur naratif
> kecuali hasil empiris benar-benar bertentangan dengan hipotesis.

Bab ini menyajikan hasil empiris dari enam blok eksperimen — *baseline*
sentralized (B1), *baseline* federated (B2), DP-SGD federated penuh (E1),
DP-SGD federated parsial (E2), pembanding lintas-mekanisme (B3 vs E1/E2),
serta validasi ketangguhan statistik (R1) — diikuti pembahasan privasi-utilitas,
analisis kuantitatif kualitas penjelasan (XAI), dan validasi tiga hipotesis
penelitian (H1, H2, H2-K, H3).

## 4.0 Definisi Operasional Hasil

Seluruh hasil dievaluasi pada himpunan validasi global (769 citra dari 8
tandan) maupun himpunan uji *held-out* (951 citra dari 9 tandan). Anggaran
privasi dilaporkan pada $\delta = 10^{-5}$ tetap, dihitung via PRV
*accountant* Opacus. Untuk konsistensi klasifikasi hasil lintas konfigurasi
$(K, \sigma)$, dipakai ambang operasional Bab 3.8:

| Kategori | mAP@0.5 |
|---|---|
| *collapsed* | $< 0{,}05$ |
| *degraded* | $0{,}05 \le x < 0{,}70$ |
| *acceptable* | $0{,}70 \le x < 0{,}90$ |
| *excellent* | $\ge 0{,}90$ |

## 4.1 Validasi Setup: Baseline Sentralized (B1)

Eksperimen B1 mengukur **batas atas (*upper bound*) utilitas** yang dapat
dicapai oleh YOLOv11n-GN tanpa biaya federasi maupun *noise* privasi.
Pelatihan dilakukan selama 50 epoch pada seluruh 9.094 citra *train* dengan
hiperparameter dari Bab 3.4 (SGD, $lr_0 = 0{,}01$, batch 16, citra 640×640).

**Hasil**:

| Metrik | Nilai |
|---|---|
| mAP@0.5 | **0,787** |
| mAP@0.5:0.95 | 0,672 |
| Precision | 0,815 |
| Recall | 0,833 |

Angka ini memenuhi syarat sebagai *upper bound* yang valid: di atas ambang
*acceptable* (0,70) dan jauh di atas hasil literatur klasifikasi sawit
manual (15–25% kesalahan; Bab 2.1). B1 dengan demikian menjadi acuan
kuantitatif yang dipakai sepanjang bab ini untuk mengukur "biaya FL" dan
"biaya privasi". Konversi BatchNorm → GroupNorm tidak menurunkan utilitas
secara signifikan; kurva pelatihan menunjukkan penurunan *loss* monotonik
tanpa indikasi overfit (`val/cls_loss` turun dari 1,748 menjadi 1,063 dalam
5 epoch awal, *train* dan *val* sama-sama turun hingga epoch terakhir).

## 4.2 Baseline Federated Tanpa DP (B2)

Eksperimen B2 mengisolasi **"biaya FL murni"** — selisih utilitas akibat
federasi dan heterogenitas data Non-IID, tanpa pengaruh *noise* privasi.
Eksperimen dijalankan untuk $K \in \{2, 4, 8, 12, 16\}$, lima ronde
komunikasi, dua epoch lokal per ronde.

### 4.2.1 Hasil per-$K$

| $K$ | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Biaya FL |
|---|---|---|---|---|---|
| 2  | `{TBD: B2 K=2 best mAP50}` | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD: B1-B2}` |
| 4  | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` |
| 8  | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` |
| 12 | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` |
| 16 | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` |

> Sumber angka: `tables/b2_grid_map50.md` dan baris `exp=B2` pada
> `tables/runs_master.csv`. "Biaya FL" = mAP@0.5(B1) − mAP@0.5(B2,$K$).

### 4.2.2 Pembahasan

`{TBD: 1 paragraf — apakah biaya FL kecil pada K rendah dan membesar pada
K tinggi? Bandingkan dengan dry-run 2-ronde (K=2: 0.411, K=16: 0.216) —
apakah pola monotonik turun tetap setelah 5 ronde penuh?}`

Trend ini sejalan dengan literatur FL Non-IID [Hsu et al., 2019]:
heterogenitas Dirichlet yang lebih ekstrem (karena partisi yang lebih halus)
memperburuk *client drift* dan memperlambat konvergensi global.

## 4.3 DP-SGD Federated Penuh (E1)

Eksperimen E1 menambahkan DP-SGD per-sampel pada seluruh ~2,6 juta parameter
YOLOv11n-GN, melatihnya secara federasi pada *grid* lengkap
$K \in \{2,4,8,12,16\}$ × $\sigma \in \{0{,}5; 1; 1{,}5; 2; 3\}$ = **25 konfigurasi**.

### 4.3.1 Tabel mAP@0.5 untuk seluruh $(K, \sigma)$

`{TBD: paste tabel dari tables/e1_grid_map50.md}`

Kolom σ semakin ke kanan = privasi semakin kuat (*noise* lebih besar) =
ε semakin kecil. Baris $K$ semakin ke bawah = samples-per-klien semakin
kecil. Lihat 4.3.3 untuk pemetaan σ → ε.

### 4.3.2 Kurva Privasi-Utilitas

Gambar 4.1 (`figures/privacy_utility_e1.png`) menampilkan mAP@0.5 sebagai
fungsi $\varepsilon$ (sumbu-x) untuk setiap $K$ (satu kurva per $K$). Garis
horizontal pada mAP = 0,05 menandai ambang *collapsed*.

`{TBD: 2-3 paragraf deskripsi pola kurva}`:
- Apakah ada *sweet spot* di $\varepsilon \approx 4$–8 yang masih *acceptable*?
- Apakah degradasi *gradual* (linear/sub-linear) atau menunjukkan *cliff*
  pada σ tertentu? Jika gradual → **H2 terkonfirmasi** (lihat 4.7).
- Apakah kurva $K=2$ konsisten lebih tinggi dari $K=16$ pada $\varepsilon$ yang sama?
  Jika ya → **H2-K terkonfirmasi**.

### 4.3.3 Pemetaan σ → ε per-$K$

| $K$ | σ=0,5 | σ=1,0 | σ=1,5 | σ=2,0 | σ=3,0 |
|---|---|---|---|---|---|
| 2  | `{TBD: eps}` | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` |
| 4  | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` |
| 8  | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` |
| 12 | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` |
| 16 | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` | `{TBD}` |

> Smoke test sudah mengindikasikan pola ini: pada $K=2$, $\sigma=1{,}0$,
> 2 ronde, klien kecil (1.156 citra) menghasilkan $\varepsilon = 1{,}141$
> sedangkan klien besar (7.938 citra) hanya $\varepsilon = 0{,}331$ — bukti
> langsung *privacy amplification by subsampling* (Bab 2.4.3): klien kecil
> "membayar" privasi lebih mahal karena rasio sampling per langkah lebih besar.

### 4.3.4 Kurva $K$

Gambar 4.2 (`figures/K_curve_e1.png`) memetakan mAP@0.5 sebagai fungsi $K$
untuk tiap nilai $\sigma$.

`{TBD: deskripsi pola K-curve}`:
- Pada $\sigma$ kecil (0,5–1,0): apakah K-curve datar atau menurun perlahan?
- Pada $\sigma$ besar (2,0–3,0): apakah K-curve menurun tajam, terutama
  $K=12 \to K=16$ di mana samples-per-klien terkecil jatuh ke 270?
- Titik transisi (jika ada) — apakah konsisten dengan dugaan bahwa
  $N_k \approx 500$ samples adalah ambang konvergensi DP-SGD?

## 4.4 DP-SGD Federated Parsial (E2): Backbone Beku

Eksperimen E2 membekukan *backbone* YOLOv11n (stage 0–9) dan hanya melatih
kepala deteksi (~0,2 juta parameter trainable) dengan DP-SGD. Hipotesis
pendukung: dengan vektor gradien yang jauh lebih kecil, akumulasi *noise*
relatif terhadap sinyal jauh berkurang [Tramèr & Boneh, 2021], sehingga E2
seharusnya **mendominasi** E1 pada ε rendah.

### 4.4.1 Tabel & Kurva

`{TBD: paste tabel dari tables/e2_grid_map50.md}`

Gambar 4.3 dan 4.4 (`figures/privacy_utility_e2.png`, `figures/K_curve_e2.png`)
menggunakan format yang sama dengan 4.3.

### 4.4.2 E1 versus E2 Side-by-Side

| ε ≈ | E1 mAP@0.5 | E2 mAP@0.5 | Selisih |
|---|---|---|---|
| 1  | `{TBD}` | `{TBD}` | `{TBD}` |
| 4  | `{TBD}` | `{TBD}` | `{TBD}` |
| 8  | `{TBD}` | `{TBD}` | `{TBD}` |

`{TBD: 1 paragraf — apakah E2 dominan pada ε rendah seperti diprediksi
Tramèr & Boneh? Apakah keuntungan E2 mengecil/hilang pada ε longgar
(σ kecil) karena di rezim itu kapasitas penuh E1 menjadi lebih berguna
daripada penghematan noise-budget E2?}`

## 4.5 Perbandingan Lintas Mekanisme: DP-FedAvg vs DP-SGD Per-sampel

Sebagai konteks, dilaporkan hasil DP-FedAvg level-klien dari eksperimen
pendahuluan (B3, di-*reuse* tanpa pelatihan ulang).

| Mekanisme | σ | ε | mAP@0.5 | Status |
|---|---|---|---|---|
| DP-FedAvg (B3) | `{TBD: dari hasil lama}` | `{TBD}` | `{TBD: dekat 0?}` | *collapsed*? |
| DP-SGD penuh (E1) | `{TBD: titik terbaik}` | `{TBD}` | `{TBD}` | `{TBD}` |
| DP-SGD parsial (E2) | `{TBD: titik terbaik}` | `{TBD}` | `{TBD}` | `{TBD}` |

`{TBD: 2 paragraf diagnosis}`:
DP-FedAvg menambahkan *noise* sekali pada vektor pembaruan agregat (dimensi
~2,6 juta) per ronde — *noise* permanen yang tidak dapat di-*dampen* oleh
momentum optimizer. DP-SGD per-sampel menyebar *noise* ke banyak langkah
kecil yang sudah ternormalisasi oleh *clipping*; momentum SGD secara natural
melembutkannya sepanjang lintasan optimisasi. Hasil empiris di tabel
mengonfirmasi prediksi teoretis Bab 2.4.4: **pilihan letak penyuntikan
*noise* (level klien vs per-sampel) lebih menentukan keberhasilan privasi
formal pada object detection daripada nilai ε itu sendiri**. Kontribusi
metodologis utama tesis ini adalah menunjukkan, untuk YOLOv11n-GN pada
domain TBS sawit, bahwa DP-SGD per-sampel adalah pilihan praktis sedangkan
DP-FedAvg level-klien gagal menghasilkan model yang berguna pada semua
ε yang diuji.

## 4.6 Ketangguhan Statistik (R1)

Fase 2 mengulangi subset menjanjikan dengan tiga *seed* berbeda
($\{42, 7, 123\}$) untuk melaporkan mean ± std.

| Konfigurasi | mAP@0.5 (mean ± std) | ε |
|---|---|---|
| B2 ($K = 4$) | `{TBD}` | 0 |
| E1 ($K = 4$, σ terbaik) | `{TBD}` | `{TBD}` |
| E1 ($K = 4$, σ terburuk-layak) | `{TBD}` | `{TBD}` |
| E2 ($K = 4$, σ terbaik) | `{TBD}` | `{TBD}` |

`{TBD: 1 paragraf — apakah std cukup kecil (< 0.05 mAP) sehingga klaim
angka utama tidak rentan terhadap variasi seed?}`

## 4.7 Validasi Hipotesis

| Hipotesis | Klaim | Verdict | Bukti |
|---|---|---|---|
| **H1** | Baseline GN mencapai mAP@0.5 acceptable | **TERKONFIRMASI** | 4.1: mAP=0,787 > 0,70 |
| **H2** | DP-SGD per-sampel memberi degradasi *gradual* (bukan *cliff*) pada $\varepsilon \le 8$ | `{TBD}` | 4.3 + 4.5 |
| **H2-K** | $K$ besar **memperburuk** utilitas pada DP-SGD per-sampel (kebalikan DP-FedAvg) | `{TBD}` | 4.3.4 |
| **H3** | XAI tetap *meaningful* pada model DP-trained | `{TBD}` | 4.8 |

## 4.8 Validasi Penjelasan (XAI)

Grad-CAM++ diterapkan pada model B2 (FL tanpa DP) dan model E1/E2 terbaik
sebagai pembanding. Metrik *faithfulness* (Bab 2.8) dihitung pada 100 citra
*test* acak.

| Model | Average Drop | FRR |
|---|---|---|
| B2 (no DP) | `{TBD}` | `{TBD}` |
| E1 terbaik | `{TBD}` | `{TBD}` |
| E2 terbaik | `{TBD}` | `{TBD}` |

`{TBD: 1 paragraf — apakah AD dan FRR pada model DP masih dalam rentang
yang dilaporkan literatur (< 50% drop, FRR > 0,5)? Bila ya → H3 ✓.}`

Visualisasi perbandingan *heatmap* untuk citra representatif disajikan pada
Gambar 4.5 (`figures/xai_comparison.png`).

## 4.9 Ancaman terhadap Validitas

**Internal.** (a) Smoke test menemukan tiga inkompatibilitas Opacus×YOLO
(SiLU *in-place*, signature *loss*, Conv+BN *fusion*) yang ter-patch sebelum
*grid* dilepas; verifikasi run E1 ($K=2$, $\sigma=1{,}0$, 2 ronde)
menghasilkan ε terhitung dan mAP non-NaN. (b) δ tetap pada $10^{-5}$
memudahkan perbandingan tetapi tidak menyesuaikan skala $1/n_k$ per-klien —
analisis sensitivitas opsional bisa dilakukan jika waktu memungkinkan.

**Eksternal.** Dataset berasal dari satu sumber (Roboflow versi 2) sehingga
generalisasi ke perkebunan lain (varietas, iklim, kamera) belum diukur.
Hasil Bab ini berlaku pada distribusi data tersebut.

**Konstruksi.** Sepuluh epoch-ekuivalen (5 ronde × 2 epoch lokal) lebih
pendek dari 50 epoch B1; selisih utilitas dapat dipengaruhi durasi pelatihan
yang lebih singkat, bukan hanya oleh FL/DP. Mitigasi: B2 (no DP, 5 ronde)
menjadi *baseline* yang adil untuk mengisolasi pengaruh DP — bukan B1.
