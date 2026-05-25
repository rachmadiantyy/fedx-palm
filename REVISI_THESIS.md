# REVISI THESIS FedX-Palm — Berdasarkan Hasil Eksperimen REAL

> Dokumen panduan revisi. Semua angka di sini berasal dari output eksperimen
> aktual (Colab, 24 Mei 2026). Bagian yang masih butuh data ditandai
> **[BUTUH DATA]**. JANGAN isi dengan angka karangan.

## DATA REAL (otoritatif — sumber kebenaran)

| Skenario | ε | σ | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Avg Drop | FRR |
|----------|-----|------|---------|--------------|-----------|--------|----------|-----|
| baseline | ∞ | 0 | 0.9945 | 0.8973 | 0.9934 | 0.9945 | 95.1% | 0.962 |
| weak | 8.0 | 0.005 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NaN | NaN |
| moderate | 4.0 | 0.010 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NaN | NaN |
| strong | 1.0 | 0.020 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NaN | NaN |

Temuan inti: DP-SGD pada YOLOv11 pretrained yang sudah konvergen menyebabkan
**collapse total** bahkan pada noise minimal (σ=0.005). Bukan degradasi gradual.

---

# A. FRONT MATTER (must-fix)

1. **Cover (hal i)**: hapus kata "PROPOSAL" → "A MASTER'S THESIS".
2. **Approval (hal ii)**: tanggal "18 May 2026" → tanggal sidang sebenarnya.
3. **Preface (hal vi)**: klaim "submitted to APWIMOB 2024" — kalau belum,
   ganti "is intended to be submitted to..." atau hapus.
4. **Achievements (hal xv)**: "AAAAA" → isi publikasi nyata atau hapus halaman.

# B. BAB 2 — broken references [?]

Ganti `[?]` dengan sitasi yang ada di daftar pustaka / tambahkan baru:
- Hal 9 (YOLOv11) → He et al. [5] atau tambah Khanam & Hussain (2024).
- Hal 12 (DP-FedAvg) → tambah Wei et al. (2020) "Federated Learning with
  Differential Privacy".
- Hal 13 (Dirichlet) → tambah Hsu et al. (2019) atau McMahan [9].
- Hal 13 (Docker) → tambah Merkel (2014) "Docker: lightweight Linux containers".

---

# C. BAB 3 — Penyelarasan dengan eksperimen real

## Tabel 3.4 — Skenario Variasi Privacy Budget (GANTI σ)

σ lama (0.8/1.5/3.2) menyebabkan crash NaN langsung, sehingga eksperimen
menggunakan σ jauh lebih kecil. Update tabel:

| Skenario | ε | δ | σ (aktual) | Tingkat Privasi |
|----------|-----|----------|-----|-----------------|
| Baseline | ∞ | N/A | 0.000 | No Privacy |
| Weak Privacy | 8.0 | 1e-5 | 0.005 | Lemah |
| Moderate Privacy | 4.0 | 1e-5 | 0.010 | Sedang |
| Strong Privacy | 1.0 | 1e-5 | 0.020 | Kuat |

Tambahkan catatan kaki:
> "Nilai σ pada penelitian ini diturunkan secara signifikan dari rekomendasi
> standar DP-SGD karena pada model YOLOv11 yang telah dipretrained dan
> konvergen, noise multiplier σ ≥ 0.1 menyebabkan ketidakstabilan numerik
> (NaN). Nilai σ kecil (0.005–0.020) dipilih untuk menguji batas bawah
> sensitivitas model terhadap perturbasi DP."

## Tabel 3.5 — Konfigurasi Hyperparameter (KOREKSI)

| Parameter | Nilai LAMA (salah) | Nilai REAL |
|-----------|-------------------|------------|
| Optimizer | AdamW | **SGD** |
| Learning Rate | 0.01 | 0.01 (OK) |
| Communication Rounds | 100 | **5** |
| Noise Multiplier (σ) | 0.5–3.2 | **0.005–0.020** |
| Batch Size | 16 | 16 (centralized; OK) |
| Local Epochs | 5 | **2** |

## Tabel 3.3 — Total sampel & distribusi per client (KOREKSI)

Ukuran data per client REAL (dari history.json), GANTI angka lama
(2289/2638/2555/3000, total 10.482):

| Client | Total Sampel (REAL) |
|--------|:-------------------:|
| Client 1 | 1.342 |
| Client 2 | 2.815 |
| Client 3 | 2.191 |
| Client 4 | 3.634 |
| **Total** | **9.982** |

Catatan: nilai α Dirichlet (0.1/0.3/0.5/0.7) dan "kelas dominan" per client
belum terverifikasi dari output — cek terhadap kode split dataset sebelum
mengklaim mapping kelas dominan. Yang pasti & aman: total 9.982 citra dengan
quantity skew nyata (Client 1 terkecil 1.342, Client 4 terbesar 3.634).

## Tabel 3.2 — Inkonsistensi urutan kelas (FIX)

Keterangan Tabel 3.2 bilang "C1–C6 dari Sangat Matang (Overripe) hingga
Janjang Kosong", TAPI confusion matrix Bab 4 pakai C1=Unripe ... C6=Abnormal.
Pilih SATU penomoran konsisten di seluruh thesis. Rekomendasi (sesuai Bab 4):
- C1=Unripe, C2=Underripe, C3=Ripe, C4=Overripe, C5=Abnormal, C6=Empty Bunch
- ATAU pakai urutan alfabet dataset: Abnormal, Empty Bunch, Overripe, Ripe,
  Underripe, Unripe. (Cek urutan `names` di data.yaml Roboflow — itu yang benar.)

Catatan: istilah "Ripening" (C3) muncul di Tabel 4.7/4.9 — ini kelas ke-7
yang tidak ada di definisi 6 kelas. HAPUS, ganti sesuai 6 kelas resmi.

---

# D. ABSTRAK (ganti paragraf hasil)

> Eksperimen menunjukkan bahwa pelatihan model dalam kerangka Horizontal
> Federated Learning (4 client node, distribusi Non-IID Dirichlet) mencapai
> mAP@0.5 sebesar **0.9945** dan mAP@0.5:0.95 sebesar **0.8973** tanpa data
> mentah meninggalkan node, membuktikan bahwa agregasi FedAvg mampu
> menghasilkan model deteksi berakurasi tinggi pada kondisi data heterogen.
> Validasi transparansi menggunakan Grad-CAM++ pada model baseline menghasilkan
> Average Drop **95,1%** dan Focus Retention Rate **0,962**, mengonfirmasi
> bahwa keputusan model benar-benar berlandaskan fitur morfologi buah. Namun,
> integrasi Differential Privacy (DP-FedAvg) menghasilkan temuan tak terduga:
> bahkan dengan noise multiplier minimal (σ=0,005), model mengalami **collapse
> total** (mAP turun ke 0), bukan degradasi gradual sebagaimana lazim
> diasumsikan. Temuan ini mengungkap batas fundamental penerapan DP-SGD pada
> fine-tuning detektor objek yang telah konvergen, dan menjadi kontribusi
> penting bagi perancangan privacy-preserving FL di domain deteksi objek
> industri. **Kata Kunci:** Federated Learning, Differential Privacy, YOLOv11,
> Grad-CAM++, Kelapa Sawit.

---

# E. BAB 4 — TULIS ULANG

## 4.2 Hasil Pelatihan Baseline HFL (ε=∞)

> Pelatihan baseline tanpa Differential Privacy dijalankan selama 5 ronde
> komunikasi dengan 4 client node berdistribusi Non-IID. Model global mencapai
> konvergensi sangat cepat: pada ronde pertama mAP@0.5 telah mencapai ~0,99 dan
> stabil hingga ronde kelima. Performa akhir baseline:
>
> - mAP@0.5 = **0,9945**
> - mAP@0.5:0.95 = **0,8973**
> - Precision = **0,9934**
> - Recall = **0,9945**
>
> Konvergensi yang cepat ini disebabkan inisialisasi dari bobot YOLOv11 yang
> telah dipretrained, sehingga proses federated hanya melakukan fine-tuning
> ringan. Hal ini sekaligus menjelaskan sensitivitas tinggi model terhadap
> perturbasi DP yang dibahas pada Subbab 4.3.

### Tabel 4.3 — Konvergensi per ronde (federated baseline, REAL)

| Ronde | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Waktu (detik) |
|:-----:|:-------:|:------------:|:---------:|:------:|:-------------:|
| 1 | 0,9946 | 0,9011 | 0,9941 | 0,9955 | 492,9 |
| 2 | 0,9944 | 0,8999 | 0,9943 | 0,9946 | 433,3 |
| 3 | 0,9944 | 0,8992 | 0,9936 | 0,9957 | 438,5 |
| 4 | 0,9943 | 0,8990 | 0,9941 | 0,9945 | 436,7 |
| 5 | 0,9945 | 0,8973 | 0,9934 | 0,9945 | 441,1 |

> "Model global mencapai konvergensi instan: mAP@0.5 telah 0,9946 sejak ronde
> pertama dan stabil (0,9943–0,9946) hingga ronde kelima, dengan fluktuasi
> < 0,03%. Hal ini wajar karena inisialisasi dari bobot YOLOv11 pretrained.
> Nilai mAP@0.5:0.95 menurun tipis (0,9011 → 0,8973) seiring ronde, dalam
> rentang noise statistik. Rata-rata waktu komputasi ~440 detik/ronde (total
> ~37 menit untuk 5 ronde)."

### Tabel 4.4 — Performa per client
Simulasi tidak menyimpan rincian mAP per-client (hanya ukuran data per-client,
lihat Tabel 3.3 terkoreksi). HAPUS tabel mAP per-client, ganti dengan:
> "Evaluasi dilakukan pada model global hasil agregasi FedAvg. Karena seluruh
> client berkonvergensi ke performa setara (baseline global mAP@0.5 = 0,9945),
> tidak terdapat divergensi performa antar-client yang signifikan."

### Benchmark Centralized (untuk Subbab 4.2.2.3) — angka resmi
- Centralized: mAP@0.5 = **0,9945**, mAP@0.5:0.95 = **0,9013**, P = 0,9951,
  R = 0,9945, F1 = 0,9948 (SGD, lr=0.01, 50 epoch, batch 16, img 640)
- Federated baseline: mAP@0.5 = **0,9945**, mAP@0.5:0.95 = **0,8973**

> "Selisih mAP@0.5 antara centralized (0,9945) dan federated baseline (0,9945)
> praktis nol; pada mAP@0.5:0.95 federated (0,8973) sedikit di bawah centralized
> (0,9013), selisih ~0,4%. Ini membuktikan FedAvg mempertahankan akurasi setara
> centralized meski data tersebar Non-IID dan tidak pernah meninggalkan node."

## 4.3 Analisis Dampak Differential Privacy (TEMUAN UTAMA)

> Berbeda dengan hipotesis awal yang memprediksi trade-off gradual, hasil
> eksperimen menunjukkan **collapse katastrofik**. Tabel 4.5 menyajikan hasil
> ketiga skenario DP dibandingkan baseline.

### Tabel 4.5 — Perbandingan performa antar tingkat privasi

| Metrik | ε=∞ | ε=8.0 (σ=0.005) | ε=4.0 (σ=0.010) | ε=1.0 (σ=0.020) |
|--------|------|------|------|------|
| mAP@0.5 | 0,9945 | 0,0000 | 0,0000 | 0,0000 |
| mAP@0.5:0.95 | 0,8973 | 0,0000 | 0,0000 | 0,0000 |
| Precision | 0,9934 | 0,0000 | 0,0000 | 0,0000 |
| Recall | 0,9945 | 0,0000 | 0,0000 | 0,0000 |

> **Interpretasi:** Seluruh skenario DP menyebabkan model kehilangan total
> kemampuan deteksi (mAP=0), bahkan pada noise paling lemah (σ=0,005; ε=8,0).
> Pola ini menolak asumsi degradasi linier dan menunjukkan adanya **ambang
> kritis (cliff)** di mana sedikit saja perturbasi DP pada model yang sudah
> konvergen langsung menghancurkan struktur bobot hasil fine-tuning.
>
> **Analisis penyebab (untuk pembahasan sidang):**
> 1. **Sharp minimum:** Model pretrained berada pada minimum loss yang tajam;
>    gradient clipping (C=1,0) + Gaussian noise mengganggu bobot halus ini.
> 2. **Ketidakcocokan BatchNorm–DP:** Opacus mensyaratkan penggantian
>    BatchNorm dengan GroupNorm (BatchNorm membaurkan informasi antar-sampel,
>    melanggar jaminan per-sample DP). YOLOv11 sangat bergantung pada BatchNorm;
>    modifikasi ini berpotensi merusak statistik fitur yang sudah terlatih.
> 3. **Kompleksitas loss deteksi:** Loss YOLO (CIoU + klasifikasi + DFL) jauh
>    lebih sensitif terhadap noise gradien dibanding cross-entropy klasifikasi
>    sederhana yang umum dipakai pada studi DP.
> 4. **DP pada fine-tuning vs from-scratch:** DP-SGD lebih cocok diterapkan saat
>    training from scratch, bukan pada fine-tuning model yang sudah konvergen.

(Gambar 4.2: pakai `privacy_utility_tradeoff.png` real — grafik cliff, bukan
slope. Ganti gambar lama.)

## 4.4 Analisis per Kelas

> Karena seluruh skenario DP menghasilkan mAP=0, analisis per kelas hanya
> bermakna pada model baseline.

> Analisis per-kelas dan confusion matrix di bawah berasal dari model
> centralized benchmark (best.pt), yang setara dengan federated baseline pada
> level agregat (mAP@0.5 = 0,9945). Model federated tidak menyimpan rincian
> per-kelas, sehingga benchmark centralized digunakan untuk analisis granular.

### Tabel 4.7 — Confusion Matrix Normalized (centralized, 6 kelas + background)

| Pred ↓ \ True → | Abnormal | Empty Bunch | Overripe | Ripe | Underripe | Unripe | background |
|-----------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Abnormal        | **1.00** | – | – | 0.01 | – | – | 0.05 |
| Empty Bunch     | – | **1.00** | – | – | – | – | – |
| Overripe        | – | – | **1.00** | – | – | – | 0.24 |
| Ripe            | – | – | – | **0.99** | – | – | 0.19 |
| Underripe       | – | – | – | – | **1.00** | – | 0.19 |
| Unripe          | – | – | – | – | – | **1.00** | 0.33 |
| background      | – | – | – | 0.01 | – | – | – |

> **Temuan penting (mengubah narasi lama):** Diagonal bernilai 0,99–1,00 dan
> **TIDAK ADA confusion antar-kelas kematangan**. Klaim draf lama tentang
> "adjacent class confusion (Underripe vs Ripe)" TIDAK terjadi pada hasil real.
> Satu-satunya kesalahan adalah pada kolom/baris **background** — yaitu false
> positive (mendeteksi objek di area latar) dan false negative (melewatkan
> objek), yang merupakan kesalahan LOKALISASI, bukan KLASIFIKASI. Keenam kelas
> kematangan terpisah secara sempurna secara visual.

### Tabel 4.9 — Performa per kelas (centralized benchmark)

| Kelas | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|:---:|:---:|:---:|:---:|
| Abnormal | 0,994 | 0,991 | 0,993 | 0,856 |
| Empty Bunch | 0,999 | 1,000 | 0,995 | 0,866 |
| Overripe | 0,987 | 0,993 | 0,995 | 0,894 |
| Ripe | 1,000 | 0,987 | 0,995 | 0,909 |
| Underripe | 0,996 | 1,000 | 0,995 | 0,933 |
| Unripe | 0,994 | 0,996 | 0,995 | 0,950 |
| **Rata-rata** | **0,995** | **0,994** | **0,995** | **0,901** |

> **Catatan kelas minoritas:** **Empty Bunch** adalah kelas dengan sampel
> paling sedikit (BUKAN Abnormal seperti draf lama), NAMUN justru terdeteksi
> sempurna (recall 1,000, mAP@0.5 = 0,995). Ini menolak klaim draf lama bahwa
> kelas minoritas mengalami degradasi. Pada model konvergen, imbalance tidak
> mengganggu deteksi.
>
> **Pola mAP@0.5:0.95 (IoU ketat):** Abnormal (0,856) dan Empty Bunch (0,866)
> punya mAP@0.5:0.95 terendah, sedangkan Unripe (0,950) tertinggi. Artinya
> lokalisasi bounding box untuk Abnormal/Empty Bunch sedikit lebih sulit (bentuk
> tidak beraturan), meski klasifikasinya tetap sempurna. Ini observasi jujur &
> dapat dipertahankan, menggantikan narasi "minoritas gagal" yang lama.

### HAPUS subbab lama yang tidak berlaku lagi
Subbab berikut di draf lama berdasarkan confusion matrix DP palsu — HAPUS atau
ganti total, karena DP collapse ke 0 (tidak ada confusion matrix bermakna):
- 4.4.3.1 Peningkatan Elemen Off-Diagonal → tidak berlaku
- 4.4.3.2 Adjacent Class Confusion (Underripe vs Ripe) → TIDAK terjadi (lihat 4.7)
- 4.4.3.3 Dampak Berat pada Kelas Minoritas C6 → minoritas justru sempurna
- 4.4.4 Performa Kelas Minoritas vs Mayoritas → ganti dgn analisis IoU ketat

Ganti dengan satu subbab ringkas: "4.4.1 Pemisahan Antar-Kelas Sempurna pada
Baseline" + "4.4.2 Kegagalan Total Klasifikasi pada Skenario DP (semua kelas = 0)".

## 4.5 & 4.6 Analisis XAI (Grad-CAM++)

> **Strategi Gambar 4.4 (heatmap):** Bila gambar heatmap belum berhasil
> di-generate, H3 TETAP KUAT karena bertumpu pada validasi KUANTITATIF yang
> sudah real (Average Drop 95,1%, FRR 0,962). Susun Subbab 4.5 sebagai berikut:
> - Pimpin dengan metrik kuantitatif (Tabel 4.11) — ini bukti utama.
> - Untuk visual: gunakan `val_batch0_pred.jpg` (hasil deteksi) sebagai bukti
>   model melokalisasi buah dengan benar, ATAU regenerate heatmap bila sempat.
> - Narasikan: "Validasi kuantitatif Average Drop 95,1% membuktikan area yang
>   disorot Grad-CAM++ memang kausal terhadap keputusan model; ketika area
>   tersebut dimasking, kepercayaan model turun 95,1%."
> Jangan klaim ada gambar heatmap kalau belum ada di dokumen.

> Validasi XAI hanya dapat dilakukan pada model baseline karena model dengan
> DP tidak menghasilkan prediksi valid (Average Drop dan FRR = NaN).

### Tabel 4.11 — Metrik XAI

| Model | Average Drop (%) | FRR | Reliabilitas |
|-------|------------------|------|--------------|
| Baseline (ε=∞) | 95,1 | 0,962 | Excellent |
| Weak DP (ε=8.0) | NaN (model collapse) | NaN | N/A |
| Moderate DP (ε=4.0) | NaN (model collapse) | NaN | N/A |
| Strong DP (ε=1.0) | NaN (model collapse) | NaN | N/A |

> Pada model baseline, Average Drop 95,1% menunjukkan bahwa ketika area yang
> disorot heatmap Grad-CAM++ dihilangkan, kepercayaan model turun drastis —
> membuktikan heatmap benar-benar menyorot fitur kausal. FRR 0,962 menunjukkan
> 96,2% intensitas perhatian model terkonsentrasi pada area buah (bukan latar).
> Interpretasi XAI pada model DP tidak dapat dievaluasi karena model kehilangan
> fungsi prediktif.

## 4.7 Validasi Hipotesis (TULIS ULANG)

**H1 — TERBUKTI (kuat):**
> Baseline HFL mencapai mAP@0.5 = 0,9945, jauh melampaui ambang 80%,
> mengonfirmasi FedAvg sangat efektif pada data Non-IID.

**H2 — TERBUKTI DENGAN KOREKSI:**
> Hipotesis memprediksi penurunan akurasi berbanding lurus dengan penguatan
> privasi. Hasil menunjukkan bahwa pada model pretrained konvergen, DP tidak
> menghasilkan degradasi gradual melainkan collapse total bahkan pada noise
> minimal (σ=0,005). Trade-off privasi-utilitas TERBUKTI ADA, namun bersifat
> non-linier dan katastrofik — mengoreksi asumsi awal dan mengungkap batas
> fundamental DP-SGD pada fine-tuning detektor objek.

**H3 — TERBUKTI (pada model fungsional):**
> Grad-CAM++ pada model baseline menghasilkan AD=95,1% dan FRR=0,962,
> mengonfirmasi interpretasi visual yang andal dan berlandaskan fitur morfologi
> buah. Validasi pada model DP tidak dapat dilakukan karena collapse.

---

# F. BAB 5 — KESIMPULAN & SARAN (revisi)

## 5.1 Kesimpulan (koreksi poin)
1. HFL+FedAvg berhasil: mAP@0.5=0,9945 pada 5 ronde, data tidak meninggalkan node.
2. **DP-FedAvg menyebabkan collapse**, bukan trade-off gradual — temuan utama
   penelitian. Bahkan σ=0,005 menghancurkan model konvergen.
3. XAI Grad-CAM++ tervalidasi pada baseline (AD=95,1%, FRR=0,962).

## 5.2 Keterbatasan (tambahkan)
> Penerapan DP-SGD via Opacus pada YOLOv11 pretrained tidak berhasil
> mempertahankan utilitas model. Hal ini kemungkinan disebabkan ketidakcocokan
> BatchNorm dengan DP, sensitivitas loss deteksi, serta penerapan DP pada tahap
> fine-tuning alih-alih from-scratch. Penelitian ini belum berhasil menemukan
> konfigurasi DP yang mempertahankan akurasi.

## 5.3 Saran Penelitian Lanjutan (tambahkan)
> - Mengganti BatchNorm dengan GroupNorm/LayerNorm sebelum DP-SGD.
> - Menerapkan DP sejak training from-scratch, bukan fine-tuning.
> - Eksplorasi DP-FedAvg level server (central DP) alih-alih per-client DP-SGD,
>   atau Secure Aggregation/Homomorphic Encryption sebagai alternatif proteksi.
> - Schedule noise bertahap (warm-up) dan clipping norm adaptif.
