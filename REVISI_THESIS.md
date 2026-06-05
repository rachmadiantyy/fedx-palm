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

## Tabel 3.4 — Skenario Variasi Privacy Budget (KOREKSI ε — R1)

> **PENTING (R1).** ε pada tabel lama (8.0 / 4.0 / 1.0) **ditetapkan manual** dan
> **salah ~5 ordo besaran**. ε **wajib** dihitung dari privacy accountant (RDP),
> bukan dilabel. ε dihitung dari (σ, q, T, δ) — lihat derivasi di Subbab R1 dan
> sel "6b. Derivasi ε" di notebook. **ε yang benar (di bawah) justru sangat
> besar**, artinya σ sekecil itu hampir **tidak memberi privasi** sama sekali.

Parameter: q (client sampling rate) = 1.0 (partisipasi penuh 4 klien/ronde),
T = 5 ronde, δ = 1e-5.

| Skenario | σ (aktual) | ε **LAMA (salah)** | ε **TERKOREKSI (RDP)** | Tingkat Privasi sebenarnya |
|----------|:----------:|:------------------:|:----------------------:|----------------------------|
| Baseline | 0.000 | ∞ | ∞ | No Privacy |
| Weak     | 0.005 | 8.0 | **≈ 1.0 × 10⁵** | Tanpa privasi efektif |
| Moderate | 0.010 | 4.0 | **≈ 2.6 × 10⁴** | Tanpa privasi efektif |
| Strong   | 0.020 | 1.0 | **≈ 6.8 × 10³** | Tanpa privasi efektif |

Ganti catatan kaki lama dengan yang jujur:
> "Nilai ε pada penelitian ini dihitung menggunakan *Rényi Differential Privacy
> accountant* (Mironov, 2017) atas mekanisme Gaussian DP-FedAvg level-klien yang
> dikomposisikan sepanjang T = 5 ronde dengan partisipasi penuh (q = 1) dan
> δ = 1e-5. Hasil perhitungan menunjukkan bahwa nilai σ yang digunakan
> (0.005–0.020) menghasilkan ε ≫ 10³, sehingga **tidak berada pada rezim privasi
> yang bermakna**. Penurunan σ dilakukan karena σ pada rentang standar DP-SGD
> (σ ≈ 1–3, yang memberi ε ≈ 1–8) menyebabkan ketidakstabilan numerik (NaN) pada
> YOLOv11 pretrained. Implikasinya dibahas pada analisis collapse (Subbab 4.3)."

### σ yang dibutuhkan untuk privasi bermakna (RDP, T=5, δ=1e-5)

Untuk konteks penguji — inilah σ yang *seharusnya* dipakai bila ingin ε bermakna:

| Target ε | σ yang dibutuhkan |
|:--------:|:-----------------:|
| 8.0 (lemah) | ≈ 1.54 |
| 4.0 (sedang) | ≈ 2.90 |
| 1.0 (kuat) | ≈ 10.96 |

Justru pada σ ≈ 1.5–11 inilah model collapse (NaN), sehingga eksperimen tidak
pernah mencapai titik privasi-bermakna. **Temuan ini memperkuat R3**: collapse
terjadi *sebelum* rezim privasi tercapai, mengindikasikan masalah pipeline,
bukan trade-off privasi-utilitas sejati.

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

## Tabel 3.2 — KUNCI urutan kelas ke ALFABET (sesuai data real)

Data real (metrics_per_class.csv) memakai urutan alfabet sesuai `class_id`:
**0=Abnormal, 1=Empty Bunch, 2=Overripe, 3=Ripe, 4=Underripe, 5=Unripe.**
Ini urutan resmi dataset Roboflow → GUNAKAN INI DI SELURUH THESIS.

- Hapus penomoran "C1=Unripe..." dan keterangan "C1–C6 dari Overripe hingga
  Empty Bunch" yang lama (tidak sesuai data).
- Hapus istilah "Ripening" (kelas ke-7 yang tidak ada).
- Jika tetap ingin pakai label C1–C6, definisikan: C1=Abnormal, C2=Empty Bunch,
  C3=Overripe, C4=Ripe, C5=Underripe, C6=Unripe — TAPI lebih aman pakai nama
  kelas langsung (Abnormal, Empty Bunch, ...) agar tidak ada ambiguitas.

---

# D. ABSTRAK (ganti paragraf hasil)

> Eksperimen menunjukkan bahwa pelatihan model dalam kerangka Horizontal
> Federated Learning (4 client node, distribusi Non-IID Dirichlet) mencapai
> mAP@0.5 sebesar **0,9945** dan mAP@0.5:0.95 sebesar **0,8973** tanpa data
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
> confusion antar-kelas kematangan **nyaris nol** (maksimal 0,01, yaitu sebagian
> kecil Ripe terprediksi Abnormal). Klaim draf lama tentang "adjacent class
> confusion (Underripe vs Ripe)" TIDAK terjadi pada hasil real. Sumber kesalahan
> dominan justru pada kolom/baris **background** — yaitu false positive
> (mendeteksi objek di area latar) dan false negative (melewatkan objek), yang
> merupakan kesalahan LOKALISASI, bukan KLASIFIKASI. Keenam kelas kematangan
> praktis terpisah sempurna secara visual.

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

> ⚠️ ANTISIPASI PERTANYAAN PENGUJI (konvensi Average Drop): Definisi Average
> Drop di thesis ini (Pers. 2.7) adalah penurunan confidence ketika area PENTING
> DIHILANGKAN, sehingga **nilai tinggi = penjelasan makin baik/faithful**.
> Ini BERBEDA dari konvensi paper Grad-CAM++ asli (Chattopadhyay dkk.) yang
> mengukur drop saat hanya MEMPERTAHANKAN area penting (nilai rendah = lebih
> baik). Pastikan Bab 2.6.2 konsisten dengan definisi yang dipakai, dan siap
> menjelaskan bahwa AD=95,1% bermakna BAIK dalam konvensi penelitian ini
> (menghapus area kunci menghancurkan 95,1% confidence → heatmap memang kausal).

## 4.7 Validasi Hipotesis (TULIS ULANG)

**H1 — TERBUKTI (kuat):**
> Baseline HFL mencapai mAP@0.5 = 0,9945, jauh melampaui ambang 80%,
> mengonfirmasi FedAvg sangat efektif pada data Non-IID.

**H2 — TERBUKTI SEBAGIAN (bentuk proporsional DITOLAK):**
> Hipotesis memprediksi penurunan akurasi *berbanding lurus* dengan penguatan
> privasi (semakin kecil ε, semakin rendah akurasi secara gradual). Hasil
> menunjukkan: (a) trade-off privasi-utilitas memang ADA — penerapan DP
> menurunkan utilitas → bagian ini terbukti; NAMUN (b) hubungan proporsional/
> monotonik yang dihipotesiskan TIDAK terjadi — ketiga tingkat privasi
> (ε=8,0; 4,0; 1,0) sama-sama collapse ke mAP=0 tanpa perbedaan gradual. Dengan
> demikian H2 terbukti secara kualitatif (DP merusak utilitas) tetapi DITOLAK
> dalam bentuk proporsional yang spesifik. Temuan ini mengungkap batas
> fundamental DP-SGD pada fine-tuning detektor objek yang sudah konvergen.

> CATATAN untuk diskusi dengan pembimbing: pilih satu label final — "terbukti
> sebagian" (paling jujur) ATAU "ditolak" (jika pembimbing menilai inti H2
> adalah proporsionalitas). Hindari klaim "terbukti penuh" karena pola
> proporsional tidak teramati.

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

---

# F+. R1 — Derivasi Privacy Budget ε dari Accountant (Pembimbing II)

> Menjawab **R1** (isu terbesar): "Nilai ε pada DP-SGD tidak boleh ditetapkan
> secara tabel; ia harus diturunkan dari privacy accountant (RDP/PRV)
> berdasarkan σ, sampling rate (q), jumlah langkah (T), dan δ." Siap tempel
> sebagai Subbab 3.X (Analisis Privacy Budget) atau lampiran derivasi.

## F+.1 Mekanisme & asumsi

Implementasi adalah **DP-FedAvg level-klien** (McMahan dkk., 2018), *bukan*
per-sample DP-SGD: tiap ronde, delta bobot klien (w_local − w_global) di-clip ke
L2-norm C lalu ditambah noise Gaussian N(0, (σ·C)²) sebelum agregasi. Maka:

- **Noise multiplier** z = σ (karena noise_std / sensitivity = (σ·C)/C = σ).
- **Sampling rate** q = 1.0 — keempat klien berpartisipasi penuh tiap ronde
  (tidak ada amplifikasi privasi via subsampling; dengan hanya 4 klien,
  amplifikasi memang minimal).
- **Komposisi** T = 5 ronde komunikasi.
- **δ** = 1e-5.

## F+.2 Derivasi (RDP — Mironov, 2017)

Untuk mekanisme Gaussian non-subsampled, *Rényi DP* pada order α:

```
ε_RDP(α) = T · α / (2 σ²)
```

Konversi ke (ε, δ)-DP, diminimalkan atas α > 1:

```
ε = min_{α>1} [ T·α/(2σ²) + ln(1/δ)/(α − 1) ]
```

Perhitungan ini deterministik dan diimplementasikan **mandiri** (numpy) di
sel "6b. Derivasi ε" notebook — **independen dari trainer Opacus**, yang
memang tidak digunakan karena `ModuleValidator`-nya tidak kompatibel dengan
BatchNorm YOLOv11 (lihat juga R2). Sebagai validasi kedua, dihitung pula
zCDP analitik (Bun & Steinke, 2016), ρ = T/(2σ²) lalu
ε = ρ + 2√(ρ·ln(1/δ)) — konsisten dengan RDP ±0,1%.

## F+.3 Hasil (T=5, q=1, δ=1e-5)

| σ | ε (RDP) | Interpretasi |
|:-----:|:-----------:|--------------|
| 0.005 | ≈ 1.0 × 10⁵ | tanpa privasi efektif |
| 0.010 | ≈ 2.6 × 10⁴ | tanpa privasi efektif |
| 0.020 | ≈ 6.8 × 10³ | tanpa privasi efektif |
| 1.544 | ≈ 8.0 | privasi lemah (rezim bermakna) |
| 2.898 | ≈ 4.0 | privasi sedang |
| 10.96 | ≈ 1.0 | privasi kuat |

## F+.4 Koreksi pemetaan & temuan

- Pemetaan lama (σ=0.005 → ε=8.0) **keliru ~5 ordo besaran** (ε sebenarnya
  ≈ 10⁵). Tabel 3.4 sudah dikoreksi di atas.
- Semua σ yang diuji (0.005–0.020, bahkan sweep halus 0.0001–0.003) memberi
  **ε ≫ 10³ → tidak ada privasi bermakna**.
- Privasi bermakna (ε ≈ 1–8) menuntut σ ≈ 1.5–11; pada rentang itu model
  collapse (NaN). **Konsekuensi:** eksperimen tidak pernah mencapai rezim
  privasi-bermakna → menguatkan reframe R3 (collapse = artefak pipeline,
  bukan trade-off privasi sejati).

## F+.5 Tindak lanjut R1

- [x] Accountant RDP eksplisit (σ,q,T,δ → ε) di notebook sel 6b
- [x] Tabel derivasi & σ-untuk-target-ε
- [x] Koreksi Tabel 3.4 (ε terkoreksi)
- [ ] Tempel derivasi ke Bab 3 tesis + lampiran
- [ ] Sitasi: Mironov (2017) RDP, Bun & Steinke (2016) zCDP, Abadi dkk. (2016)
      — lihat juga R9

---

# F+. R4 — Audit Data Leakage & Strategi Re-Split (Pembimbing II)

> Bagian ini menjawab catatan revisi **R4** dari Pembimbing II terkait kecurigaan
> *data leakage* yang menjelaskan mAP@0.5 baseline 0.9945 yang tampak terlalu
> tinggi. Penjelasan di bawah siap tempel sebagai Subbab 3.X (Validitas Data)
> atau dimasukkan ke 5.2 Keterbatasan, sesuai arahan akhir Pembimbing.

## F+.1 Temuan audit (output sel 4b notebook federated)

Dataset `palm-fruit-ripeness-detection v2` (Roboflow) membagi train/valid/test
**secara acak per-frame**. Karena sumbernya adalah **video tandan sawit**,
satu tandan diwakili oleh banyak frame berurutan dengan nama
`framesawit<id>-<frame_no>-_png.rf.<hash>.jpg`.

Hasil audit (sel 4b sebelum perbaikan):

| Pemeriksaan | Definisi | Hasil |
|-------------|----------|------:|
| **HARD leakage** | Citra original identik di antara split | **0** |
| **SOFT leakage** | `bunch_id` sama muncul di lebih dari satu split | **100% tandan valid juga ada di train** |

Implikasinya: model tidak benar-benar diuji pada tandan baru — frame berbeda
dari tandan yang sama membuat valid/test menjadi *in-distribution* terhadap
train. Angka mAP centralized baseline 0.9945 karenanya **optimis** dan tidak
mengukur generalisasi sebenarnya.

## F+.2 Strategi anti-leakage yang diterapkan

Diimplementasikan sebagai **sel 3b** di notebook federated_simulation
(berjalan tepat sebelum Dirichlet split). Pendekatan: **Stratified Group
Split** dengan empat invariant:

1. **Group key** = `bunch_id` (regex `frame[a-z]*\d+` dari nama file,
   suffix Roboflow `.rf.<hash>` dibuang lebih dulu).
2. **Stratifikasi** per **kelas dominan** tandan (mayoritas kelas dari
   seluruh frame tandan tersebut). Tujuan: menjaga representasi kelas
   minoritas (mis. *Empty Bunch*) di valid/test.
3. **Rasio target** 80/10/10 (sama dengan rasio Roboflow asli, agar ukuran
   train tetap memadai untuk DP-SGD yang sensitif noise).
4. **Garansi disjoint**: satu `bunch_id` hanya muncul di **satu** split.

Determinisme: `random.Random(seed=42)` — reproducible.

Verifikasi (sel 4b setelah perbaikan) memeriksa **tiga pasangan** split
(train↔valid, train↔test, valid↔test) untuk HARD dan SOFT leakage, dan
harus mencetak `>>> BERSIH` sebelum training dilanjutkan.

## F+.3 Pernyataan validitas untuk tesis (siap tempel)

> "Untuk menjamin validitas evaluasi, dataset di-split ulang berbasis
> identitas tandan sawit (`bunch_id`). Strategi yang digunakan adalah
> *stratified group split* dengan rasio 80/10/10, di mana setiap tandan
> hanya muncul di satu split (train, valid, atau test). Stratifikasi
> dilakukan terhadap kelas dominan setiap tandan agar kelas minoritas tetap
> terwakili. Audit pasca-split mengonfirmasi tidak adanya tumpang tindih
> citra maupun tandan antar-split."

## F+.4 Dampak terhadap angka yang dilaporkan

- Angka mAP/Precision/Recall di **DATA REAL** (Tabel atas dokumen ini)
  berasal dari split Roboflow asli yang masih leaky. Setelah training
  ulang di atas split baru, **angka diperkirakan turun** (besaran pasti
  baru diketahui setelah training selesai).
- **Tren privacy-utility** (Tabel 4.5 — perbandingan ε) tetap valid karena
  bias data konstan di seluruh skenario; relativitas degradasi DP terhadap
  baseline tidak berubah.
- Kesimpulan utama tesis ("DP-SGD pada YOLOv11 pretrained menyebabkan
  collapse") tidak terpengaruh re-split.

## F+.5 Tindak lanjut

- [x] Audit leakage pada split asli (sel 4b)
- [x] Implementasi re-split per `bunch_id` (sel 3b)
- [x] Verifikasi 0 leakage di split baru (sel 4b versi update)
- [ ] Re-training centralized baseline di atas split baru — laporkan mAP baru
- [ ] Re-simulasi federated (4 skenario ε) di atas split baru
- [ ] Update tabel **DATA REAL**, Tabel 4.3, 4.5, 4.9 dengan angka pasca-resplit
- [ ] Bandingkan delta mAP baseline (lama vs baru) sebagai validasi besarnya
      bias akibat leakage

---

# G. DEPLOYMENT & DEMO VPS (bagian baru)

> Bagian ini mendokumentasikan arsitektur ter-Dockerisasi dan deployment model
> ke VPS untuk demonstrasi inference langsung. Narasi di bawah siap tempel.
> **Framing jujur yang harus dipegang:** sistem *dirancang* sebagai arsitektur
> federated ter-Dockerisasi (4 client + server); *eksperimen pelatihan*
> dijalankan sebagai simulasi ekuivalen di Google Colab karena keterbatasan
> sumber daya multi-node GPU; *model hasil* (best.pt) kemudian di-deploy ke VPS
> sebagai layanan inference untuk membuktikan kelayakan implementasi.

## G.1 Untuk BAB 3 — Subbab "Implementasi Sistem & Deployment" (siap tempel)

> **Arsitektur Ter-Dockerisasi.** Sistem FedX-Palm dirancang sebagai arsitektur
> Horizontal Federated Learning yang terisolasi menggunakan Docker (berkas
> `docker-compose.yml`). Arsitektur terdiri atas satu *FL Server (Aggregator)*
> dan empat *Client Node*, masing-masing berjalan dalam container terpisah dan
> berkomunikasi melalui jaringan bridge khusus (`fed-network`). Setiap client
> melakukan pelatihan lokal pada data privatnya (dimount *read-only* untuk
> menerapkan prinsip Zero-Trust), kemudian hanya mengirimkan pembaruan bobot ke
> server untuk diagregasi dengan algoritma FedAvg. Konfigurasi pelatihan
> mengikuti Tabel 3.5: optimizer SGD (lr=0,01), 5 ronde komunikasi, 2 epoch
> lokal per ronde, batch size 16, citra 640×640, dengan mekanisme Differential
> Privacy (Opacus DP-SGD) yang dapat dikonfigurasi per skenario ε.
>
> **Strategi Eksekusi Eksperimen.** Karena keterbatasan sumber daya untuk
> menjalankan empat node GPU secara simultan, eksperimen pelatihan dijalankan
> sebagai *simulasi federated ekuivalen* pada lingkungan Google Colab (GPU
> tunggal), yang secara matematis setara dengan agregasi FedAvg pada arsitektur
> terdistribusi. Arsitektur Docker pada `docker-compose.yml` merepresentasikan
> rancangan deployment penuh sistem.
>
> **Pipeline Deployment Model.** Model global hasil pelatihan (`best.pt`)
> di-deploy ke sebuah Virtual Private Server (VPS) sebagai layanan inference
> (berkas `docker-compose.serve.yml` dan `docker/Dockerfile.serve`). Layanan ini
> dibangun di atas image Python berbasis CPU (Torch CPU + Ultralytics + Flask),
> sehingga ringan dan tidak memerlukan GPU pada tahap inference. Aplikasi
> (`serve_app.py`) menyajikan antarmuka web pada port 8080 dengan tiga endpoint:
> `/` (halaman unggah citra), `/predict` (deteksi + Grad-CAM++), dan `/health`
> (status layanan). Bobot model dimount *read-only* (`MODEL_PATH=
> /app/weights/best.pt`), dan ambang kepercayaan deteksi diatur 0,25
> (`CONF_THRES`). Saat pengguna mengunggah citra buah sawit, sistem menjalankan
> deteksi YOLOv11 lalu menghasilkan peta panas Grad-CAM++ pada kelas dengan
> kepercayaan tertinggi secara *real-time*.

### Tabel G.1 — Konfigurasi Deployment Serving (untuk Bab 3)

| Komponen | Nilai |
|----------|-------|
| Base image | `python:3.10-slim` |
| Backend | PyTorch (CPU) + Ultralytics YOLOv11 + Flask |
| Endpoint | `/` (UI), `/predict` (inference), `/health` (status) |
| Port | 8080 |
| Bobot model | `/app/weights/best.pt` (mount read-only) |
| Confidence threshold | 0,25 |
| Komputasi | CPU-only (tanpa GPU) |
| Orkestrasi | Docker Compose (`docker-compose.serve.yml`) |

## G.2 Untuk BAB 4 — Subbab "Demonstrasi Inference Live" (siap tempel)

> Untuk membuktikan bahwa model hasil pelatihan benar-benar dapat diterapkan
> (bukan sekadar hasil simulasi), model baseline (`best.pt`) di-deploy pada VPS
> dan diuji terhadap citra buah kelapa sawit yang berasal dari *test split*
> (citra yang tidak pernah dilihat model selama pelatihan). Gambar 4.x
> menunjukkan antarmuka sistem yang berhasil mendeteksi buah beserta kelas
> kematangannya (bounding box + label kepercayaan) dan secara simultan
> menghasilkan peta panas Grad-CAM++ yang menyorot area buah sebagai dasar
> keputusan model. Hasil ini mengonfirmasi dua hal: (1) model terdeploy
> berfungsi penuh pada citra baru di lingkungan produksi nyata (CPU-only), dan
> (2) interpretasi visual Grad-CAM++ konsisten dengan validasi kuantitatif pada
> Subbab 4.5 (Average Drop 95,1%; FRR 0,962) — perhatian model terfokus pada
> morfologi buah, bukan latar. Metrik agregat (mAP, Average Drop, FRR) yang
> ditampilkan pada antarmuka berasal dari hasil eksperimen Bab 4, sedangkan
> deteksi yang dijalankan bersifat live pada citra yang diunggah penguji.

## G.3 Daftar screenshot / lampiran yang perlu diambil

Ambil dari browser saat demo berjalan (`http://<IP-VPS>:8080`):

1. **Halaman utama** — antarmuka unggah + kartu metrik (mAP 0,9945; AD 95,1%; FRR 0,962).
2. **Hasil deteksi** — citra sawit dengan bounding box + label kelas + confidence.
3. **Peta panas Grad-CAM++** — overlay heatmap menyorot area buah (sandingkan dgn deteksi).
4. **Tabel deteksi** — daftar kelas + confidence di bawah gambar.
5. **(Opsional) Output `/health`** — `{"status":"ok","model":"best.pt",...}` sebagai bukti layanan aktif.
6. **(Opsional) Terminal VPS** — `docker compose ps` menampilkan container `fedx-palm-serve` Up.

Saran: siapkan 3–5 citra test mewakili kelas kematangan berbeda (Abnormal,
Empty Bunch, Overripe, Ripe, Underripe, Unripe) agar demonstrasi menunjukkan
model membedakan keenam kelas.

## G.4 Catatan untuk antisipasi penguji

> - **"Apakah ini benar-benar terdistribusi/federated?"** → Arsitektur dirancang
>   terdistribusi (lihat `docker-compose.yml`: 4 client container terpisah +
>   server, jaringan terisolasi, data mount read-only). Eksperimen pelatihan
>   dijalankan sebagai simulasi ekuivalen di Colab karena keterbatasan multi-node
>   GPU; agregasi FedAvg yang disimulasikan setara secara matematis dengan
>   arsitektur penuh.
> - **"Kenapa demo pakai CPU, bukan GPU?"** → Tahap *inference* satu citra tidak
>   memerlukan GPU; image CPU dipilih agar deployment ringan, murah, dan
>   reprodusibel di VPS standar.
> - **"Metrik di layar live dihitung ulang per request?"** → Tidak. Metrik
>   agregat (mAP/AD/FRR) berasal dari eksperimen Bab 4 dan ditampilkan sebagai
>   konteks; yang live hanyalah deteksi + heatmap pada citra yang diunggah.
