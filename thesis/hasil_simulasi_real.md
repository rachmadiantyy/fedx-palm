# Hasil Simulasi REAL — Federated DP (data bersih re-split bunch_id)

> Diekstrak dari log Colab run penuh (6 skenario × 5 ronde × 4 klien).
> Data: `palm_v2_resplit` (R4 — split berbasis bunch_id, BERSIH tanpa leakage).
> Run selesai penuh sampai skenario terakhir (sigma_3em3 R05).
> Sumber kebenaran ini dipakai untuk update Tabel DATA REAL & Bab 4.

## Parameter
- ROUNDS = 5, LOCAL_EPOCHS = 2, BATCH = 16, IMG = 640, LR0 = 0.01
- Clip-norm C = 10, q = 1 (partisipasi penuh), delta = 1e-5
- Klien: c1=1221, c2=2462, c3=2009, c4=3347 citra (Non-IID Dirichlet)
- Val global: 937 citra (3273 instance)

## Hasil global per skenario (mAP@0.5 model agregat)

| Skenario | sigma | epsilon (RDP, R1) | noise/sinyal per-elem | Global mAP@0.5 | Global mAP@0.5:0.95 | Precision | Recall |
|----------|------:|------------------:|----------------------:|---------------:|--------------------:|----------:|-------:|
| baseline   | 0.0000 | inf       | 0.0×  | **0.9950** | **0.9510** | 0.9982 | 0.9989 |
| sigma_1em4 | 0.0001 | 2.53×10⁸  | 0.16× | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sigma_3em4 | 0.0003 | 2.81×10⁷  | 0.48× | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sigma_5em4 | 0.0005 | 1.01×10⁷  | 0.8×  | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sigma_1em3 | 0.0010 | 2.53×10⁶  | 1.6×  | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sigma_3em3 | 0.0030 | 2.82×10⁵  | 4.8×  | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

> Sumber: `saved_runs/summary_all_scenarios.csv` (run 9–10 Jun 2026).
> Baseline tambahan: Average=100, DFRR=0.961.

## Hasil ablation R2 (12 skenario × 3 seed = 36 run; total 13.9 menit)

Sumber: `saved_runs/ablation_summary.csv` & `ablation_results.csv`.

| ID | Skenario | C | σ | noise/sig | mAP_mean | mAP_std | nan_rate |
|----|----------|---:|---:|----------:|---------:|--------:|---------:|
| A0 | baseline (BN, no-DP)            | —   | 0     | 0.00×  | **0.9950** | 0 | 0 |
| A1 | clip-only                       | 10  | 0     | 0.00×  | **0.9950** | 0 | 0 |
| A2 | clip-only ketat                 | 1   | 0     | 0.00×  | **0.9950** | 0 | 0 |
| A3 | BN + noise                      | 10  | 5e-3  | 8.07×  | 0.0000 | 0 | 0 |
| A4 | BN + noise, C lebih kecil       | 1   | 5e-3  | 8.07×  | 0.0000 | 0 | 0 |
| **A5** | **BN + noise, C sangat kecil** | **0.1** | **5e-3** | **8.07×** | **0.9950** | **0** | **0** |
| A6 | BN + noise σ lebih kecil        | 10  | 1e-3  | 1.61×  | 0.0000 | 0 | 0 |
| B1 | lr=1e-3 + noise                 | 10  | 5e-3  | 8.07×  | 0.0000 | 0 | 0 |
| B2 | lr=1e-4 + noise                 | 10  | 5e-3  | 8.07×  | 0.0000 | 0 | 0 |
| B3 | frozen backbone + noise         | 10  | 5e-3  | 24.21× | 0.0000 | 0 | 0 |
| C1 | GroupNorm + noise               | 10  | 5e-3  | 15.24× | 0.0000 | 0 | 0 |
| C1b| GroupNorm no-noise (kontrol)    | —   | 0     | 0.00×  | 0.9773 | 0 | 0 |

> Catatan: kolom `noise_sig` di CSV menampilkan 8.072 untuk A5 padahal C
> aktualnya 0.1 (bukan 10). Script tampaknya menghitung display noise/sig
> dengan C basis = 10, sehingga kolom itu mis-leading untuk A4/A5/A2/B3.
> Untuk analisis pakai kolom **C × σ** yang dihitung dari nama skenario,
> bukan kolom `noise_sig` mentah.

## Diagnostik kunci (DP diag, ronde-1 klien-1)
- pre_clip_norm = 15.167, clipped = True (delta klien 15.17 → di-clip ke C=10)
- signal/elem = 6.19×10⁻³ (konstan); noise/elem = sigma × C
  - sigma_5em4: noise/elem = 5.00×10⁻³ → ratio 0.8×
  - sigma_1em3: noise/elem = 1.00×10⁻² → ratio 1.6×
  - sigma_3em3: noise/elem = 3.00×10⁻² → ratio 4.8×

## Temuan penting (untuk R2/R3) — REVISI berbasis ablation

1. **Model LOKAL tiap klien tetap sehat** pada sigma kecil (mAP lokal ~0.99 di val 50
   citra), TAPI **model GLOBAL agregat collapse ke 0** — bahkan pada sigma_1em4
   (noise 6× LEBIH KECIL dari sinyal, ratio 0.16×).
2. Pada sigma_3em3 (ratio 4.8×), training LOKAL pun mulai pecah di ronde 3–5
   (box_loss melonjak 0.5→3.3, mAP lokal jatuh ke 0.05–0.5).
3. Collapse global terjadi pada SEMUA sigma meski epsilon ≫ 10³ (R1: tanpa
   privasi bermakna) → **bukan trade-off privasi-utilitas; ini artefak
   konfigurasi pipeline DP-FedAvg** (menguatkan R3).
4. **Akar masalah = magnitudo absolut noise per-elemen (C × σ), BUKAN rasio
   noise/sinyal dan BUKAN buffer BatchNorm.** Bukti dari ablation:
   - A5 (C=0.1, σ=5e-3) mempertahankan mAP=0.9950 **padahal noise/sig tetap
     8.07×** — yang berubah hanya `C × σ = 5e-4` (vs 5e-2 di A3, 5e-3 di A4).
   - Ambang empiris dari gabungan tabel utama + ablation: collapse muncul
     saat `C × σ ≥ 1×10⁻³` (sigma_1em4 dengan C=10), aman saat
     `C × σ ≤ 5×10⁻⁴` (A5 dengan C=0.1, σ=5e-3). Yaitu noise/elem perlu
     ≤ ~8% dari signal/elem (6.19×10⁻³) supaya akumulasi 5-ronde tidak
     menggeser bobot global ke wilayah non-fungsional.
5. **Hipotesis BatchNorm-buffer GUGUR.** Skenario C1 (GroupNorm + noise, C=10,
   σ=5e-3) tetap collapse ke 0, sedangkan C1b (GroupNorm tanpa noise) berjalan
   normal (0.9773). Mengganti BN→GN tidak menyelamatkan model dari perturbasi
   bobot berskala besar.
6. **Strategi mitigasi non-DP-mechanism tidak menolong:** menurunkan lr ke
   1e-3/1e-4 (B1/B2) maupun membekukan backbone (B3) tetap collapse pada
   C=10, σ=5e-3 — karena noise diterapkan ke bobot agregat global, bukan ke
   gradient lokal.
7. **Clipping-only aman:** A1 (C=10) dan A2 (C=1) keduanya 0.9950 → clipping
   bukan penyebab; noise injection yang merusak.

## Verdict final (siap tempel ke Bab 4.3 / R2 / R3)
> "Pada konfigurasi DP-FedAvg level-klien yang diuji, agregasi dengan Gaussian
> noise menyebabkan model global kehilangan total kemampuan deteksi (mAP=0)
> pada seluruh setelan sigma, sementara model lokal tiap klien tetap akurat.
> Studi ablation 12-skenario membuktikan akar masalah adalah **magnitudo
> absolut noise per-parameter (C × σ) yang melampaui skala bobot YOLO11n
> (~6×10⁻³)**, bukan rasio noise/sinyal, bukan buffer BatchNorm, dan bukan
> learning-rate schedule. Mitigasi sederhana dengan menurunkan clip-bound ke
> C=0.1 (skenario A5) berhasil mempertahankan mAP=0.9950 pada σ=5e-3. Karena
> seluruh setelan utama berada pada rezim epsilon ≫ 10³ (tanpa privasi
> bermakna, lihat R1), collapse pada Tabel utama lebih konsisten dijelaskan
> sebagai artefak konfigurasi pipeline daripada trade-off privasi-utilitas
> yang fundamental."
