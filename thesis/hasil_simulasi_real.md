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

| Skenario | sigma | epsilon (RDP, R1) | noise/sinyal per-elem | Global mAP@0.5 | Global mAP@0.5:0.95 |
|----------|------:|------------------:|----------------------:|---------------:|--------------------:|
| baseline   | 0.0000 | inf        | 0.0×  | **[BACA dari summary_all_scenarios.csv]** (lokal ~0.99) | [ISI] |
| sigma_1em4 | 0.0001 | 2.5×10⁸    | 0.16× | 0.0000 | 0.0000 |
| sigma_3em4 | 0.0003 | 2.8×10⁷    | 0.48× | 0.0000 | 0.0000 |
| sigma_5em4 | 0.0005 | 1.0×10⁷    | 0.8×  | 0.0000 | 0.0000 |
| sigma_1em3 | 0.0010 | 2.5×10⁶    | 1.6×  | 0.0000 | 0.0000 |
| sigma_3em3 | 0.0030 | 2.8×10⁵    | 4.8×  | 0.0000 | 0.0000 |

> Catatan: angka baseline global tidak tertangkap di potongan log (ter-truncate).
> Baca dari `summary_all_scenarios.csv` (Sel 8) atau jalankan ulang baseline.

## Diagnostik kunci (DP diag, ronde-1 klien-1)
- pre_clip_norm = 15.167, clipped = True (delta klien 15.17 → di-clip ke C=10)
- signal/elem = 6.19×10⁻³ (konstan); noise/elem = sigma × C
  - sigma_5em4: noise/elem = 5.00×10⁻³ → ratio 0.8×
  - sigma_1em3: noise/elem = 1.00×10⁻² → ratio 1.6×
  - sigma_3em3: noise/elem = 3.00×10⁻² → ratio 4.8×

## Temuan penting (untuk R2/R3)

1. **Model LOKAL tiap klien tetap sehat** pada sigma kecil (mAP lokal ~0.99 di val 50
   citra), TAPI **model GLOBAL agregat collapse ke 0** — bahkan pada sigma_1em4
   (noise 6× LEBIH KECIL dari sinyal, ratio 0.16×).
2. Pada sigma_3em3 (ratio 4.8×), training LOKAL pun mulai pecah di ronde 3–5
   (box_loss melonjak 0.5→3.3, mAP lokal jatuh ke 0.05–0.5).
3. Collapse global terjadi pada SEMUA sigma meski epsilon ≫ 10³ (R1: tanpa
   privasi bermakna) → **bukan trade-off privasi-utilitas; ini artefak
   konfigurasi pipeline DP-FedAvg** (menguatkan R3).
4. Dugaan mekanisme (untuk ablation R2): noise Gaussian ditambahkan ke SELURUH
   entri state_dict termasuk buffer BatchNorm (running_mean/running_var);
   perturbasi pada running_var berpotensi merusak normalisasi → output global
   rusak meski delta bobot kecil. Perlu dikonfirmasi via ablation (clip-only,
   BatchNorm→GroupNorm, exclude-BN-buffers).

## Verdict sementara (siap tempel ke Bab 4.3 / R3)
> "Pada konfigurasi DP-FedAvg level-klien yang diuji, agregasi dengan
> Gaussian noise menyebabkan model global kehilangan total kemampuan deteksi
> (mAP=0) pada seluruh setelan sigma, sementara model lokal tiap klien tetap
> akurat. Karena seluruh setelan berada pada rezim epsilon ≫ 10³ (tanpa privasi
> bermakna, lihat R1), collapse ini lebih konsisten dijelaskan sebagai artefak
> konfigurasi pipeline daripada trade-off privasi-utilitas yang fundamental."
