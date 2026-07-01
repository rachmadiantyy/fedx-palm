# FedX-Palm — REBUILD Plan (DP-SGD edition, 2 weeks)

> Branch: `claude/thesis-rebuild-dp-sgd`
> Tanggal mulai: 13 Juni 2026
> Deadline: ~27 Juni 2026 (sidang + IIUM submit)

## Narasi baru (linear, satu tema utuh)

**Kelapa Sawit → YOLOv11 → Federated Learning → Differential Privacy (DP-SGD) → XAI**

Setiap tahap menjawab kebutuhan tahap berikutnya:
1. **Kelapa Sawit** punya masalah klasifikasi kematangan TBS yang otomatis
2. **YOLOv11** memecahkan deteksi cepat 6 kelas
3. **Federated Learning** mencegah pemilik kebun mengirim citra mentah ke pusat
4. **Differential Privacy (DP-SGD)** memproteksi parameter yang dipertukarkan dari serangan inferensi
5. **XAI (Grad-CAM++)** menjelaskan keputusan model untuk dipertanggungjawabkan

## Perubahan KUNCI dari versi sebelumnya

| Aspek | Versi LAMA (DP-FedAvg) | Versi BARU (DP-SGD) |
|---|---|---|
| Mekanisme DP | Noise pada delta bobot teragregasi | Noise pada per-sample gradient |
| Library | Manual implementation | **Opacus** |
| Normalisasi | BatchNorm (default YOLO11n) | **GroupNorm** (Opacus-compatible) |
| Starting point | Random/pretrained BN | C1b baseline (GN, mAP 0.977) |
| Hasil DP yang diharap | Collapse (sudah dibuktikan) | **Trade-off curve nyata** |
| Klaim paper/thesis | Negative result + characterization | Trade-off privasi-utilitas terdokumentasi |

## Scope & venue target

**Scope inti:** Federated Learning + Differential Privacy (DP-SGD).
**XAI:** dipertahankan sebagai validasi interpretasi, BUKAN kontribusi inti.

**SKIP (out of scope untuk venue Q3 FL/ML applied):**
- Empirical Membership Inference Attack (MIA) — fitur security-pure venue
- Extensive threat model section (cukup brief 1 paragraf di Bab 3)

**Target venue Q3:** IEEE Access, Sensors (MDPI), Computers and Electronics
in Agriculture, Applied Sciences. BUKAN security-pure (Computers & Security,
TIFS) yang akan minta MIA.

**Kontribusi paper akan ada 3:**
1. HFL+YOLOv11 untuk deteksi 6-kelas TBS sawit (Non-IID realistic).
2. DP-SGD per-sampel pada object detection — privacy-utility trade-off
   curve terdokumentasi (target ε ≤ 10 dengan utility dipertahankan).
3. Comparison empiris DP-SGD vs DP-FedAvg — pemilihan mekanisme penting
   untuk detector pretrained (reuse hasil collapse DP-FedAvg sebagai
   baseline pembanding).

## Strategi compute 2-fase (adopsi dari konsultasi sebelumnya)

Daripada full multi-seed semua kombinasi, pakai 2-fase untuk hemat compute
~50% sambil tetap punya bukti statistik yang valid:

**Fase 1 — Eksplorasi (1 seed, full grid):**
- B1: 1 run
- B2 × 4 K: 4 run
- E1-FL × 4 K × 5 σ: 20 run
- E2-FL × 4 K × 5 σ: 20 run
- Total: 45 run (~200 jam)
- Tujuan: temukan "zona menarik" — (K, σ) mana yang mempertahankan mAP > 0.5
  pada ε ≤ 8

**Fase 2 — Konfirmasi (3 seed, subset menjanjikan):**
- 3 seed × {B2 K=4, E1-FL 2 (K,σ) kunci, E2-FL 2 (K,σ) kunci}
- Estimasi 15-18 run (~60 jam)
- Tujuan: mean ± std untuk angka utama paper

Total realistis: ~63 run ~260 jam (vs 51 run × 3 seed = 153 run ~600 jam
kalau semua multi-seed dari awal).

## Definisi operasional collapse

Untuk klasifikasi konsisten lintas (K, σ, strategi):
- **collapsed**: mAP@0.5 < 0.05 (tidak berguna, model gagal belajar)
- **degraded**: 0.05 ≤ mAP@0.5 < 0.70 (training jalan tapi tidak praktis)
- **acceptable**: 0.70 ≤ mAP@0.5 < 0.90 (trade-off privasi yang berguna)
- **excellent**: mAP@0.5 ≥ 0.90 (mendekati baseline)

## Catatan hipotesis K untuk DP-SGD per-sampel

Berbeda dari hipotesis untuk DP-FedAvg (di mana K besar = noise lebih kecil),
pada DP-SGD per-sampel **arah K terbalik**:

- K kecil → samples-per-klien besar → DP-SGD stabil → utilitas lebih baik
- K besar → samples-per-klien kecil → noise mendominasi sinyal → collapse cepat

Untuk K=16 dengan dataset ~7200 train, samples-per-klien ~450 → di ambang
batas konvergensi DP-SGD. Hasil K=16 akan dilaporkan sebagai *limit study*.

## Workflow PARALEL (write + train berbarengan)

Setelah user pilih full grid K {2,4,8,16} x sigma full (45 main runs ~200h),
compute time melebihi sisa hari. Tidak realistis tunggu training selesai
baru menulis. Track A dan B HARUS jalan paralel:

```
Hari:  3  4  5  6  7  8  9  10 11 12 13 14
A:    [============= TRAIN ==================|R1 multi-seed]
B:        [=== Bab 1,2,3 draft ===|Bab 4,5 sambil hasil masuk]
```

- **Track A (Compute, background)**: Workstation 24/7 jalanin
  `run_full_grid.py`. User cek pagi/sore: progress di
  `thesis_rebuild/runs/`, error log, restart kalau crash.
- **Track B (Writing, foreground)**: Aku draft Bab 1 (Pendahuluan),
  Bab 2 (Landasan Teori), Bab 3 (Metodologi) -- semua bisa ditulis
  TANPA hasil eksperimen final. Mulai Hari 3 paralel dengan training.
- **Track C (Analisis, on demand)**: Begitu CSV phase keluar
  (b2_fl_summary, e1_fl_full_grid, e2_fl_full_grid), aku langsung
  generate tabel/plot dan tempelkan ke Bab 4.

## Timeline 2 minggu (Hari 1-14)

### Week 1 — EKSPERIMEN

**Hari 1 (Sabtu, 13 Jun) — DONE ✅**
- ✅ Buat branch + PR #3
- ✅ Susun rebuild plan + outline 5 bab
- ✅ Feasibility test PASSED di workstation RTX 4080:
  - Opacus 1.5.4 + PyTorch 2.5.1+cu121 + Ultralytics 8.4.51
  - 81 BatchNorm → GroupNorm conversion clean
  - PrivacyEngine.make_private wrap YOLOv11n-GN sukses
  - PRVAccountant default (lebih tight dari RDP)
- ✅ Scope decision: FL + DP-SGD core, XAI sebagai validasi
- ✅ Experiment design final: B1/B2/B3/E1/E2/R1
- ✅ Decision IIUM: rewrite paper jadi DP-SGD (submit Week 2)

**Data pipeline from-zero (reproducible, seed=42) — lihat thesis_rebuild/PIPELINE.md**
- ✅ `00_download_dataset.py` — download fresh dari Roboflow
- ✅ `01_resplit_bunch_id.py` — anti-leakage stratified group split 80/10/10
- ✅ `02_dirichlet_partition.py` — Non-IID 4-client (alpha 0.1/0.3/0.5/0.7)
- Tidak ada reuse dataset lama; semua dibangun ulang dari nol untuk
  reproducibility thesis. data/ di-gitignore (regenerable).

**Hari 2 (Min, 14 Jun) — IN PROGRESS**
- ✅ `utils/gn_convert.py` — reusable BN→GN swap (extracted from feasibility)
- ✅ `train_b1_centralized.py` — B1 wrapper (Ultralytics native + GN swap)
- ✅ `utils/yolo_dp_loop.py` — custom DP-SGD training loop:
  - reuses Ultralytics `v8DetectionLoss`, `build_yolo_dataset`
  - bypasses BaseTrainer DDP/AMP plumbing
  - wraps optimizer + dataloader via PrivacyEngine BEFORE loop
  - supports `freeze_backbone` flag for E2
- ✅ `train_e1_dp_sgd_full.py` — sweep wrapper (σ ∈ {0.5,1,1.5,2,3})
- ✅ `train_e2_dp_sgd_partial.py` — sweep wrapper with backbone freeze
- 🔲 **NEXT (user runs)**: smoke test B1 on workstation, then E1 single-σ
- 🔲 Add per-epoch DetectionValidator → fills `best_mAP50` in CSV (Day 3)
- 🔲 Wire federation on top (B2 + federated DP-SGD) (Day 3)

**Known TODOs in Day 2 scripts (deferred to Day 3):**
- No per-epoch val (saves `final.pt` instead of `best.pt`) — Day 3 adds
  `DetectionValidator` integration so we track best-on-val mAP
- No federated wrapper yet — Day 3 builds B2 by running B1-style train
  per client + FedAvg aggregation; same applies to FL variant of E1/E2
- Possible Opacus quirk: YOLO loss returns batch-summed loss; per-sample
  hooks rely on this being differentiable per sample. Smoke test on
  workstation confirms. Fallback: BatchMemoryManager for memory safety

**Hari 3-4 (Sen-Sel, 15-16 Jun)**
- 🔲 Run σ sweep DP-SGD: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0}
- 🔲 Compute ε per σ via RDP accountant (Opacus built-in)
- 🔲 Save best.pt per skenario + log metrics

**Hari 5 (Rab, 17 Jun)**
- 🔲 Multiple seeds (3 seed) untuk baseline + 2-3 σ kunci
- 🔲 Generate confusion matrix + per-class AP per skenario

**Hari 6 (Kam, 18 Jun)**
- 🔲 Generate figures: privacy_utility_curve, convergence per σ
- 🔲 Re-run XAI (Grad-CAM++) pada model DP-SGD berhasil
- 🔲 Compute Average Drop & FRR untuk DP-trained model

**Hari 7 (Jum, 19 Jun)**
- 🔲 Buffer / fix issues
- 🔲 Compile semua hasil ke `hasil_eksperimen_dpSGD.md`

### Week 2 — PENULISAN THESIS

**Hari 8 (Sab, 20 Jun)**
- 🔲 Tulis Bab 4 (Hasil): subbab baseline FL, subbab DP-SGD sweep, subbab XAI
- 🔲 Insert semua tabel dan figure

**Hari 9 (Min, 21 Jun)**
- 🔲 Tulis Bab 5 (Kesimpulan): kontribusi, limitations, future work
- 🔲 Abstrak final

**Hari 10 (Sen, 22 Jun)**
- 🔲 Update Bab 1 (Pendahuluan): rumusan masalah disesuaikan, hipotesis updated
- 🔲 Update Bab 3 (Metodologi): DP-SGD section, Opacus setup, GN backbone

**Hari 11 (Sel, 23 Jun)**
- 🔲 Update Bab 2 (Landasan Teori): tambah DP-SGD detail, Rényi DP, Opacus
- 🔲 References cleanup (Abadi 2016 DP-SGD jadi primary, geyer & mcmahan jadi context)

**Hari 12 (Rab, 24 Jun)**
- 🔲 Rewrite IIUM paper jadi DP-SGD version
- 🔲 OR: submit DP-FedAvg version (paper v5 lama) ke IIUM, thesis fokus DP-SGD

**Hari 13 (Kam, 25 Jun)**
- 🔲 Polish: typo, format, table/figure numbering
- 🔲 Compile thesis final PDF

**Hari 14 (Jum, 26 Jun)**
- 🔲 Buffer + practice presentation
- 🔲 Submit IIUM paper

**Hari 15 (Sab, 27 Jun) — DEADLINE/SIDANG**

## Risk register

| Risk | Mitigation |
|---|---|
| Opacus + YOLO11n + GN tidak jalan | Day 1: test feasibility. Fallback: pakai custom per-sample gradient |
| DP-SGD juga collapse di semua σ | Tetap punya kontribusi: comparison DP-SGD vs DP-FedAvg sebagai sub-bab |
| Waktu mepet rewrite Bab 1-3 | Reuse 70% dari versi sebelumnya, edit DP section saja |
| GPU tidak tersedia full 2 minggu | Gunakan RunPod ($0.34/jam × 20 jam total = ~$7) |

## Decision points untuk hari ini

1. **Paper IIUM**: submit versi lama (DP-FedAvg) atau tunggu DP-SGD versi?
2. **GPU**: workstation, RunPod, atau Colab Pro?
3. **Backbone**: YOLO11n-GN dari scratch atau adapt pretrained?
