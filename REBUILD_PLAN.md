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

## Timeline 2 minggu (Hari 1-14)

### Week 1 — EKSPERIMEN

**Hari 1 (Sabtu, 13 Jun) — HARI INI**
- ✅ Buat branch
- ✅ Susun rebuild plan (file ini)
- 🔲 Feasibility test: Opacus + GN + YOLO11n di Colab/workstation
- 🔲 Plan grid σ untuk DP-SGD sweep (target ε bermakna: 1, 4, 8, ∞)
- 🔲 Decide: IIUM paper sync atau independent?

**Hari 2 (Min, 14 Jun)**
- 🔲 Implement DP-SGD training pipeline (Opacus wrapper)
- 🔲 Replace BN → GN di YOLO11n (gunakan C1b approach)
- 🔲 Sanity test: baseline DP-SGD σ=0 → harus reproduce ~0.977

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
