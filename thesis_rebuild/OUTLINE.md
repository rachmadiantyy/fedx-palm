# Outline Thesis FedX-Palm (DP-SGD edition)

Mengikuti alur: **Kelapa Sawit → YOLOv11 → FL → DP-SGD → XAI**

**Scope thesis & jurnal target:**
- Kontribusi inti: **Federated Learning + Differential Privacy (DP-SGD)**
- XAI sebagai validasi interpretasi (bukan kontribusi inti)
- **Bukan** security-deep venue (skip MIA empirical attack, skip extensive threat model)
- Target: Q3 FL/ML applied venue (mis. IEEE Access, Sensors MDPI, Computers and
  Electronics in Agriculture)

---

## BAB 1 — PENDAHULUAN

### 1.1 Latar Belakang
1. **Industri kelapa sawit** Indonesia/Malaysia — produsen utama dunia, kebutuhan
   otomasi penilaian kematangan TBS (subjektif → AI standar).
2. **YOLOv11** sebagai detektor 6-kelas real-time yang ringan untuk edge deployment.
3. **Data berskala besar diperlukan** → tapi citra perkebunan = informasi komersial
   sensitif → motivasi **Federated Learning (HFL)**.
4. **Parameter model masih bocor** → membership inference, model inversion → motivasi
   **Differential Privacy (DP-SGD)**.
5. **Model deep tetap black-box** → agronom sulit percaya → motivasi **XAI (Grad-CAM++)**.

### 1.2 Rumusan Masalah
RQ1: Bagaimana melatih YOLOv11 secara kolaboratif (FL) tanpa memindahkan citra mentah?
RQ2: Bagaimana DP-SGD per-sampel pada client lokal **mempertahankan utilitas** sambil mencapai privasi formal yang bermakna (ε ≤ 8)?
RQ3: Bagaimana Grad-CAM++ memvalidasi keputusan model FL+DP secara visual & kuantitatif?

### 1.3 Tujuan
1. Membangun arsitektur HFL-YOLOv11 untuk deteksi 6 kelas TBS sawit.
2. Menerapkan DP-SGD via Opacus pada client lokal, mengukur kurva trade-off ε vs mAP.
3. Mengembangkan modul Grad-CAM++ untuk validasi visual + metrik Average Drop / FRR.

### 1.4 Batasan
- 6 kelas (Abnormal, Empty Bunch, Overripe, Ripe, Underripe, Unripe).
- Backbone YOLOv11n dengan **GroupNorm** (kompatibilitas Opacus).
- K=4 client (cross-silo), T=5 ronde, Non-IID Dirichlet.
- DP-SGD per-sample gradient (BUKAN DP-FedAvg level-klien).
- Simulasi single-GPU + Dockerized deployment blueprint.

### 1.5 Metode (high-level — detail di Bab 3)
8 tahap: studi literatur → perancangan → dataset → local training → DP-SGD →
agregasi → XAI → evaluasi.

### 1.6 Hipotesis
- **H1**: HFL baseline mencapai mAP@0.5 > 0.9.
- **H2**: DP-SGD menghasilkan trade-off gradual antara ε dan mAP (bukan collapse).
- **H3**: Grad-CAM++ pada model FL+DP tetap menyorot area buah (AD > 80%, FRR > 0.8).

### 1.7 Sistematika

---

## BAB 2 — LANDASAN TEORI

### 2.1 Kelapa Sawit & TBS
- Klasifikasi kematangan (6 kelas)
- Praktik panen tradisional (visual manual)
- Penelitian deteksi kematangan sawit sebelumnya

### 2.2 Object Detection & YOLOv11
- Evolusi YOLO
- Arsitektur YOLOv11n: backbone-neck-head
- CIoU loss, Distribution Focal Loss

### 2.3 Federated Learning
- FedAvg (McMahan 2017)
- Cross-silo vs cross-device (Kairouz 2021)
- Non-IID & Dirichlet partition (Hsu 2019)

### 2.4 Differential Privacy (DP-SGD)
- Formal definition (ε, δ)-DP (Dwork)
- DP-SGD algorithm (Abadi 2016) — clipping + Gaussian noise per-sample
- Rényi DP accountant (Mironov 2017), zCDP (Bun-Steinke 2016)
- Privacy amplification by subsampling
- **Mengapa per-sample gradient lebih ramah optimizer** dibanding noise pada bobot

### 2.5 Threat Model di FL
- Membership inference (Shokri 2017)
- Model inversion (Fredrikson 2015)
- Gradient leakage (Zhu 2019)
- Bagaimana DP-SGD menutup celah ini

### 2.6 BatchNorm vs GroupNorm
- BatchNorm: dependent across batch → tidak compatible dgn per-sample DP
- GroupNorm (Wu & He 2018): per-sample → compatible dgn Opacus

### 2.7 Opacus
- Library DP-SGD untuk PyTorch
- ModuleValidator constraint (no BN, no in-place ops)
- PrivacyEngine wrapper

### 2.8 Explainable AI (XAI)
- Grad-CAM (Selvaraju 2017)
- Grad-CAM++ (Chattopadhyay 2018)
- Metrik faithfulness: Average Drop, FRR

### 2.9 Container Deployment (Docker)
- Konsep deployment blueprint
- Multi-container architecture (server + clients)

---

## BAB 3 — METODOLOGI

### 3.1 Arsitektur Sistem
- 1 server aggregator + 4 client nodes
- Bridge network terisolasi
- Volume mount read-only (zero-trust)
- Docker-compose blueprint

### 3.2 Dataset
- Roboflow palm-fruit-ripeness-detection v2
- 6 kelas urutan alfabet
- Re-split bunch_id-based (anti-leakage)
- Train 9030 / Val 935 / Test held-out
- Dirichlet α ∈ {0.1, 0.3, 0.5, 0.7} untuk 4 client

### 3.3 Model
- YOLOv11n
- **Modifikasi**: BatchNorm → GroupNorm (untuk Opacus compatibility)
- Pretrained init dari COCO + adapt GN

### 3.4 Pelatihan Lokal
- SGD, lr=0.01, batch=16, image 640×640
- YOLO loss: CIoU + cls + DFL
- 2 epoch per round

### 3.5 DP-SGD (KUNCI BAB INI)
- Per-sample gradient computation (vmap atau Opacus)
- Per-sample clipping ke ℓ2-norm C
- Add Gaussian N(0, σ²C²) ke gradient
- Aggregate per-batch lalu update
- Privacy budget tracking via Opacus PrivacyEngine
- Setup grid (C, σ) → eksperimental nanti

### 3.6 Agregasi FedAvg
- Weighted average bobot client per ronde
- 5 ronde komunikasi

### 3.7 XAI Grad-CAM++
- Target layer: terakhir sebelum head
- Average Drop & FRR formulasi

### 3.8 Metrik Evaluasi
- mAP@0.5, mAP@0.5:0.95
- Precision, Recall, F1
- ε via PRV accountant (Opacus default, lebih tight dari RDP) — δ=1e-5
- XAI: AD, FRR

### 3.9 Skenario Eksperimen

**Baselines (no DP):**
| ID | Setup | Tujuan | Compute (RTX 4080) |
|---|---|---|---|
| **B1** | Centralized YOLOv11n-GN, no DP | Upper bound utility | ~2 jam |
| **B2** | Federated YOLOv11n-GN, no DP, K=4 Non-IID α=0.5 | Mengukur "FL cost" murni | ~3 jam |
| **B3** | Federated + DP-FedAvg (reuse hasil ablation collapse) | Pembanding mekanisme DP | 0 jam (data eksis) |

**Experiments DP-SGD (per-sampel via Opacus):**
| ID | Setup | Sweep | Tujuan | Compute |
|---|---|---|---|---|
| **E1** | Full DP-SGD (semua 2.6M param) | σ ∈ {0.5, 1.0, 1.5, 2.0, 3.0} | Privacy-utility curve utama | ~15 jam |
| **E2** | Partial DP-SGD (backbone frozen, ~0.2M head param) | σ ∈ {0.5, 1.0, 1.5, 2.0, 3.0} | Tunjukkan dimensionality matter (Tramer & Boneh 2021) | ~12 jam |

**Robustness:**
| ID | Setup | Tujuan | Compute |
|---|---|---|---|
| **R1** | 3 seed × {B2, E1 best σ, E1 worst feasible σ} | Statistical robustness | ~6 jam |

**Total compute: ~38 jam single-GPU → 1.5-2 hari training**

**Pilihan σ-grid:** dipilih untuk meng-cover rentang ε bermakna dari ~1 (privasi
ketat) hingga ~10+ (privasi longgar), dengan target sweet spot ε ≈ 4-8.

**Catatan teknis:**
- GroupNorm WAJIB di semua skenario (Opacus tidak support BatchNorm).
  Bukan eksperimen terpisah — ini foundation.
- Privacy accountant: PRVAccountant (Opacus 1.5.4 default, lebih tight dari RDP).
- max_grad_norm (C) initial = 1.0; ablation C opsional di akhir kalau ada waktu.

### 3.10 Lingkungan Implementasi
- Hardware: workstation RTX 4080 (training), VPS (Docker deployment blueprint)
- Software stack: PyTorch 2.5.1+cu121, Ultralytics 8.4.51, **Opacus 1.5.4**
- Per-sample gradient: Opacus `GradSampleModule` (functorch backend di PyTorch 2.5)
- OS: Windows 11 + miniconda env `fedx`

---

## BAB 4 — HASIL DAN PEMBAHASAN

### 4.1 Validasi Setup (B1)
- Sanity test: Centralized GN baseline mencapai mAP ≥ baseline literatur
- Reproduce hasil ablation C1b (~0.977) sebagai cross-check

### 4.2 Baseline FL no-DP (B2)
- 5 ronde komunikasi, 4 client Non-IID α=0.5
- mAP@0.5, mAP@0.5:0.95, P, R
- Tabel konvergensi per round
- Per-class AP@0.5 + confusion matrix
- FL cost = (B1 - B2) mAP

### 4.3 DP-SGD Full Sweep (E1)
- Tabel: σ ∈ {0.5, 1, 1.5, 2, 3}, ε (PRV), mAP@0.5, mAP@0.5:0.95
- Plot privacy-utility curve (mAP vs ε log-scale)
- Diskusi: titik trade-off optimal
- Verdict H2

### 4.4 DP-SGD Partial Sweep (E2): Backbone frozen, head only
- Same σ-grid sebagai E1, tapi hanya head (~0.2M param) yang trainable
- Tabel & curve side-by-side dengan E1
- Hipotesis verifikasi: E2 dominate E1 pada ε rendah (high privacy)
  karena noise budget tersebar di parameter jauh lebih sedikit
- Reference: Tramer & Boneh 2021 ("DP learning needs better features")

### 4.5 Comparison: DP-FedAvg vs DP-SGD (B3 vs E1/E2)
- Reuse hasil eksperimen DP-FedAvg lama sebagai baseline pembanding
- Tabel side-by-side: ε vs mAP untuk tiga mekanisme
- Highlight: DP-FedAvg collapse pada σ kecil; DP-SGD memberi trade-off gradual
- Diagnosis: kenapa DP-SGD lebih ramah optimizer (noise di gradient bisa
  di-dampen oleh momentum, vs noise pada bobot teragregasi yang permanen)
- Kontribusi: pemilihan mekanisme DP penting untuk object detection

### 4.5b Statistical Robustness (R1)
- 3 seed × {B2, E1 best σ, E1 worst feasible σ}
- Mean ± std table, confidence interval untuk angka utama

### 4.6 XAI Validation
- Grad-CAM++ pada FL baseline (AD, FRR)
- Grad-CAM++ pada FL+DP terbaik (AD, FRR — harus tetap > threshold)
- Comparison heatmap visual

### 4.7 Validasi Hipotesis
- H1 ✓ kalau baseline > 0.9
- H2 ✓ kalau ada kurva (bukan cliff)
- H3 ✓ kalau Grad-CAM++ masih meaningful pada DP-trained

### 4.8 Threats to Validity

---

## BAB 5 — KESIMPULAN DAN SARAN

### 5.1 Kesimpulan
- HFL+FedAvg+DP-SGD+XAI: integrated framework yang berfungsi
- Trade-off ε vs mAP konkret terdokumentasi
- XAI tetap valid pada model dengan privasi formal

### 5.2 Kontribusi
1. Sistem HFL ter-Dockerisasi untuk deteksi sawit (mAP > 0.9)
2. Empirical trade-off curve DP-SGD untuk object detection (gap di literatur)
3. Validasi XAI tetap meaningful pada model DP-trained

### 5.3 Keterbatasan
- K=4 cross-silo (bukan large-scale cross-device)
- Single dataset, single backbone
- Single-GPU simulation

### 5.4 Saran Penelitian Lanjutan
- Multi-dataset cross-domain
- Backbone scaling (YOLOv11s, m, l)
- Real distributed deployment dengan Docker multi-host
- Secure aggregation sebagai komplemen DP
