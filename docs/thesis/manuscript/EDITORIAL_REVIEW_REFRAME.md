# Editorial Review & Reframe — JUTIF Manuscript
### Multi-Seed Reproducibility and Convergence in Non-IID Federated Oil Palm FFB Detection

*Catatan editorial dan peringatan risiko ditulis dalam Bahasa Indonesia. Seluruh bagian manuskrip revisi ditulis dalam bahasa Inggris akademik formal, sesuai instruksi.*

*Sumber: `docs/thesis/manuscript/manuscript.tex` (versi commit terbaru di branch `claude/tesis-b1-b2-c77afk`). Tidak ada PDF terpisah yang dilampirkan pada permintaan ini — file `.tex` di repo sudah dibaca penuh sebagai sumber utama, dan seluruh angka di bawah dicek silang terhadap file itu (semuanya cocok dengan angka yang kamu berikan di prompt, tidak ada koreksi yang diperlukan).*

---

## 1. Editorial Verdict

Manuskrip saat ini **ilmiah valid tapi framing-nya salah sasaran** untuk JUTIF. Ia disajikan sebagai "centralized vs federated comparison" padahal perbandingan itu sendiri **belum matched** (B1 diuji di held-out test, B2 baru di validation) — sehingga judul dan abstract-nya menjanjikan sesuatu yang datanya sendiri belum bisa buktikan secara ketat. Nilai ilmiah yang sebenarnya paling kuat dan **sudah didukung penuh oleh data yang ada** adalah: tiga training seed menunjukkan federated learning di bawah partisi Non-IID tetap *reproducible* (varians kecil, SD=0.0157) meski *waktu konvergensinya* sangat berbeda (best round 9 vs 11 vs 40). Itu cerita yang lebih jujur, lebih kuat secara metodologis, dan tidak tumpang tindih dengan manuskrip JIKI kamu.

**Rekomendasi: reframe, jangan submit versi saat ini.** Reframe ini tidak memerlukan eksperimen baru untuk draft awal — semua angka yang dibutuhkan versi reframe sudah ada di manuskrip saat ini. Yang masih dibutuhkan sebelum submission (evaluasi B2 di held-out test, breakdown per-kelas B2, client-by-class distribution) adalah *penguat*, bukan syarat mutlak untuk mulai revisi framing.

## 2. Penilaian Risiko Overlap dengan JIKI

**Risiko rendah**, asalkan reframe di bawah diikuti. Manuskrip JIKI kamu (Grad-CAM++, Average Drop, Focus Retention Rate, faithfulness) memakai B1/B2 hanya sebagai *baseline angka* untuk analisis XAI-nya — tidak membahas convergence trajectory, inter-seed variance, atau client heterogeneity sama sekali. Manuskrip JUTIF versi reframe below fokus persis di area yang JIKI **tidak** sentuh. Risiko yang tersisa hanya kalau kamu membiarkan versi lama (title "Centralized vs Federated") tetap dipakai — itu berisiko dibaca sebagai versi ringkas dari temuan akurasi yang sama dengan yang mendasari JIKI (self-plagiarism/salami-slicing risk), karena kontribusinya tidak berdiri sendiri.

## 3. Tabel Pembeda JIKI vs JUTIF

| Aspek | JIKI (sudah ada) | JUTIF (reframe) |
|---|---|---|
| Pertanyaan inti | Apakah penjelasan model (Grad-CAM++) bisa dipercaya untuk mendukung keputusan panen? | Apakah federated training di bawah partisi Non-IID *reproducible* dan *stabil* lintas seed? |
| Peran B1/B2 | Baseline angka akurasi saja, dipakai sebagai konteks untuk analisis XAI | **Objek penelitian utama** — convergence, variance, per-round behavior |
| Metrik utama | Average Drop, Focus Retention Rate, Grad-CAM++ heatmap | mAP50/mAP50-95 per-seed, best-vs-final round, inter-seed SD, per-class stability |
| Federated learning | Disebutkan sebagai konteks deployment, tidak dianalisis mendalam | Dianalisis langsung: dampak Dirichlet partition, client imbalance, FedAvg sample-weighting |
| Sentralized model | Baseline akurasi untuk XAI | Reference baseline saja (bukan fokus) |
| Explainability | Kontribusi utama | **Tidak dibahas sama sekali** |

**Topik yang tidak boleh diulang sebagai kontribusi utama di JUTIF**: Grad-CAM++, Average Drop, Focus Retention Rate, faithfulness metric apa pun, trustworthy/human-review decision support framing. Boleh disebut sekali di Future Work sebagai penghubung ke studi terpisah (dengan sitasi ke paper JIKI setelah publish), tapi tidak boleh jadi bagian dari abstract/contributions/discussion utama.

## 4. Lima Alternatif Judul

1. **"Multi-Seed Reproducibility and Convergence Stability of Non-IID Federated YOLO11n for Oil Palm Fresh Fruit Bunch Detection"**
2. "Convergence Dynamics of FedAvg Under Non-IID Client Partitions: A Multi-Seed Study on Oil Palm Ripeness Detection"
3. "How Reproducible is Federated Object Detection Under Non-IID Data? A Three-Seed Evaluation on Oil Palm FFB Ripeness"
4. "Reproducibility and Robustness of Non-IID Federated YOLO11n Training for Six-Class Oil Palm Ripeness Detection"
5. "Seed Variance and Best-Checkpoint Behavior in Non-IID Federated Learning for Oil Palm Fresh Fruit Bunch Detection"

## 5. Judul Terpilih

**Judul #1: "Multi-Seed Reproducibility and Convergence Stability of Non-IID Federated YOLO11n for Oil Palm Fresh Fruit Bunch Detection"**

Alasan: (a) memuat kata kunci yang jadi kontribusi inti (*multi-seed*, *reproducibility*, *convergence stability*, *Non-IID federated*) tanpa menjadikan "vs centralized" sebagai novelty; (b) 15 kata, sesuai batas; (c) tidak menyebut DP; (d) langsung membedakan diri dari judul lama yang framing-nya "comparison"; (e) cukup spesifik untuk tidak berbenturan dengan judul JIKI kamu yang fokus XAI/Grad-CAM++.

## 6. Revised Research Question

**Primary RQ**: Under a fixed Non-IID client partition (Dirichlet $\alpha=0.5$, $K=4$), how reproducible is FedAvg-trained YOLO11n's detection performance across independent stochastic training seeds, and how consistent is its convergence behavior across communication rounds?

**Supporting RQ 1**: Does peak validation performance remain stable across training seeds even when the communication round at which that peak occurs varies substantially? *Testable interpretation*: if peak mAP50 SD is small (≤0.02) while best-round varies widely (e.g., round 9 vs. round 40), this indicates outcome-level reproducibility without trajectory-level reproducibility — a distinction future federated deployments should account for in checkpoint-selection policy.

**Supporting RQ 2**: Does severe client data-quantity imbalance (here, 8.67%–59.36% of training images per client) destabilize FedAvg convergence under sample-weighted aggregation? *Testable interpretation*: if the largest client (59.36% share) correlates with faster or more stable convergence across seeds, this is consistent with — but does not prove — sample-weighted aggregation being dominated by the largest client; a controlled ablation (e.g., uniform-weighted FedAvg) would be needed to confirm causally.

**Supporting RQ 3 (conditional on Task O checklist item being completed)**: Is federated training's per-class detection stability consistent with its global mAP stability, or do specific ripeness classes show disproportionate inter-seed variance that a single aggregate metric would hide? *Testable interpretation*: per-class AP50 SD across seeds, once available, should be compared directly against the global mAP50 SD (0.0157) reported here.

## 7. Revised Contributions

(1) A multi-seed (n=3) empirical evaluation of FedAvg under a fixed Non-IID Dirichlet partition ($\alpha=0.5$, $K=4$) for six-class oil palm FFB ripeness detection using YOLO11n, reporting outcome-level reproducibility (mean validation mAP50 = 0.8775, SD = 0.0157).

(2) A convergence-trajectory analysis across 40 communication rounds per seed, showing that the best-validation checkpoint round varies substantially across seeds (round 9, 11, and 40) despite convergent final performance — distinguishing outcome reproducibility from trajectory reproducibility.

(3) A quantitative characterization of federated client heterogeneity by training-image count (8.67%–59.36% share across four simulated clients under Dirichlet partitioning), motivating discussion of sample-weighted FedAvg aggregation dynamics. \todo{[AUTHOR INPUT REQUIRED: extend this to a client-by-class or client-by-bounding-box breakdown once available — see Task J below — since image count alone does not establish class-distribution skew.]}

(4) A leakage-audited, source-group-aware train/validation/held-out-test protocol that separates the reproducibility analysis (validation split, multi-seed) from a single locked centralized reference point (held-out test split), supporting future matched comparison once B2 checkpoints are evaluated on held-out test.

(5) An architectural adaptation (Batch Normalization $\to$ Group Normalization throughout) applied consistently across all training configurations, reported here as a design choice compatible with small Non-IID per-client batches — **not** presented as an empirically demonstrated performance or stability improvement, since no BN-vs-GN ablation has been run.

*Centralized training (B1) is retained strictly as a single reference point, not as a co-equal contribution or comparison target.*

## 8. Revised Abstract (English, ~210 words)

> Federated learning (FL) enables collaborative model training across distributed data holders without centralizing raw images, which is attractive for oil palm plantation operators facing data-sharing and bandwidth constraints. However, FL deployments in agricultural computer vision are frequently reported from a single training run, leaving open how reproducible their reported performance is under independent stochastic training realizations. This study evaluates the reproducibility and convergence stability of Federated Averaging (FedAvg) for six-class oil palm fresh fruit bunch (FFB) ripeness detection using YOLO11n with Group Normalization, under a fixed Non-IID client partition (Dirichlet $\alpha=0.5$, $K=4$ simulated clients, training-image share ranging 8.67%–59.36% per client). Training was replicated across three independent seeds (42, 123, 2026) over 40 communication rounds each. Validation mAP50 converged to a stable mean of 0.8775 (SD = 0.0157) across seeds, but the best-validation checkpoint occurred at markedly different rounds (9, 11, and 40, respectively), indicating that outcome-level reproducibility does not imply trajectory-level reproducibility. A centralized reference model, evaluated once on a leakage-free held-out test split, achieved mAP50 = 0.8820, which is reported descriptively alongside — not as a matched comparison against — the federated validation results. These findings suggest that multi-seed reporting is necessary to characterize federated training behavior under Non-IID partitions, and that checkpoint-selection policy should account for seed-dependent convergence speed. A matched held-out-test evaluation of the federated checkpoints is identified as the immediate next step.

**Keywords**: Federated Learning; Non-IID Data; Reproducibility; Convergence Stability; Oil Palm Ripeness Detection; YOLO11.

## 9. Revised Introduction

> Precise, non-destructive assessment of fresh fruit bunch (FFB) ripeness is a long-standing operational problem in oil palm cultivation, since harvesting decisions made from ripeness stage directly affect oil extraction rate and downstream yield \cite{suharjito2021,goh2025}. Because ripeness assessment in commercial operations is still predominantly performed through manual visual inspection, harvesting decisions can be inconsistent across harvesters and observation conditions, motivating automated, image-based ripeness assessment \cite{suharjito2021,goh2025}. Deep-learning-based object detectors, particularly the YOLO family, are widely adopted for this task due to their combination of real-time inference speed and competitive detection accuracy \cite{redmon2016,jocher2024}.
>
> In practical deployments, FFB image data are frequently distributed across multiple sources (harvesting sites, collection batches, partner organizations) rather than residing in a single centralized repository, which motivates Federated Learning (FL) as a means of collaborative model training without centralizing raw images \cite{mcmahan2017fedavg,konecny2016}. FL does not, by itself, provide a formal privacy guarantee — model updates and gradients can still leak information about underlying training samples \cite{zhu2019,zhao2020} — but it removes the specific logistical requirement of transmitting raw images to a central server, which is the operational constraint motivating its use here.
>
> A separate and, for deployment purposes, equally important question is *reproducibility*: because FedAvg optimization under Non-IID client data is a stochastic process influenced by client sampling order, local minibatch composition, and initialization, a single federated training run cannot by itself establish whether a reported performance level is a stable, repeatable outcome or an artifact of one favorable stochastic trajectory. This distinction matters operationally: a practitioner deciding whether to deploy a federated-trained model needs to know not only its peak achievable performance, but whether that performance — and the number of communication rounds required to reach it — is consistent across independent training runs.
>
> \subsection*{Research Gap}
> Prior work has separately examined YOLO-based object detection for oil palm FFB ripeness \cite{lai2022,suharjito2021,suharjito2023dataset,asrol2023,goh2025} and Federated Learning for visual classification and detection under Non-IID data \cite{mcmahan2017fedavg,hsu2019,kairouz2021}. However, federated learning studies in agricultural computer vision typically report results from a single training run, without characterizing performance variance or convergence-trajectory variance across independent stochastic seeds \todo{[NEEDS RECENT REFERENCE: a 2022–2025 survey or empirical study explicitly noting the single-run reporting gap in federated learning for agricultural/vision tasks — see search keywords in Related Work below]}. This gap is consequential because single-run reports cannot distinguish a reproducible training regime from a favorable outlier run, which is precisely the distinction this study is designed to test.
>
> \subsection*{Research Question}
> [see Section 6 above for the full RQ set]
>
> \subsection*{Contributions}
> [see Section 7 above]

## 10. Revised Related Work

> \subsection*{Oil Palm FFB Detection and Grading}
> Deep-learning-based detection of oil palm FFB ripeness has predominantly used one-stage object detectors, particularly YOLO-based architectures \cite{lai2022,suharjito2021,suharjito2023dataset,asrol2023,goh2025}. Lai et al. applied YOLOv4 to real-time ripe-bunch detection \cite{lai2022}; Suharjito et al. targeted mobile-device deployment \cite{suharjito2021}; a recent review notes that most reported systems remain single-site, centrally trained models \cite{goh2025}.
>
> \subsection*{Federated Learning for Computer Vision and Object Detection}
> FedAvg \cite{mcmahan2017fedavg} remains the standard aggregation baseline for federated visual learning. Federated object detection under realistic (non-IID, non-uniformly distributed) client data has been studied for general vision tasks \cite{hsu2019,kairouz2021}, but agricultural-domain federated object detection specifically remains comparatively under-reported. \todo{[NEEDS RECENT REFERENCE: a 2023–2025 paper applying FedAvg or a federated variant to any agricultural detection/classification task, ideally with multiple clients — keywords below.]}
>
> \subsection*{Non-IID Heterogeneity and Convergence Stability}
> Non-IID client data is a well-documented source of federated optimization instability \cite{kairouz2021,hsu2019}. Normalization schemes that avoid dependence on cross-sample batch statistics, such as Group Normalization \cite{wu2018groupnorm}, have been proposed as one mitigation for batch-statistic instability under small, heterogeneous per-client batches \cite{li2021fedbn}. \todo{[NEEDS RECENT REFERENCE: a 2022–2025 paper specifically analyzing convergence-round variance (not just final-accuracy variance) under Dirichlet-partitioned Non-IID FL — this is the most important gap to fill, since it is the closest prior work to this paper's core claim. Search keywords: "federated learning convergence round variance", "FedAvg training seed variance Non-IID", "checkpoint selection federated learning heterogeneous clients".]}
>
> \subsection*{Reproducibility and Multi-Seed Evaluation in Machine Learning}
> Multi-seed reporting is an established methodology for characterizing training variance in general deep learning \todo{[NEEDS RECENT REFERENCE: a general ML reproducibility paper, e.g., on the importance of multi-seed reporting for benchmark claims — keywords: "reproducibility crisis deep learning benchmarks", "seed variance deep learning training"]}, but its adoption in federated learning research specifically — as opposed to centralized training — is comparatively rare, particularly for object detection tasks in applied domains such as agriculture. \todo{[NEEDS RECENT REFERENCE: a paper specifically calling for or performing multi-seed federated learning evaluation. Keywords: "multi-seed federated learning evaluation", "federated learning experimental reproducibility".]}
>
> This study addresses this combined gap by reporting three independent training seeds under a fixed Non-IID partition for a federated object-detection task, characterizing both outcome-level (final performance) and trajectory-level (convergence-round) reproducibility — a distinction not, to the authors' knowledge, jointly reported for agricultural federated object detection.

**Catatan referensi (Bahasa Indonesia)**: Referensi berikut di manuskrip lama sudah cukup baik dan dipertahankan: `redmon2016`, `jocher2024`, `lai2022`, `suharjito2021`, `suharjito2023dataset`, `asrol2023`, `goh2025`, `mcmahan2017fedavg`, `wu2018groupnorm`, `hsu2019`, `li2021fedbn`, `kairouz2021`. Referensi berikut sebaiknya **dihapus atau dipindah** karena topiknya (privacy leakage, DP) tidak lagi jadi fokus utama dan porsinya harus dikurangi drastis sesuai instruksi: `zhu2019`, `zhao2020`, `shokri2017`, `dwork2006`, `dworkroth2014`, `abadi2016`, `mcmahan2017dprnn`, `geyer2017`, `gopi2021`, `yousefpour2021`, `de2022` — sisakan maksimal 1–2 sitasi DP singkat di Future Work saja (misal `abadi2016` untuk DP-SGD secara umum), bukan subbagian penuh. `konecny2016` (arXiv 2016) sebaiknya dicek apakah punya versi proceedings/journal; kalau tidak, boleh tetap dipakai tapi ditandai sebagai referensi lama.

## 11. Revised Materials and Methods (struktur)

Struktur baru mengikuti Task I. **Isi teknis (dataset 10,814 gambar, split 8,937/826/1,051, K=4 Dirichlet α=0.5 seed=42, YOLO11n 2,591,010 parameter, 0 BatchNorm/81 GroupNorm modules, RTX 4080, Python 3.10/PyTorch 2.5.1/Ultralytics 8.4.51) semuanya tetap sama persis seperti manuskrip saat ini** — bagian ini sudah solid dan tidak perlu ditulis ulang secara substantif, hanya perlu ditambah dua hal:

1. **Definisi eksplisit** (baru, taruh di awal subbab *Federated Training Protocol*):
   - **Partition seed**: controls the Dirichlet client-partition assignment (fixed at 42 for all runs, so all three training seeds share an identical client data allocation).
   - **Training seed**: controls stochastic elements of the training process itself (weight-initialization-adjacent randomness, data shuffling order, augmentation sampling) — varied across 42/123/2026.
   - **Communication round**: one complete cycle of local training at all four clients followed by FedAvg aggregation (40 total per seed).
   - **Local epoch**: one full pass over a single client's local training partition within one communication round (2 per round).
   - **Best-validation checkpoint**: the model state, saved after the communication round with the highest validation mAP50 for that seed.
   - **Final-round checkpoint**: the model state after the last (40th) communication round, regardless of whether it is also the best-validation checkpoint.

2. **Subbab baru**: *Client Heterogeneity Measurement* — lihat Task J / Section 12 di bawah untuk isi & placeholder.

Perbedaan struktural dari manuskrip lama: subbab *Planned Extension: Differential Privacy* **dipindah ke akhir Discussion sebagai satu paragraf pendek Future Work**, bukan subbab penuh di Methods — sesuai instruksi untuk tidak mendominasi.

## 12. Karakterisasi Non-IID Tambahan (Task J)

Manuskrip saat ini (Tabel `tab:clients`) hanya menunjukkan **quantity imbalance** (jumlah gambar per klien), bukan **class-distribution skew**. Ini penting dilengkapi karena Dirichlet partitioning secara umum digunakan justru untuk mengontrol skew kelas, bukan cuma jumlah gambar — jadi klaim "Non-IID" saat ini under-evidenced tanpa tabel ini.

**Template Tabel (isi dengan [INSERT VERIFIED CLIENT-CLASS COUNT] dari partition manifest kamu, jangan ditebak):**

| Client | Abnormal | Empty Bunch | Overripe | Ripe | Underripe | Unripe | Total |
|---|---|---|---|---|---|---|---|
| 0 | [INSERT VERIFIED CLIENT-CLASS COUNT] | ... | ... | ... | ... | ... | 860 |
| 1 | ... | ... | ... | ... | ... | ... | 775 |
| 2 | ... | ... | ... | ... | ... | ... | 5,305 |
| 3 | ... | ... | ... | ... | ... | ... | 1,997 |

*Catatan: angka Total per baris di atas SUDAH ada di manuskrip (Tabel client image distribution) — cuma perlu dipecah per kelas.*

**Sumber data**: file partisi klien yang dihasilkan `scripts/03_partition_clients.py` (biasanya per-klien `data.yaml` atau manifest JSON yang menyimpan daftar file/label per klien) — hitung jumlah instance bounding box per kelas dari file label YOLO (`.txt`) masing-masing klien.

**Rumus heterogeneity statistic (opsional tapi direkomendasikan)**:
- *Normalized Shannon entropy* per client $c$: $H(c) = -\sum_{k=1}^{6} p_{c,k} \log p_{c,k} / \log 6$, dengan $p_{c,k}$ = proporsi kelas $k$ di klien $c$. $H(c)=1$ berarti distribusi merata; mendekati 0 berarti sangat skewed.
- *Jensen–Shannon divergence* klien $c$ terhadap distribusi global: $\text{JSD}(P_c \| P_{\text{global}})$, dihitung standar (rata-rata KL divergence masing-masing terhadap distribusi campuran $M=(P_c+P_{\text{global}})/2$).
- *Coefficient of variation* jumlah sampel antar klien: $CV = SD(\{860,775,5305,1997\}) / \text{mean}(\{860,775,5305,1997\})$ — ini bisa dihitung sekarang juga dari angka yang sudah ada (tidak perlu data baru): mean = 2,234.25, SD $\approx$ 1,867.9, sehingga $CV \approx 0.836$ — nilai tinggi ini sendiri sudah bisa dilaporkan sebagai bukti kuantitatif quantity imbalance tanpa menunggu breakdown kelas.

**Usulan caption**: "Table X. Per-client, per-class training-instance distribution under the Dirichlet ($\alpha=0.5$) Non-IID partition. Normalized entropy and Jensen–Shannon divergence from the global class distribution are reported per client to quantify class-level (not only quantity-level) heterogeneity."

**Panduan interpretasi**: jika entropy klien besar (misal Client 2, yang punya paling banyak gambar) mendekati 1 sementara klien kecil (Client 0/1) entropy-nya rendah, itu menunjukkan quantity imbalance dan class imbalance **berkorelasi** di partisi ini — relevan untuk Discussion poin 6 (Task L) yang minta keduanya dianalisis terpisah.

## 13. Revised Results (struktur + isi yang sudah tersedia)

> \subsection*{Centralized Reference Performance}
> [Tetap seperti manuskrip lama — Tabel B1 held-out test, mAP50=0.8820, mAP50-95=0.7309, Precision=0.8745, Recall=0.8653, Ripe terlemah di 0.5571. Tidak berubah.]
>
> \subsection*{Multi-Seed Federated Validation Performance}
> [Tetap seperti manuskrip lama — Tabel B2 tiga seed, mean mAP50=0.8775 (SD=0.0157), mean mAP50-95=0.7374 (SD=0.0206).]
>
> \subsection*{Convergence Trajectory Across Seeds}
> Figure~\ref{fig:b2-convergence} (sudah ada, `figure_b2_convergence.png`) is now the **primary results figure**, not a supporting one — reposisikan secara naratif sebagai bukti utama reproducibility, bukan sebagai pendukung perbandingan B1 vs B2.
>
> \subsection*{Best-Round versus Final-Round Behavior}
> **[RESULT NOT YET AVAILABLE — perlu dihitung]**: selisih mAP50 antara best-validation checkpoint dan final-round (round 40) checkpoint, per seed. Untuk seed 2026 selisihnya nol (best round = final round = 40). Untuk seed 42 dan 123, ini **belum dilaporkan** di manuskrip saat ini — perlu angka mAP50 di round 40 untuk seed 42 dan 123 (bukan cuma best-round-nya) untuk mengisi tabel berikut:

| Seed | Best round | Best mAP50 | Final (round 40) mAP50 | Δ (Best − Final) |
|---|---|---|---|---|
| 42 | 9 | 0.8850 | [RESULT NOT YET AVAILABLE] | [RESULT NOT YET AVAILABLE] |
| 123 | 11 | 0.8594 | [RESULT NOT YET AVAILABLE] | [RESULT NOT YET AVAILABLE] |
| 2026 | 40 | 0.8880 | 0.8880 | 0.0000 |

> \subsection*{Inter-Seed Variability}
> [Tetap — mAP50 mean=0.8775, SD=0.0157; mAP50-95 mean=0.7374, SD=0.0206; ini sudah cukup kuat sebagai bukti outcome-level reproducibility.]
>
> \subsection*{Per-Class Federated Stability}
> **[RESULT NOT YET AVAILABLE]** — Template tabel:

| Class | Seed 42 AP50 | Seed 123 AP50 | Seed 2026 AP50 | Mean | SD |
|---|---|---|---|---|---|
| Abnormal | [RESULT NOT YET AVAILABLE] | ... | ... | ... | ... |
| Empty Bunch | ... | ... | ... | ... | ... |
| Overripe | ... | ... | ... | ... | ... |
| Ripe | ... | ... | ... | ... | ... |
| Underripe | ... | ... | ... | ... | ... |
| Unripe | ... | ... | ... | ... | ... |

> \subsection*{Matched Held-Out-Test Comparison (once available)}
> **[RESULT NOT YET AVAILABLE]** — Template tabel:

| Model | Split | mAP50 | mAP50-95 |
|---|---|---|---|
| B1 (centralized) | Held-out test | 0.8820 | 0.7309 |
| B2 seed 42 | Held-out test | [RESULT NOT YET AVAILABLE] | [RESULT NOT YET AVAILABLE] |
| B2 seed 123 | Held-out test | [RESULT NOT YET AVAILABLE] | [RESULT NOT YET AVAILABLE] |
| B2 seed 2026 | Held-out test | [RESULT NOT YET AVAILABLE] | [RESULT NOT YET AVAILABLE] |

**Caption konvergensi (revisi)**: "Figure X. Validation mAP50 per communication round across three independent FedAvg training seeds (42, 123, 2026) under a fixed Non-IID partition (Dirichlet $\alpha=0.5$, K=4). Filled markers denote each seed's best-validation round, illustrating substantial trajectory-level variance (round 9–40) despite convergent final-outcome performance (SD = 0.0157)."

**Statistical summary yang disarankan**: laporkan mean $\pm$ SD untuk mAP50 dan mAP50-95 (sudah ada), tambahkan range best-round (9–40) dan, kalau item Task O #4 selesai, coefficient of variation untuk best-round itu sendiri sebagai ukuran trajectory-variance kuantitatif.

## 14. Revised Discussion

> The multi-seed results reported here separate two distinct notions of federated-training reliability that are often conflated in single-run reports: **outcome-level reproducibility** (how similar is the final achievable performance across independent runs) and **trajectory-level reproducibility** (how similar is the path — specifically, the number of communication rounds — taken to reach that performance).
>
> On outcome-level reproducibility, the three seeds converge to a comparable peak validation mAP50 (0.8594–0.8880, SD = 0.0157), which is consistent with — though not proof of — FedAvg being a numerically stable aggregation procedure under this specific Non-IID partition (Dirichlet $\alpha=0.5$, K=4). On trajectory-level reproducibility, the same three seeds reach their best checkpoint at markedly different rounds (9, 11, and 40), an approximately four-fold spread. \emph{This is an interpretation, not a proven causal claim:} a single-seed federated experiment reporting only "round 9, mAP50=0.8850" would have substantially understated the number of rounds this configuration can require to converge, illustrating why single-seed federated reports risk incomplete conclusions about training-time budgets.
>
> Client 2 contributes 59.36% of the training subset (Table~\ref{tab:clients}), substantially more than the remaining three clients combined. Because FedAvg aggregates client updates weighted by local sample count \cite{mcmahan2017fedavg}, Client 2's locally well-supported gradient signal plausibly exerts disproportionate influence on the aggregated global update each round; \emph{this is offered as a plausible contributing hypothesis for the observed convergence stability, not as a demonstrated causal mechanism} — confirming it would require a controlled ablation (e.g., uniform-weighted aggregation) that is outside the scope of the present study.
>
> This quantity imbalance (Table~\ref{tab:clients}) is distinct from, and should not be conflated with, class-distribution imbalance across clients; the present manuscript currently characterizes only the former. \todo{[AUTHOR INPUT REQUIRED: once the client-by-class breakdown (Section 12 / Task J above) is available, discuss whether class skew tracks quantity skew or diverges from it.]}
>
> Per-class stability across seeds has not yet been evaluated in this manuscript \todo{[RESULT NOT YET AVAILABLE — see Section 13]}; global mAP50 stability does not guarantee that every class is individually stable, and a class with high inter-seed variance could be masked by a low-variance aggregate metric if its AP50 magnitude is small relative to others.
>
> The centralized reference (B1, held-out test mAP50 = 0.8820) and the federated multi-seed mean (B2, validation mAP50 = 0.8775) are numerically close, but this manuscript deliberately does not characterize this as a matched centralized-versus-federated comparison, since the two are evaluated on different splits. \emph{This closeness is reported descriptively, not as evidence that federation carries a specific, quantified utility cost}; establishing that requires evaluating the locked B2 checkpoints on the same held-out test split used for B1 (Task O checklist item 1).
>
> It is also important to note that this study varies only *statistical* client heterogeneity (via Dirichlet partitioning) under a single-host sequential simulation; it does not exercise distributed-systems behavior such as client dropout, communication failure, or network latency, which are separate and equally relevant dimensions of real-world federated deployment reliability.
>
> \subsection*{Future Work: Differential Privacy}
> A planned extension of this pipeline will incorporate sample-level DP-SGD \cite{abadi2016} to quantify the additional utility cost of formal per-sample privacy guarantees, building on the reproducibility baseline established here. Results for this extension are outside the scope of the present study.

## 15. Revised Limitations

- Federated training was simulated sequentially on a single GPU (single-host simulation); no distributed-systems factors (network latency, client dropout, communication failure, bandwidth constraints) were exercised.
- The dataset originates from a single public source (Roboflow); generalization to other plantations, cultivars, cameras, or lighting conditions is not directly evaluated.
- Four clients are simulated sample-level data shards drawn from one training subset, not four physically distinct plantations, estates, organizations, or devices; conclusions are limited to the statistical heterogeneity captured by the Dirichlet partition, not real distributed-deployment heterogeneity.
- $K$ is fixed at 4 and $\alpha$ is fixed at 0.5; sensitivity of the reported reproducibility findings to other client counts or Non-IID severities has not been tested.
- Only three training seeds were used; this establishes a first-order variance estimate but a larger number of replications would tighten it.
- Client heterogeneity is currently characterized only by training-image count, not by class distribution or bounding-box count (Section 12 / Task J).
- B2 has not yet been evaluated on the held-out test split; consequently, no matched centralized-versus-federated comparison is currently possible, and the descriptive B1/B2 closeness noted in the Discussion should not be read as such a comparison.
- No BatchNorm-versus-GroupNorm ablation has been run; Group Normalization's effect on the observed stability is not empirically isolated from other factors.
- This study does not provide a formal (Differentially Private) privacy guarantee; FL here only avoids transmitting raw images, which is a narrower property than formal privacy.

## 16. Usulan Tabel dan Gambar Baru

| # | Isi | Sumber data | Status |
|---|---|---|---|
| T-new-1 | Best-vs-final-round mAP50 per seed | Re-evaluate seed 42 & 123 checkpoints at round 40 | Perlu 2 angka baru |
| T-new-2 | Per-class AP50 per seed (3 seed × 6 kelas) | Re-run evaluation dengan `plots`/`per-class` output per seed | Perlu evaluasi ulang |
| T-new-3 | Client-by-class instance count | Partition manifest `scripts/03_partition_clients.py` | Sudah ada lokal, tinggal dihitung |
| T-new-4 | Matched held-out-test B1 vs B2 (3 seed) | Evaluasi checkpoint B2 terkunci di held-out test | Perlu evaluasi ulang (checklist #1) |
| G-new-1 | Client-class heatmap (normalized) | Dari T-new-3 | Turunan T-new-3 |
| G-new-2 | Stacked bar proporsi kelas per klien | Dari T-new-3 | Turunan T-new-3 |

## 17. Prioritized Experiment Checklist

### WAJIB sebelum submission

1. **Evaluasi B2 (3 seed) di held-out test split yang sama dengan B1.**
   - *Kenapa*: ini syarat untuk klaim "reproducibility" yang benar-benar matched, dan untuk mengisi T-new-4.
   - *Dipakai di*: Results (Matched comparison), Discussion, Conclusion.
   - *Data dibutuhkan*: checkpoint terkunci ketiga seed (`best.pt` per seed).
   - *Retraining atau evaluation saja*: **evaluation saja** — bisa pakai script serupa `scripts/42_generate_b1_test_plots.py` tapi diarahkan ke checkpoint B2 dan `--split test` (perlu cek/adaptasi scriptnya untuk terima path checkpoint B2 per seed).

2. **Per-class AP50 untuk B2, ketiga seed** (T-new-2).
   - *Kenapa*: memvalidasi/mengganti klaim "global mAP stability" agar tidak menutupi ketidakstabilan per-kelas.
   - *Dipakai di*: Results, Discussion.
   - *Data*: hasil evaluasi Ultralytics per-class (biasanya otomatis muncul di output evaluasi, cukup dicatat).
   - *Evaluation saja.*

3. **Confusion matrix dan PR curve B1** (sudah dibahas di percakapan sebelumnya — jalankan `scripts/42_generate_b1_test_plots.py`).
   - Ini Gambar 4 yang sudah lama ditunggu di manuskrip lama, tetap relevan di versi reframe sebagai bagian dari centralized reference reporting.

4. **Client-by-class distribution** (T-new-3, Section 12).
   - *Kenapa*: bukti kuantitatif Non-IID class skew, bukan cuma quantity imbalance.
   - *Dipakai di*: Methods (Client Heterogeneity Measurement), Discussion.
   - *Data*: partition manifest yang sudah ada secara lokal — cukup dihitung ulang (skrip Python sederhana), **tidak perlu retraining**.

5. **Best-vs-final-round difference untuk seed 42 dan 123** (T-new-1).
   - *Kenapa*: melengkapi klaim trajectory-variance di RQ Supporting #1.
   - *Data*: mAP50 checkpoint round 40 untuk seed 42 dan 123 (checkpoint kemungkinan sudah tersimpan dari training, tinggal dievaluasi) — **evaluation saja**, bukan retraining, ASALKAN checkpoint round-40 masih tersimpan di disk.

6. **Hapus seluruh `\todo{}` yang tersisa** setelah semua di atas terisi (author list/email sudah selesai; tanggal submisi & Vol/No/DOI memang diisi editor, bukan ditodo-kan penulis).

7. **Verifikasi seluruh angka terhadap raw JSON/output evaluasi** — cocokkan tabel manuskrip vs `results/*.json` / `history.json` sebelum submit.

### SANGAT DIREKOMENDASIKAN (tidak wajib, tapi memperkuat paper)

8. Evaluasi satu setting Non-IID tambahan (misal $\alpha=0.1$ atau $\alpha=1.0$) untuk menunjukkan sensitivitas reproducibility terhadap tingkat heterogenitas — **butuh retraining** (3 seed × setting baru).
9. Analisis per-round variance (bukan cuma best-round) — bisa dihitung dari `history.json` yang sudah ada, **evaluation/analysis saja**, tidak perlu retraining.
10. Tabel perbandingan related-work (ringkas metode/dataset/#clients dari studi FL agrikultur lain) — riset literatur, bukan eksperimen.

### OPSIONAL

11. Tambah seed dari 3 ke 5 — **butuh retraining** (2 seed baru).
12. BN-vs-GN ablation — **butuh retraining** (minimal 1 run tambahan tanpa GroupNorm).
13. Sensitivity analysis jumlah klien $K$ (misal $K=2,8$) — **butuh retraining** (multiple runs).

## 18. Daftar Author Input yang Masih Dibutuhkan

- [AUTHOR INPUT REQUIRED] Confirm whether round-40 checkpoints for seeds 42 and 123 are still saved on disk (needed for Checklist #5).
- [AUTHOR INPUT REQUIRED] Client-by-class partition manifest location (needed for Checklist #4 / Section 12).
- [AUTHOR INPUT REQUIRED] Confirm whether B2's three locked checkpoints (`best.pt` per seed) are still available for held-out-test re-evaluation (Checklist #1).
- [AUTHOR INPUT REQUIRED] Decision: keep the current DP "Planned Extension" as a short Future Work paragraph only (recommended), or remove it entirely from this manuscript and reserve it fully for the separate DP paper?
- [AUTHOR INPUT REQUIRED] Confirm acknowledgement wording (already flagged as a `\todo` in the current manuscript — unrelated to this reframe).
- [AUTHOR INPUT REQUIRED / editor] Vol/No/Month/Year/Page/DOI/Received/Revised/Accepted/Published — journal-assigned, no action needed from you before submission.

## 19. Daftar Unsupported Claim yang Dihapus/Diperlemah

| Klaim lama | Masalah | Perubahan |
|---|---|---|
| "the small indicative gap... suggests that FedAvg-based federated training can approach centralized detection performance" (Abstract lama) | Membandingkan test-split (B1) dengan validation-split (B2) seolah matched | Diganti: dilaporkan deskriptif, eksplisit disebut belum matched, novelty dipindah ke reproducibility |
| Judul "Non-IID FL **versus** Centralized Training" | Framing utama = comparison yang belum valid secara metodologis | Diganti ke reproducibility framing (lihat judul terpilih) |
| GroupNorm dikaitkan implisit dengan stabilitas B2 di Discussion lama ("This stability is consistent with the expectation that Group Normalization reduces...") | Belum ada ablation BN-vs-GN | Dilabel eksplisit sebagai "consistent with expectation", bukan bukti; ditambah catatan di Limitations |
| Subbab penuh "Planned Extension: Differential Privacy" + paragraf Discussion khusus DP | DP mendominasi porsi manuskrip padahal belum dievaluasi sama sekali di paper ini | Dipangkas jadi satu paragraf Future Work singkat di akhir Discussion |
| Bagian besar Introduction & Related Work soal privacy leakage/membership inference (`zhu2019`, `zhao2020`, `shokri2017` dst.) | Tidak relevan dengan kontribusi reproducibility; menciptakan kesan paper tentang privasi | Dihapus dari badan teks utama, referensinya dipangkas (lihat Section 10) |
| Frasa yang menyiratkan federated = privasi terjamin | FL tanpa DP formal, hanya menghindari transmisi raw image, bukan formal privacy guarantee | Ditegaskan eksplisit di Introduction paragraf baru |

## 20. Final Submission-Readiness Verdict

**Belum siap submit apa adanya** — bukan karena kualitas tulisan, tapi karena framing lama menjanjikan "centralized vs federated comparison" yang datanya sendiri belum matched, dan porsi DP terlalu dominan untuk paper yang belum melaporkan hasil DP sama sekali.

**Jalur tercepat ke submission-ready** (perkiraan, tanpa retraining baru):
1. Terapkan reframe (judul, RQ, contributions, abstract, intro, related work, discussion, conclusion) — bisa langsung dari draft di dokumen ini.
2. Selesaikan Checklist item #4 (client-by-class — data sudah ada lokal, tinggal dihitung) dan #6 (evaluation-only, bergantung checkpoint masih tersimpan).
3. Item #1–#2 (held-out test matched comparison, per-class B2) **sangat direkomendasikan** sebelum submit tapi secara teknis manuskrip reframe ini **tetap valid secara ilmiah tanpa itu**, asalkan bagian yang bergantung padanya tetap eksplisit ditandai "future work" / placeholder — karena klaim utama paper (outcome-vs-trajectory reproducibility) sudah sepenuhnya didukung oleh data yang ADA SEKARANG (Tabel B2 tiga seed + convergence curve).

---

*Dokumen ini adalah draft usulan, belum diterapkan ke `manuscript.tex`. Perubahan judul/abstract/contributions ini adalah keputusan besar yang mengubah identitas paper — beri tahu aku kalau kamu setuju dengan arah reframe ini, nanti langsung aku terapkan ke `manuscript.tex` dan build ulang PDF-nya.*
