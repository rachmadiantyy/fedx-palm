# Paper C — IIUM Engineering Journal (IIUMEJ)

**Paper payung untuk SELURUH tesis** "FedX-Palm: A Federated Explainable
Framework for Privacy-Preserving Palm Fruit Ripeness". Beda fokus dengan
JIKI (paper companion XAI):

| Paper | Venue | Fokus |
|---|---|---|
| JIKI (`../paper_b_jiki_fl_xai/`) | JIKI UI, SINTA 2 | **XAI / Grad-CAM++ faithfulness** (FL tanpa DP) |
| **IIUM (ini)** | IIUM Engineering Journal | **Framework lengkap** — HFL + **DP-SGD privacy-utility** (inti) + YOLOv11 + XAI |

> Catatan: versi JUTI (`../paper_b_juti_fl_xai/`) sebelumnya ditolak karena
> dianggap "terlalu fokus ML/FL". Paper XAI sekarang diarahkan ke **JIKI UI**
> yang scope-nya eksplisit mencakup machine learning, federated learning, dan
> computer vision. Folder JUTI dipertahankan sebagai arsip saja.

Co-author Prof. Teddy Surya Gunawan adalah dosen IIUM (tsgunawan@iium.edu.my),
jadi venue ini natural fit untuk paper utama tesis.

## Centerpiece: analisis privacy-preserving DP-SGD

Inti paper IIUM = bagian yang menjawab semua revisi mayor pembimbing (R1–R4):

| Revisi pembimbing | Dijawab di paper |
|---|---|
| R1: ε dari privacy accountant | Tabel ε (PRV accountant, δ=1e-5), Sec. 4.2 |
| R2: ablation collapse | Grid 25 sel E1 + 25 sel E2, Sec. 4.2–4.3 |
| R3: reframe klaim DP | "configuration-specific, gradual not catastrophic" |
| R4: anti-leakage split | bunch_id split + audit, Sec. 3.1 |

Plus temuan kunci: **degradasi gradual (bukan cliff)**, **K besar memperburuk
per-sample DP (H2-K)**, dan **partial DP (E2) KALAH dari full DP (E1)** —
menolak hipotesis Tramèr & Boneh untuk object detection domain baru.

## Cara compile

```bash
pdflatex main
pdflatex main      # 2x untuk cross-ref (Eq/Table/Fig/Section)
```
Tanpa bibtex — referensi inline (`thebibliography`, 20 entri).

Overleaf: New Project → Upload → folder ini → Compiler **pdfLaTeX**.

## Gambar yang dibutuhkan (folder `figures/`)

| # | File | Sumber |
|---|---|---|
| Fig. 1 | `privacy_utility_e1.png` | copy dari `thesis_rebuild/figures/privacy_utility_e1.png` |
| Fig. 2 | `xai_per_class.png` | copy dari `pic/xai_comparison.png` (rename) |
| Fig. 3 | `docker_ui.png` | screenshot halaman utama layanan VPS (port 8080) |
| Fig. 4 | `result_heatmap.png` | screenshot hasil deteksi + heatmap pada citra uji |

Tanpa keempat file ini compile gagal di `\includegraphics`. Fig. 3-4 untuk
section *Deployment Demonstration*.

## Struktur paper

1. Introduction — 4 pilar + kontribusi
2. Literature Review — vision palm, FL cross-silo, **DP for deep learning**, XAI
3. Methodology — anti-leakage split, Non-IID, YOLOv11n-GN, FedAvg,
   **per-sample DP-SGD + PRV accountant**, full vs partial DP, XAI metrics
4. Results — FL cost · **E1 grid + ε + privacy-utility** · **E2 vs E1
   (Tramèr-Boneh ditolak)** · tiga rezim · XAI · threats
5. Conclusion

## Beda format vs JUTI

- 2 kolom A4 (JUTI 1 kolom)
- **ABSTRACT + ABSTRAK (Melayu)** — wajib IIUMEJ
- Heading bernomor `1.`/`2.1.`/`2.1.1.`
- Tabel Arabic (Table 1, 2, …), caption period-style
- Referensi IIUMEJ author-year (surname + inisial, tahun dalam kurung)

## Sebelum submit — checklist

- [ ] Copy 2 gambar ke `figures/` (`privacy_utility_e1.png`, `xai_per_class.png`).
- [ ] **Review ABSTRAK Melayu** oleh Prof. Teddy (draft sudah ada).
- [ ] Isi tanggal Received/Accepted/Published + Vol/No di running head.
- [ ] **Konversi LaTeX → DOCX** (template IIUMEJ resmi, via pandoc/Word).
- [ ] **Ethical Agreement Form** (lampiran IIUMEJ) ditandatangani semua author.
- [ ] **Dua proposed reviewers** independen (di luar Telkom & IIUM).
- [ ] **Cover letter dengan deklarasi COI** — lihat di bawah.
- [ ] (Opsional, untuk Q-class rigour) tambah 1 seed untuk B2 K=4 25-ronde dan
      E1 best (K=4, σ=0.5), laporkan mean ± SD.

## Conflict-of-interest (COI) — penting

Co-author **Prof. Teddy Surya Gunawan** terdaftar sebagai *Principal Contact*
IIUMEJ. Ini bukan blocker (banyak editor mempublikasikan di jurnal sendiri),
tetapi **wajib dideklarasikan eksplisit** di cover letter dan handling editor
harus dialihkan ke editor lain agar tidak ada konflik penanganan.

### Draft kalimat cover letter

> *We wish to disclose that one of the authors, Prof. Teddy Surya Gunawan, is
> listed as Principal Contact of IIUM Engineering Journal. We respectfully
> request that the editorial handling of this submission be assigned to an
> independent editor with no co-authorship, supervisory, or collaborative
> relationship with the authors, and that the peer-review process proceed
> single-blind to the authors without involvement of Prof. Gunawan in any
> editorial decision concerning this manuscript.*

## Angka kunci (sumber kebenaran — sinkron tesis Bab 4)

- B1 centralized: 0.787 / 0.672 / 0.815 / 0.833
- B2 federated K=4: 0.738 / 0.584 / 0.626 / 0.727; FL cost 0.049
- E1 best: 0.190 @ K=4,σ=0.5, ε=8.36 · E2 best: 0.113 @ K=2,σ=0.5
- Tiga rezim: no-DP (0.74–0.79) / weak-DP (~0.19, ε≈8) / strict-DP (collapse, ε<0.5)
- XAI federated: AD 15.90%, FRR 0.281 · centralized: AD 4.95%, FRR 0.182
