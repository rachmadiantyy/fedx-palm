# FedX-Palm — IEEE Conference Paper

Draft makalah konferensi IEEE (English) dengan angle **sistem FedX-Palm utuh**:
HFL + Differential Privacy + XAI untuk deteksi kematangan TBS kelapa sawit.

## File
- `fedx_palm_ieee.tex` — manuskrip utama (kelas `IEEEtran`, mode `conference`).
- `references.bib` — daftar pustaka (22 sitasi terpakai, sudah diverifikasi).

## Gambar yang harus disalin ke folder ini sebelum compile
Paper mereferensikan `privacy_utility_tradeoff.png` (Fig. 1, dua panel:
konvergensi + cliff trade-off). **Salin file PNG dari
`saved_runs/privacy_utility_tradeoff.png` ke folder `paper/`** sebelum compile,
atau Overleaf akan error "file not found". Tanpa file itu, comment dulu blok
`\begin{figure*}...\end{figure*}` agar tetap bisa compile.

## ‼️ BUG yang ditemukan: ε di `privacy_budget_derivation.csv` salah faktor 10^6
`privacy_budget_derivation.csv` melaporkan eps_rdp ~10^14 (σ=1e-4), TAPI
`summary_all_scenarios.csv`, anotasi pada `privacy_utility_tradeoff.png`, dan
verifikasi manual rumus RDP semuanya memberi ~10^8. Paper memakai angka yang
BENAR (10^8). **Jangan pakai kolom eps di `privacy_budget_derivation.csv`**
tanpa memperbaiki skrip derivasinya (ada faktor 10^6 yang keliru, kemungkinan
σ ter-skala salah). Conclusion tidak terpengaruh: semua tetap ε ≫ 10³.

## Cara compile (paling mudah: Overleaf)
1. Buka https://overleaf.com → New Project → Upload Project (zip folder `paper/`),
   atau New Project → Blank, lalu upload kedua file.
2. Pastikan menu **Menu → Compiler = pdfLaTeX**.
3. Compile. Overleaf menjalankan pdflatex → bibtex → pdflatex×2 otomatis.

## Compile lokal (kalau ada TeX Live / MiKTeX)
```bash
pdflatex fedx_palm_ieee
bibtex   fedx_palm_ieee
pdflatex fedx_palm_ieee
pdflatex fedx_palm_ieee
```

## Yang HARUS dilengkapi / diverifikasi sebelum submit
- [ ] **Penulis & afiliasi** (placeholder `[Author Name]`, `[Institution]`, dst).
- [ ] **VERIFIKASI angka per-class** yang sekarang dikutip kualitatif di Sec. IV-A
      ("AP@0.5 uniformly 0.99--1.00, adjacent-class confusion negligible").
      Angka ini berasal dari centralized benchmark (thesis Tabel 4.9/4.7) yang
      mungkin di-run pada split berbeda. **Regenerasi confusion matrix + tabel
      per-class pada split bersih (palm_v2_resplit) yang dipakai paper**, lalu
      idealnya tambahkan sebagai tabel/gambar agar makin meyakinkan reviewer.
- [ ] **Konfirmasi jumlah citra** dari folder data autoritatif: client train
      1145/2479/2067/3339 (=9030) dan val global 935/3140. Catatan: dokumen lama
      `hasil_simulasi_real.md` menyebut 937/3273 dan total klien berbeda karena
      berasal dari run Colab terdahulu; paper memakai angka run workstation
      (ablation log). Pastikan satu sumber konsisten sebelum submit.
- [ ] **Gambar** (disarankan): arsitektur sistem, kurva konvergensi, grafik
      cliff privacy-utility (`saved_runs/privacy_utility_tradeoff.png`), heatmap
      Grad-CAM++. Tambah via `\begin{figure}...\includegraphics`.
- [ ] **Verifikasi 4 entri palm-oil** (`septiarini`, `suharjito`, `mansour`,
      `saleh`) di Google Scholar.
- [ ] **Pemilihan venue:** paper bertipe *characterized negative result +
      mitigation*. Targetkan venue/workshop yang ramah reproducibility/negative
      results (mis. workshop FL/privacy) agar fit lebih baik.
- [ ] Sesuaikan template ke venue tujuan (kalau bukan IEEE Conference standar).

## Revisi v3 (jawab review lanjutan)
- Aritmetika dataset diperbaiki: split 80/10/10 dinyatakan *per-bunch*; rasio
  citra menyimpang karena tiap tandan punya jumlah frame berbeda; val 935/3140
  ditegaskan sebagai scoring set, test set disisihkan & tidak dipakai di paper.
- Ditambah kalimat per-class di Sec. IV-A (membela baseline 0.995: kelas
  terpisah genuine, error dominan = lokalisasi/background, bukan misklasifikasi
  kematangan).
- Ditambah justifikasi nilai diagnostik ambang $C\times\sigma$ di Sec. IV-C
  (kenapa $C{=}0.1$ tetap berguna untuk pencarian konfigurasi privasi-bermakna
  walau belum privat).

## Yang sudah dibersihkan (revisi v2)
- Title & Contribution #1 di-rebrand: tidak lagi mengklaim "Dockerized FL
  training"; eksplisit "single-GPU FedAvg simulation + Docker deployment
  blueprint + real inference container".
- "Privacy-Preserving" dihapus dari title — paper sekarang dipresentasikan
  sebagai *characterized failure mode + mitigation*, sesuai dengan
  $\varepsilon \gg 10^{3}$ pada seluruh setelan.
- Dataset subsection diperinci: total 9{,}030 training images, per-client
  Dirichlet $\alpha$, global val 935/3{,}140; sumber Roboflow disebut.
- Threats to Validity dinaikkan jadi Limitations eksplisit: execution model,
  optimistic baseline, limited federated dynamics, no meaningful privacy.
- `references.bib` dirapikan: catatan editorial Indonesia (`VERIFIKASI`,
  `TIPS PENCARIAN`) dihapus; tipe entri ICSITech diperbaiki ke
  `@inproceedings`.

## Catatan konsistensi angka (sumber kebenaran)
- Baseline global: mAP@0.5 = 0.995, mAP@0.5:0.95 = 0.951, P = 0.998, R = 0.999.
- DP sweep (C=10): semua mAP = 0; ε (RDP) ≫ 10³.
- Ablation: collapse diatur magnitudo C×σ; A5 (C=0.1) selamat (0.995);
  GroupNorm+noise (C1) tetap collapse → BatchNorm bukan akar masalah.
- XAI baseline: Average Drop 95.1%, FRR 0.962.
- Detail penuh: `thesis/hasil_simulasi_real.md`.
