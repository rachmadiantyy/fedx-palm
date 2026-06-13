# FedX-Palm — IEEE Conference Paper

Draft makalah konferensi IEEE (English) dengan angle **sistem FedX-Palm utuh**:
HFL + Differential Privacy + XAI untuk deteksi kematangan TBS kelapa sawit.

## File
- `fedx_palm_ieee.tex` — manuskrip utama (kelas `IEEEtran`, mode `conference`).
- `references.bib` — daftar pustaka (22 sitasi terpakai, sudah diverifikasi).

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

## Yang HARUS dilengkapi sebelum submit
- [ ] **Penulis & afiliasi** (placeholder `[Author Name]`, `[Institution]`, dst).
- [ ] **Gambar** (opsional tapi disarankan): arsitektur sistem, kurva konvergensi,
      grafik cliff privacy-utility (`privacy_utility_tradeoff.png`), contoh
      heatmap Grad-CAM++. Tambah dengan `\begin{figure}...\includegraphics`.
- [ ] **Verifikasi 4 entri palm-oil** (`septiarini`, `suharjito`, `mansour`,
      `saleh`) di Google Scholar — pastikan year/venue/halaman cocok dengan
      publikasi asli sebelum submit.
- [ ] Sesuaikan template ke venue tujuan (kalau bukan IEEE Conference standar).

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
