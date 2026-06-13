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
- [ ] **Per-client training counts** (footnote di Sec. III-B) — ambil dari
      `saved_runs/summary_all_scenarios.csv` / split akhir. Saat ini hanya val
      global yang dicantumkan (935 citra, 3140 instance).
- [ ] **Gambar** (opsional tapi disarankan): arsitektur sistem, kurva konvergensi,
      grafik cliff privacy-utility (`privacy_utility_tradeoff.png`), contoh
      heatmap Grad-CAM++. Tambah dengan `\begin{figure}...\includegraphics`.
- [ ] **Verifikasi entri palm oil** di `references.bib` (ada tanda `VERIFIKASI`)
      sebelum dipakai/diperluas.
- [ ] Sesuaikan judul venue/konferensi bila template panitia berbeda.

## Catatan konsistensi angka (sumber kebenaran)
- Baseline global: mAP@0.5 = 0.995, mAP@0.5:0.95 = 0.951, P = 0.998, R = 0.999.
- DP sweep (C=10): semua mAP = 0; ε (RDP) ≫ 10³.
- Ablation: collapse diatur magnitudo C×σ; A5 (C=0.1) selamat (0.995);
  GroupNorm+noise (C1) tetap collapse → BatchNorm bukan akar masalah.
- XAI baseline: Average Drop 95.1%, FRR 0.962.
- Detail penuh: `thesis/hasil_simulasi_real.md`.
