# Manuskrip JIKI -- Explainable Federated Detection (Grad-CAM++ Faithfulness Study)

- `manuscript.tex` -- transkripsi LaTeX lengkap dari paper `original.pdf` yang sudah jadi (Introduction, Literature Review, Methodology, Results and Discussion, Conclusion, CRediT, Declarations, Data Availability, 18 referensi). Dua kolom (`extarticle`, 9pt) mengikuti format asli JIKI, dengan header berjalan meniru `original.pdf`.
- `original.pdf` -- PDF paper asli yang diunggah, dipertahankan sebagai referensi/sumber transkripsi.
- `manuscript.pdf` -- hasil build dari `manuscript.tex` (6 halaman, sudah diverifikasi visual per halaman).
- `figure1_methodology.png` -- Gambar 1 (diagram alur metodologi penelitian), diekstrak lossless dari `original.pdf` via `pdfimages -png`.
- `figure2_gradcam_heatmaps.png` -- Gambar 2 (grid heatmap Grad-CAM++), diekstrak lossless dari `original.pdf` via `pdfimages -png`.
- `fedxpalm_jiki_overleaf.zip` -- paket siap unggah ke Overleaf (isinya `manuscript.tex` + kedua gambar).

## Status gambar

Berbeda dengan manuskrip JUTIF, paper ini **sudah lengkap** -- kedua gambar (metodologi & Grad-CAM++ heatmaps) sudah final dan sudah ditempel di `manuscript.tex` sebagai `figure*` (spanning dua kolom), tidak ada `\todo{}` untuk gambar yang tersisa.

## Cara pakai di Overleaf (tidak perlu install apa-apa)

1. Buka [overleaf.com](https://www.overleaf.com) -> **New Project** -> **Upload Project**.
2. Pilih file `fedxpalm_jiki_overleaf.zip`.
3. Overleaf otomatis compile begitu project terbuka (kalau tidak, klik tombol **Recompile** di atas panel PDF).

## Cara build lokal (opsional, kalau tidak pakai Overleaf)

Butuh TeX Live (`pdflatex`, `latexmk`):

```bash
latexmk -pdf manuscript.tex
```

Untuk bersihkan file bantu (`.aux`/`.log`/`.out`/dst.) tanpa menghapus PDF:

```bash
latexmk -c
```
