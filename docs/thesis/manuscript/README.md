# Manuskrip JUTIF -- FedXPalm B1/B2

- `manuscript.tex` -- sumber LaTeX (isi sama persis dengan `JUTIF_Manuscript_FedXPalm_B1B2.docx`, dipertahankan sebagai referensi/backup).
- `manuscript.pdf` -- hasil build dari `manuscript.tex`.
- `figure_b2_convergence.png` + `b2_k4_seed{42,123,2026}_history.json` -- Gambar 6 (kurva konvergensi B2) dan data sumbernya, lihat `scripts/43_plot_b2_convergence.py`.

## Cara build

Butuh TeX Live (`pdflatex`, `latexmk`). Kalau belum ada:

```bash
sudo apt-get install texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended latexmk
```

Lalu build dari folder ini:

```bash
latexmk -pdf manuscript.tex
```

`latexmk` otomatis menjalankan `pdflatex` berulang sampai referensi silang (`\ref`, daftar pustaka) stabil -- tidak perlu jalankan `pdflatex` manual berkali-kali. Untuk bersihkan file bantu (`.aux`/`.log`/`.out`/dst.) tanpa menghapus PDF:

```bash
latexmk -c
```

## Status

Item `\todo{...}` (tampil merah di PDF) masih menunggu diisi -- **jangan diisi dengan tebakan**:

- Nama/urutan co-author, email corresponding author, tanggal submisi.
- Gambar 1 (diagram alur riset), Gambar 2 (contoh 6 kelas), Gambar 4 (confusion matrix/PR curve B1 di *held-out test* -- jalankan `scripts/42_generate_b1_test_plots.py`), Gambar 5 (contoh deteksi kualitatif).
- Gambar 3 (distribusi klien) sifatnya opsional -- angkanya sudah ada di Tabel 2.
- Pernyataan ketersediaan kode/data, dan acknowledgement.

Setelah semua `\todo{}` terisi, hapus definisi `\newcommand{\todo}` beserta pemanggilannya (cari-ganti ke teks final) sebelum submit ke JUTIF.
