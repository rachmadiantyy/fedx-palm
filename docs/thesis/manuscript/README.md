# Manuskrip JUTIF -- FedXPalm B1/B2

- `manuscript.tex` -- sumber LaTeX (isi sama persis dengan `JUTIF_Manuscript_FedXPalm_B1B2.docx`, dipertahankan sebagai referensi/backup). Header/footer meniru `header1.xml`/`footer1.xml` template JUTIF asli (nama jurnal, Vol/No/Bulan/Tahun/Halaman, P-ISSN/E-ISSN, URL jurnal, DOI, nomor halaman otomatis).
- `manuscript.pdf` -- hasil build dari `manuscript.tex`.
- `fedxpalm_manuscript_overleaf.zip` -- paket siap unggah ke Overleaf (isinya `manuscript.tex` + `figure_b2_convergence.png`).
- `figure_b2_convergence.png` + `b2_k4_seed{42,123,2026}_history.json` -- Gambar 6 (kurva konvergensi B2) dan data sumbernya, lihat `scripts/43_plot_b2_convergence.py`.

## Cara pakai di Overleaf (tidak perlu install apa-apa)

1. Buka [overleaf.com](https://www.overleaf.com) -> **New Project** -> **Upload Project**.
2. Pilih file `fedxpalm_manuscript_overleaf.zip`.
3. Overleaf otomatis compile begitu project terbuka (kalau tidak, klik tombol **Recompile** di atas panel PDF).
4. Kalau nanti nambah gambar baru (Gambar 1/2/4/5), upload file gambarnya lewat menu **Upload** di panel kiri Overleaf, lalu ganti kotak `\todo{...}` yang bersangkutan di `manuscript.tex` dengan `\includegraphics[width=0.85\linewidth]{nama_file.png}` (contoh polanya persis seperti Gambar 6 yang sudah ada).

## Cara build lokal (opsional, kalau tidak pakai Overleaf)

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
