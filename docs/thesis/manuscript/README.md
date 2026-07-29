# Manuskrip JUTIF -- FedXPalm B1/B2

**Judul & framing (update):** "Multi-Seed Reproducibility and Convergence Stability of Non-IID Federated YOLO11n for Oil Palm Fresh Fruit Bunch Detection" -- direframe dari versi lama ("Non-IID FL versus Centralized Training") karena perbandingan B1 (held-out test) vs B2 (validation) belum matched. Fokus sekarang: reproducibility & convergence lintas 3 seed federated (bukan perbandingan centralized-vs-federated). Detail penuh audit & rasional reframe ada di `EDITORIAL_REVIEW_REFRAME.md` di folder ini -- sudah diterapkan ke `manuscript.tex`.

- `manuscript.tex` -- sumber LaTeX. Diverifikasi langsung terhadap `JUTIF-Template.docx` (bukan cuma tebakan): margin (2,5/2,5/2,5/2,0 cm kiri/atas/kanan/bawah), *line spacing* 1,15, indentasi baris pertama paragraf 1,1cm, judul bold 14pt, nama penulis bold 10pt, header/footer (nama jurnal, Vol/No/Bulan/Tahun/Halaman, P-ISSN/E-ISSN, URL jurnal, DOI, nomor halaman otomatis), dan -- yang paling penting -- judul bab/subbab di template **TIDAK diberi nomor otomatis** (cuma bold huruf kapital utuh untuk bab utama, bold huruf normal untuk subbab), jadi seluruh referensi "Section N" di teks sudah diganti jadi nama subbab langsung (bukan `\ref{}` ke nomor yang memang tidak ada di template aslinya).
- `JUTIF-Template.docx` -- template resmi kosong dari JUTIF, dipakai sebagai rujukan verifikasi di atas.
- `manuscript.pdf` -- hasil build dari `manuscript.tex`.
- `fedxpalm_manuscript_overleaf.zip` -- paket siap unggah ke Overleaf (isinya `manuscript.tex` + semua gambar yang sudah final).
- `figure_b2_convergence.png` + `b2_k4_seed{42,123,2026}_history.json` -- Gambar 6 (kurva konvergensi B2) dan data sumbernya, lihat `scripts/43_plot_b2_convergence.py`.
- `figure2_sixclass_grid.png` -- Gambar 2 (grid 2x3 contoh 6 kelas ripeness: Unripe, Underripe, Ripe, Overripe, Abnormal, Empty Bunch), disusun dari foto lapangan asli (`frame*.jpg`) dengan label kelas dikonfirmasi langsung oleh penulis.

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

- ~~Nama/urutan co-author, email corresponding author~~ (sudah diisi), tanggal submisi (Received/Revised/Accepted/Published) dan Vol/No/Halaman/DOI (ini memang diisi editor jurnal, bukan tugas penulis).
- ~~Gambar 1 (diagram alur riset)~~, ~~Gambar 2 (contoh 6 kelas)~~ -- keduanya sudah final.
- Gambar 4 (confusion matrix/PR curve B1 di *held-out test* -- jalankan `scripts/42_generate_b1_test_plots.py`), Gambar 5 (contoh deteksi kualitatif) -- masih ditunggu dari penulis.
- Gambar 3 (distribusi klien) sifatnya opsional -- angkanya sudah ada di Tabel 2.
- ~~Pernyataan ketersediaan kode/data, dan acknowledgement~~ (sudah diisi).
- ~~Tabel 5 (best-round vs final-round mAP50 seed 42 & 123)~~, ~~Tabel 6 (per-class AP50 tiga seed B2)~~ -- **sudah terisi dari `b2_k4_seed{42,123,2026}_history.json` yang sudah ada di repo ini**, tidak perlu training/evaluasi ulang. Temuan baru: kelas Underripe punya varians antar-seed jauh lebih besar (SD=0.0727) daripada mAP50 global (SD=0.0157) -- ini langsung menjawab supporting RQ ketiga di paper.
- **Masih tersisa**: Tabel 7 (matched held-out-test B1 vs B2 tiga seed) -- ini **butuh checkpoint B2 asli** yang cuma ada di server training kamu, tidak bisa diisi dari sini. Jalankan evaluasi held-out test untuk ketiga `best.pt` seed (lihat instruksi Gambar 4 sebelumnya, tinggal arahkan ke checkpoint B2 & `--split test`).
- Client-by-class breakdown -- opsional, manuskrip sudah eksplisit menandainya sebagai future work, tidak wajib.
- Beberapa `\todo{NEEDS RECENT REFERENCE: ...}` di Related Work (referensi 2022-2025 soal convergence-round variance & multi-seed FL reproducibility) -- perlu dicari manual, jangan dikarang.

**Perbaikan pasca-review (sudah diterapkan)**: koreksi perhitungan coefficient of variation (0.836 -> 0.822, salah hitung sebelumnya), penghalusan istilah "final performance" jadi "peak/best-validation performance" di beberapa tempat, Gambar 1 (diagram alur) digambar ulang tanpa kotak "Indicative B1 vs B2 Comparison / Δ=0.0045", Gambar 6 (kurva konvergensi) digambar ulang tanpa garis referensi B1, subbab "Matched Held-Out-Test Comparison" diganti nama jadi "Held-Out-Test Evaluation Protocol" tanpa lagi membahas selisih 0,0045, supporting RQ kedua diubah dari klaim kausal ("does imbalance destabilize...") jadi deskriptif, penjelasan sumber stochastic training diperbaiki (bukan "client sampling order" karena keempat klien selalu ikut tiap ronde), ditambah detail nyata soal `deterministic=True` dan fix seeding dataloader dari kode asli (`src/fedxpalm/federated/trainer_utils.py`), dan baris placeholder nomor telepon (`xxxxxxx`) dihapus dari halaman judul karena memang tidak seharusnya tercetak.

Setelah semua `\todo{}` terisi, hapus definisi `\newcommand{\todo}` beserta pemanggilannya (cari-ganti ke teks final) sebelum submit ke JUTIF.
