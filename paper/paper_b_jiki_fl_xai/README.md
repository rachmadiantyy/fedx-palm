# Paper B — JIKI UI version (SINTA 2)

Konversi Paper B dari JUTI (SINTA 3) ke **JIKI UI** (Jurnal Ilmu Komputer dan
Informasi, FASILKOM Universitas Indonesia, SINTA 2). Setelah ditolak JUTI
karena dianggap "terlalu fokus ML/FL", paper diarahkan ke JIKI yang scope-nya
**eksplisit mencakup machine learning, federated learning, dan computer
vision** — pas sekali dengan kontribusi paper ini.

Versi JUTI (`../paper_b_juti_fl_xai/`) sengaja dipertahankan sebagai cadangan.

## Cara compile

JIKI memakai template **IEEEtran** (LaTeX). File pendukungnya
(`IEEEtran.cls`, `algorithm2e.sty`, `balance.sty`, `caption.sty`, `cite.sty`,
`import.sty`, `IEEEabrv.bib`, `IEEEexample.bib`) **sudah ada di zip
"AuthorGuideline_JIKI"** dari JIKI. Letakkan SEMUA file `.cls` dan `.sty` itu
di folder ini sebelum compile.

```bash
# Compile sequence:
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

Atau di **Overleaf**: New Project → Upload Project → unggah folder ini
beserta semua `.cls`/`.sty` dari zip JIKI. Compiler **pdfLaTeX**.

## File yang dibutuhkan

| File | Sumber |
|---|---|
| `main.tex` | sudah ada (versi JIKI) |
| `references.bib` | sudah ada (sama dengan versi JUTI) |
| `xai_per_class.png` | **harus disalin** dari `../paper_b_juti_fl_xai/figures/xai_per_class.png` |
| `IEEEtran.cls` + `*.sty` + `IEEEabrv.bib` + `IEEEexample.bib` | **dari zip JIKI** |

## Perubahan dibanding versi JUTI

- **Template berubah total** dari Typst (`@preview/juti`) ke LaTeX IEEEtran
  (`twocolumn, conference, compsoc`, A4, Times 10pt).
- **Title diringkas** menjadi 13 kata (JIKI batas 20).
- **Abstract diringkas** ke ~200 kata (JIKI batas 200).
- **5 keywords alfabetis** (JIKI batas 5).
- **Sitasi** numbered `\cite{key}` IEEE-style (JIKI pakai `natbib`
  `[numbers,sort&compress]`).
- **Ditambah** untuk memperkuat angle ML/FL:
  - Section 3.1 **formalisasi cross-silo FL** dengan persamaan tujuan
    global `min f(w)` Eq. (1) dan FedAvg Eq. (2).
  - Section 3.7 **Grad-CAM++ recap** dengan persamaan bobot $w_k^c$
    Eq. (3), serta justifikasi pemilihan layer (last shared conv sebelum
    decoupled head).
  - **Algorithm 1** memformalkan pipeline FL+faithfulness end-to-end.
  - Diskusi **threat-to-validity** ditambah baris "communication-constrained
    FL framing" agar reviewer melihat bahwa $T=5$ adalah pilihan desain
    yang dimotivasi, bukan kekurangan.
- Hasil empiris **identik** dengan versi JUTI (B2 $K{=}4$ mAP 0.738; AD/FRR
  per-kelas dari `xai_per_class.csv`; tabel federated-vs-centralized dari
  `xai_per_class_B1.csv`). Tidak ada angka baru yang difabrikasi.

## Daftar `\fill{...}` yang HARUS diisi sebelum submit

1. **Author block** (3 author): nama, institusi, email.
   *(Untuk first submission JIKI, biarkan anonim sesuai template.)*
2. **Acknowledgement**: nama departemen + universitas; nama lengkap dua
   pembimbing tesis (Pak Favian dan Prof. Teddy, dari cover tesis).
3. **Section 3.3 Non-IID statistics** (Chi-square + JSD + Tabel I) -- isi
   dari output script:

   ```bash
   python thesis_rebuild/scripts/quantify_noniid.py --K 4 \
       --out thesis_rebuild/tables/noniid_K4.md
   ```

   Script melaporkan $\chi^2$, df, p, mean pairwise JSD, dan tabel
   distribusi kelas per-klien -- semua angka yang dibutuhkan paper.

## Apa yang ditambahkan vs versi awal JIKI

Setelah memeriksa dua paper acuan JIKI (Riyadi 2026 untuk FL+IoT;
Permana 2025 untuk YOLOv11+waste), ditambahkan empat hal untuk
mencocokkan ekspektasi JIKI:

- **Fig. 1 (TikZ flow diagram)** -- end-to-end pipeline alur: data
  split -> Dirichlet partition -> BN$\rightarrow$GN -> FedAvg -> Grad-CAM++ ->
  per-class AD/FRR. Visual sistem seperti Permana Fig. 2.
- **Fig. 2 (TikZ YOLOv11 schematic)** -- backbone/neck/head dengan
  highlight 81 BN $\rightarrow$ GroupNorm. Memperjelas arsitektur seperti
  Permana Fig. 1.
- **Section 3.3 quantification block** -- Chi-Square + JSD untuk
  partisi $K=4$, mirip protokol Riyadi (Chi^2=240k, JSD>0.5).
- **5 referensi tambahan**: FedBN, FedProx, SCAFFOLD, DIoU, Zhao
  non-IID. References total ~21.
