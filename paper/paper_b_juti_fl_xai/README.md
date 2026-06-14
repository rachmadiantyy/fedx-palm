# Paper B — JUTI (Jurnal Ilmiah Teknologi Informasi), ITS — SINTA 3

Manuskrip kedua dari proyek FedX-Palm, mengikuti **template JUTI EN** resmi
(single-column A4, Times Roman 10pt, English).

- **Sumber tunggal:** `main.tex` (compile dengan pdfLaTeX di Overleaf).
- `draft.md` versi awal sudah dihapus — semua isi sekarang di `main.tex`.

## Logo JUTI (opsional)

`main.tex` punya slot otomatis untuk logo JUTI di header kiri. **Tanpa
logo, paper tetap compile bersih** (fallback ke header teks-only) — editor
JUTI biasanya menambahkan logo saat final formatting.

Kalau mau menyertakan sendiri:

1. Buka template DOCX JUTI yang kamu unduh (di komputer): klik kanan logo
   di pojok kiri-atas → *Save as Picture* → simpan sebagai
   `figures/juti_logo.png`. **ATAU**
2. Kunjungi situs jurnal JUTI ITS (juti.if.its.ac.id) → unduh asetnya
   resmi.
3. Letakkan file di `paper/paper_b_juti_fl_xai/figures/juti_logo.png`,
   compile ulang — logo otomatis muncul. Tidak perlu edit LaTeX.

## Cara upload ke Overleaf hari ini

1. Overleaf → **New Project → Upload Project** → unggah folder
   `paper_b_juti_fl_xai/` (atau zip-nya).
2. Compiler: **pdfLaTeX**. Tekan **Recompile**.
3. PDF jadi. Semua angka yang belum ada tampil **MERAH** sebagai
   `[FILL: keterangan]`.
4. **WAJIB** ganti semua `\tbd{...}` (merah) dengan angka NYATA sebelum
   submit. Sumber angka:
   - Akurasi (Tabel I, Tabel II): hasil `aggregate_results.py` →
     `thesis_rebuild/tables/runs_master.csv` baris `exp=B2`.
   - Faithfulness per-kelas (Tabel III): hasil `evaluate_xai.py` →
     `thesis_rebuild/tables/xai_per_class.csv`.
   - Gambar: `figures/xai_per_class.png` (rangkai 2×3 *heatmap*
     representatif, satu per kelas).

## Pemenuhan template JUTI

| Item | Status |
|---|---|
| Single-column A4 | ✅ |
| Title UPPERCASE | ✅ |
| 150–250 word abstract | ✅ (≈230 kata) |
| 3–5 keywords, alfabetis | ✅ (5: explainable artificial intelligence, federated learning, Grad-CAM++, oil palm ripeness, YOLOv11) |
| Section numbering "1.", "1.1." (subsection italic) | ✅ |
| Table dengan Roman numerals | ✅ |
| "Fig." abbreviated, "Table" tidak | ✅ |
| Header genap/ganjil JUTI | ✅ |
| Footer halaman 1 (corresponding, received, CC BY-SA, DOI) | ✅ |
| CRediT Authorship Contribution Statement | ✅ |
| Declaration of Competing Interest | ✅ |
| Acknowledgments | ✅ (opsional, bisa dihapus) |
| Data Availability | ✅ |
| Declaration of Generative AI | ✅ |
| References dengan format JUTI | ✅ (10 referensi inti) |
| Min. 9 halaman | Diperkirakan ~9–10 (dengan figure terisi) |

## Hubungan dengan Paper A (anti self-plagiarism)

| Aspek | Paper A (Scopus Q3, di-condense dari tesis nanti) | Paper B (ini, SINTA 3) |
|---|---|---|
| **Pertanyaan riset** | Mengapa DP-FedAvg level-klien collapse, dan bukti DP-SGD per-sampel memberi trade-off privasi-utilitas yang sah? | Bisakah FL menghasilkan deteksi sawit yang akurat **dan dapat dijelaskan/dipercaya** untuk pendukung keputusan panen? |
| **Bintang utama** | Mekanisme DP (full grid K×σ) | **Explainability** (Grad-CAM++): AD/FRR per-kelas |
| **Sumber hasil** | Full grid DP-SGD baru | B2 federasi (no-DP) + analisis XAI per-kelas |
| **Peran DP** | Sentral | Minor — disebut singkat, **dirujuk ke Paper A** [10] |
| **Peran XAI** | Pendukung (1 angka global) | **Sentral & per-kelas** |

> Catatan etika: mengganti judul saja TIDAK menghindari deteksi plagiarisme
> (checker membandingkan isi). Pembeda sejati = kontribusi & isi yang berbeda
> + saling-sitir antar paper sendiri. Yang diterapkan di sini.

## Daftar `\tbd{...}` yang HARUS diisi

1. Abstract: `B2 mAP50`, `AD`, `FRR`, `best/worst class`.
2. Pendahuluan: tanggal "Available online".
3. Section 2.1: 2–3 studi terbaru deteksi kematangan sawit.
4. Section 3.2: jumlah K (mis. 4).
5. Section 3.5: jumlah N citra uji XAI (mis. 120).
6. Tabel II (akurasi): seluruh baris Federated + FL cost.
7. Tabel III (faithfulness): seluruh sel + baris Global.
8. Section 4.2: best/worst class + interpretasi agronomis.
9. Section 4.3: 1 paragraf perbandingan federated vs centralized AD/FRR.
10. Section 4.4: 1 paragraf interpretasi heatmap kualitatif.
11. Gambar: `figures/xai_per_class.png`.
12. Section 5 (Conclusion): isi sesuai temuan.
13. Acknowledgments: tulis atau hapus.
14. Gen-AI Declaration: tulis alat & versi.
15. Referensi [10] (Paper A): venue Q3 + status.
