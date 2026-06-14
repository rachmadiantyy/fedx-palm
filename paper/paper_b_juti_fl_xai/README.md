# Paper B — JUTI ITS (SINTA 3)

Draf artikel jurnal **kedua** dari proyek FedX-Palm, ditujukan ke
**JUTI (Jurnal Teknik Informatika dan Sistem Informasi), ITS — SINTA 3**.

## Hubungan dengan Paper A (dan kenapa ini BUKAN self-plagiarism)

| Aspek | Paper A (IEEE, `../fedx_palm_ieee.tex`) | Paper B (ini) |
|---|---|---|
| **Pertanyaan riset** | *Kenapa* DP level-klien (DP-FedAvg) collapse pada detektor, dan bagaimana memitigasinya? | *Bisakah* FL menghasilkan deteksi sawit yang akurat **dan dapat dijelaskan/dipercaya** untuk pendukung keputusan panen? |
| **Bintang utama** | Karakterisasi failure mode DP + mitigasi clip-norm | **Explainability** (Grad-CAM++): AD/FRR per-kelas, Insertion/Deletion, validasi agronomis |
| **Peran DP** | Sentral | Minor — disebut singkat, **dirujuk ke Paper A** untuk detail |
| **Peran XAI** | Pendukung (1 angka AD/FRR global) | **Sentral & diperdalam** (per-kelas + metrik tambahan + interpretasi visual) |
| **Sudut** | Sistem/privasi | Aplikasi pertanian + trustworthy AI |
| **Data** | Lama (9.030/935, mAP 0.995) | Rebuild (9.094/769/951, anti-leakage bunch_id) |
| **Audiens** | Venue ML/security | Informatika terapan / sistem informasi |

**Prinsip anti-self-plagiarism yang dipakai di sini:**
1. Body text ditulis ulang dari nol dengan RQ & narasi berbeda — bukan parafrase Paper A.
2. Bagian yang tak terhindarkan tumpang-tindih (deskripsi dataset, arsitektur YOLOv11) ditulis ringkas dan **menyitir Paper A** alih-alih mengulang panjang lebar.
3. Hasil inti Paper B (analisis XAI per-kelas + Insertion/Deletion + validasi agronomis) **tidak ada** di Paper A → kontribusi baru yang nyata.

> Catatan etika: mengganti judul saja TIDAK menghindari deteksi plagiarisme
> (checker membandingkan isi). Pembeda sejati adalah kontribusi & isi yang
> berbeda + saling-sitir antar paper sendiri. Itu yang diterapkan di sini.

## Status

Draf (`draft.md`) berisi struktur IMRaD lengkap dengan `{TBD: ...}` untuk
angka yang menunggu: (a) full grid FL selesai, (b) script XAI per-kelas
dijalankan. Setelah keduanya ada, isi `{TBD}` lalu port ke template Word/LaTeX
JUTI.
