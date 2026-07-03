<!--
TEMPLATE -- ditulis setelah Bab 4 selesai diisi dengan hasil nyata.
Struktur mengikuti ToC kerangka FedX-Palm generasi pertama.
-->

# CHAPTER 5 -- KESIMPULAN DAN SARAN (TEMPLATE)

## 5.1 Kesimpulan

*(Satu paragraf ringkas per rumusan masalah, Subbab 1.2 -- tulis setelah
Bab 4 terisi. Jangan menyalin kesimpulan kerangka generasi pertama, karena
dataset dan konfigurasi berbeda dapat menghasilkan simpulan numerik yang
berbeda meski arah kualitatifnya serupa.)*

1. *(Kesimpulan atas rumusan masalah 1 -- kelayakan arsitektur FL.)*
2. *(Kesimpulan atas rumusan masalah 2 -- dampak DP-SGD terhadap utilitas.)*
3. *(Kesimpulan atas rumusan masalah 3 -- validitas penjelasan Grad-CAM++.)*

## 5.2 Keterbatasan Penelitian

*(Rujuk Subbab 4.12, dirangkum ulang di sini secara lebih ringkas.)*

## 5.3 Saran untuk Penelitian Selanjutnya

*(Contoh arah yang relevan berdasarkan keterbatasan Bab 3/4 -- sesuaikan
setelah hasil nyata diperoleh:)*

- Replikasi tiap sel *grid* eksperimen dengan *seed* berbeda untuk
  memperoleh interval kepercayaan pada mAP dan ε.
- Pelatihan FL terdistribusi lintas-*host* fisik sungguhan (bukan simulasi
  satu GPU), untuk mengukur biaya komunikasi nyata.
- Eksplorasi mekanisme privasi lain (*secure aggregation*, DP-Adam) sebagai
  pembanding DP-SGD per-sampel.
- Evaluasi pada varian YOLOv11 yang lebih besar (s/m) jika target mAP
  belum tercapai dengan varian *nano*, dengan konsekuensi *deployment*
  yang perlu dievaluasi ulang (lihat Subbab 3.12).

## 5.4 Implikasi Praktis untuk Industri Perkebunan

*(Tulis setelah Bab 4 terisi -- kaitkan dengan kelayakan operasional model
pada Subbab 4.1 dan demonstrasi deployment pada Subbab 4.11.)*
