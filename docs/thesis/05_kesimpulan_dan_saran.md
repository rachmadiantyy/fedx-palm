<!--
CATATAN PENULISAN (hapus komentar ini sebelum submit):
Bab ini diperluas agar konsisten dengan cakupan penuh Bab 1-4: mencakup
kesimpulan dan saran atas DP-SGD (E1/E2), XAI (Grad-CAM++), biaya
komputasi/komunikasi, dan deployment -- yang pada draf sebelumnya sengaja
ditunda sebagai "arah lanjutan belum dikerjakan" (lihat riwayat pada
docs/thesis/NOTES_FOR_RACHMA.md). Seluruh komponen tersebut sekarang
SUDAH dikerjakan dan dilaporkan dengan hasil nyata pada Bab 4, sehingga
kesimpulan/keterbatasan/saran di bawah ini ditulis berdasarkan hasil
NYATA tersebut, bukan lagi sebagai rencana yang belum terealisasi.
-->

# CHAPTER 5 -- KESIMPULAN DAN SARAN

## 5.1 Kesimpulan

1. Arsitektur *Federated Learning* berbasis FedAvg berhasil dibangun dan
   diintegrasikan dengan YOLOv11n untuk klasifikasi enam tingkat kematangan
   TBS kelapa sawit, dievaluasi pada dataset publik Roboflow dengan
   pembagian *train*/*validation*/*held-out test* bebas-kebocoran pada
   level kelompok sumber (8.937 / 826 / 1.051 citra) -- menjawab rumusan
   masalah 1 (Subbab 1.2).

2. Pelatihan federasi Non-IID (B2; FedAvg, K = 4, Dirichlet $\alpha =
   0{,}5$) mendekati utilitas pelatihan tersentral (B1) secara indikatif:
   B1 mencapai mAP50 *held-out test* 0,8820 (mAP50-95 = 0,7309), sedangkan
   B2 mencapai mAP50 *validation* rata-rata 0,8775 (SD = 0,0157) di tiga
   *seed* pelatihan independen -- selisih indikatif hanya 0,0045, meski
   ketimpangan data antar klien substansial (8,67% hingga 59,36% dari
   *split train* per klien). Namun, ketika *checkpoint* B2 *seed* 42
   dievaluasi tersandingkan penuh pada *split held-out test* yang sama
   dengan B1, selisih *validation*-ke-*test*-nya ternyata jauh lebih besar
   (0,8850 → 0,7951) -- menunjukkan bahwa perbandingan berbasis *validation*
   semata dapat meremehkan gap utilitas federasi yang sesungguhnya. Ini
   menjawab rumusan masalah 2 dan mendukung H1 (Subbab 1.6) secara
   sebagian: FedAvg tanpa DP menghasilkan konvergensi yang relatif stabil
   lintas *seed*, namun besaran gap terhadap *baseline* tersentral perlu
   diukur pada *split* yang tersandingkan penuh, bukan perbandingan
   indikatif lintas-*split*.

3. Penambahan DP-SGD (E1 *full*-parameter, E2 *partial*-parameter; σ=0,75,
   C=1, 20 ronde) di atas titik rujukan B2 *seed* 42 menyebabkan
   ***severe utility degradation*** pada *split held-out test* tersandingkan
   penuh -- mAP50 turun dari 0,7951 (B2) ke 0,2116 (E1) dan 0,2127 (E2) --
   namun **bukan *total collapse* untuk keseluruhan model**: kedua model DP
   tetap menunjukkan pembelajaran yang koheren dan *recall* yang cukup
   tinggi pada sebagian kelas. E1 dan E2 tercatat **praktis setara** pada
   *held-out test* meski cakupan parameter *trainable*-nya sangat berbeda
   (2.591.010 vs 929.522) -- pembatasan cakupan parameter (*partial* DP)
   **tidak** memberikan keunggulan utilitas maupun privasi yang jelas
   (ε maksimum identik, 22,106, karena akuntansi PRV ditentukan oleh jumlah
   langkah optimisasi, bukan jumlah parameter), hanya keunggulan komputasi
   (E2 ~15,1% lebih cepat per ronde dari E1). Ini menjawab rumusan masalah
   3 dan sebagian besar mendukung H2 (Subbab 1.6).

4. *Flat clipping* (C=1) menunjukkan keunggulan arah yang konsisten pada
   tiga *seed* validasi dibanding *per-layer clipping* dan dipilih sebagai
   konfigurasi *final* E1/E2, meski *per-layer clipping* menunjukkan
   variansi antar-*seed* yang jauh lebih kecil (SD=0,0043 berbanding
   0,0208) -- suatu *trade-off* rata-rata-versus-varians, bukan
   keunggulan mutlak salah satu strategi. Anggaran privasi (ε) per klien
   ternyata **tidak berbanding lurus dengan ukuran data klien**: klien
   terkecil (Klien 1, 775 citra) mencapai ε tertinggi (22,1062), sedangkan
   klien terbesar (Klien 2, 5.305 citra) mencapai ε terendah (7,8085),
   didorong oleh laju *sampling* $q$ yang lebih tinggi pada klien
   bersampel sedikit. Ini menjawab rumusan masalah 4.

5. Analisis Grad-CAM++ tersandingkan pada 3.821 sampel identik lintas B2,
   E1, dan E2 menunjukkan *Average Drop* dan *Focus Retention Rate* yang
   secara numerik lebih tinggi pada kedua model DP dibanding B2, namun
   perbedaan ini **tidak dapat langsung ditafsirkan** sebagai penjelasan
   visual yang lebih *faithful* pada model DP, karena metrik *Average Drop*
   diketahui bias terhadap model dengan keyakinan dasar yang rendah dan
   rapuh -- persis kondisi E1/E2 dibanding B2. Kelas *Empty Bunch*
   mengalami *zero-recall collapse* total pada kedua model DP (*recall*
   =0,0000 dari 760 instans), memperparah kelemahan yang sudah ada pada B2
   (*recall*=0,3212) menjadi kegagalan absolut. Model operasional (B2
   *seed* 42) berhasil didemonstrasikan berjalan sebagai layanan inferensi
   Docker pada VPS, dengan pengujian fungsional yang berhasil (deteksi
   kelas *Ripe* pada 93,4% keyakinan disertai visualisasi Grad-CAM++ yang
   tervisualisasi sesuai ekspektasi). Ini menjawab rumusan masalah 5 dan
   mendukung H3 (Subbab 1.6): pola perhatian visual berbeda secara
   kualitatif antar model, namun interpretasi kuantitatifnya menuntut
   kehati-hatian metodologis.

## 5.2 Keterbatasan Penelitian

Pertama, pelatihan federasi disimulasikan secara sekuensial pada satu GPU,
bukan multi-*host* fisik, sehingga hanya menangkap heterogenitas statistik
klien, bukan faktor sistem terdistribusi nyata (latensi, klien putus
koneksi); volume komunikasi yang dilaporkan bersifat teoretis, bukan
pengukuran jaringan sungguhan. Kedua, dataset berasal dari satu sumber,
sehingga generalisasi ke perkebunan, kultivar, kamera, atau kondisi
pencahayaan lain belum dievaluasi langsung. Ketiga, meski B2 dan pemilihan
strategi *clipping* divalidasi tiga *seed*, konfigurasi DP-SGD *final*
(E1/E2) hanya dijalankan pada **satu *seed* (42)** karena keterbatasan
waktu komputasi. Keempat, ε maksimum yang dicapai (22,11) tergolong
relatif longgar sebagai jaminan privasi formal, dan `secure_mode` Opacus
(RNG kriptografis) tidak diaktifkan pada seluruh eksperimen tahap ini --
konsisten untuk perbandingan internal, namun perlu diaktifkan sebelum
klaim jaminan privasi produksi. Kelima, keempat klien adalah pecahan data
tersimulasi dari satu *split train*, bukan representasi organisasi fisik
yang berbeda. Keenam, kalibrasi ambang *per-layer clipping* bersifat
*data-dependent* dan tidak masuk dalam akuntansi privasi formal. Ketujuh,
kelas *Empty Bunch* mengalami *zero-recall collapse* pada kedua model DP
-- kesimpulan privasi-utilitas penelitian ini tidak berlaku merata di
seluruh kelas. Kedelapan, analisis XAI bersifat *post-hoc* semata dan
tidak memengaruhi pemilihan *checkpoint*/konfigurasi manapun; metrik
*Average Drop* memiliki keterbatasan yang telah diketahui dalam
membandingkan model dengan tingkat keyakinan prediksi yang jauh berbeda.
Kesembilan, demonstrasi *deployment* pada VPS dijalankan sebagai satu
*container* `docker run` tanpa mekanisme *restart* otomatis dan tanpa
*reverse proxy* -- cukup untuk demonstrasi fungsional, namun bukan
konfigurasi tingkat produksi. Rincian lengkap ada pada Subbab 4.10.3.

## 5.3 Saran untuk Penelitian Selanjutnya

Arah berikut teridentifikasi langsung dari temuan dan keterbatasan
penelitian ini (Subbab 5.2, 4.10.4):

- **Replikasi multi-*seed* untuk DP-SGD *final*.** Menjalankan E1/E2 pada
  lebih dari satu *seed* pelatihan untuk memperkuat klaim generalisasi
  hasil *privacy-utility trade-off*, melengkapi validasi tiga-*seed* yang
  sejauh ini baru diterapkan pada pemilihan strategi *clipping* (Subbab
  4.4) dan B2 (Subbab 4.2).
- **Strategi *class-aware* untuk mengatasi *zero-recall collapse*.**
  Merancang *sampling* atau fungsi kerugian berbobot untuk mengatasi
  kegagalan deteksi total kelas *Empty Bunch* di bawah DP-SGD (Subbab
  4.5.6), yang tidak teratasi oleh pemilihan strategi *clipping* maupun
  cakupan parameter *trainable* pada penelitian ini.
- **Implementasi komunikasi hemat-parameter untuk *partial* DP.**
  Memperbaiki `fedavg()` agar memfilter parameter berdasarkan status
  *trainable* sebelum agregasi, merealisasikan potensi penghematan
  komunikasi ~64% untuk E2 yang sejauh ini baru bersifat estimasi
  hipotetis (Subbab 4.8.3).
- ***Clipping* adaptif yang tetap *privacy-preserving*.** Mengatasi
  *trade-off* rata-rata-versus-varians antara *flat* dan *per-layer
  clipping* (Subbab 4.4.4) dengan skema ambang yang beradaptasi tanpa
  mengorbankan akuntansi privasi formal.
- ***Secure aggregation* dan penguatan jaminan privasi produksi.**
  Mengaktifkan `secure_mode` Opacus (RNG kriptografis) dan menjajaki
  *secure aggregation* tingkat-klien sebagai lapisan privasi tambahan,
  serta menguji konfigurasi dengan ε yang lebih ketat, sebelum klaim
  jaminan privasi tingkat produksi dapat diajukan.
- **Pelatihan FL terdistribusi nyata dan replikasi tambahan B2.**
  Menambah jumlah *seed* B2 untuk interval kepercayaan yang lebih ketat
  pada mAP, dan menguji pelatihan FL lintas-*host* fisik sungguhan untuk
  mengukur biaya komunikasi nyata, melengkapi simulasi satu-GPU dan
  estimasi komunikasi teoretis pada penelitian ini.
- **Investigasi lanjutan kelas *Ripe* dan *Underripe*.** Analisis lebih
  dalam atas kesalahan batas *Ripe*/*Underripe* yang teridentifikasi lewat
  *confusion matrix* B1 (Subbab 4.2.1), termasuk kemungkinan augmentasi
  data atau fitur tambahan yang membedakan kedua tahap kematangan yang
  bertetangga ini secara lebih tegas.
- ***Deployment* tingkat produksi.** Menambahkan mekanisme *restart*
  otomatis, *reverse proxy*, dan pengukuran performa formal (*latency*,
  *throughput*) pada demonstrasi *deployment* yang sejauh ini baru
  membuktikan kelayakan fungsional (Subbab 4.9), sebelum diklaim siap
  untuk penggunaan produksi.
- **Validasi lapangan yang lebih luas.** Menguji model pada data dari
  perkebunan, kultivar, kamera, dan kondisi pencahayaan yang lebih
  beragam di luar sumber dataset tunggal yang dipakai penelitian ini.

## 5.4 Implikasi Praktis untuk Industri Perkebunan

Selisih utilitas yang kecil antara B1 dan B2 secara indikatif (Subbab 4.2.3)
mengindikasikan bahwa, untuk tugas dan dataset ini, berpindah dari
pelatihan tersentral ke pelatihan federasi Non-IID tanpa DP membawa biaya
utilitas yang relatif kecil pada level *validation* -- meski perbandingan
tersandingkan penuh pada *held-out test* menunjukkan gap yang lebih besar
(0,0789) dari yang terlihat pada perbandingan indikatif. Operator
perkebunan yang menghadapi kendala berbagi-data atau bandwidth jaringan
dapat mempertimbangkan federasi sebagai alternatif yang layak, dengan
catatan bahwa evaluasi *held-out test* yang tersandingkan penuh -- bukan
sekadar *validation* -- diperlukan untuk estimasi biaya utilitas yang
akurat. Sebaliknya, penambahan *Differential Privacy* (DP-SGD) di atas
federasi ini membawa biaya utilitas yang jauh lebih besar (mAP50 turun ke
kisaran 0,21) -- pada anggaran privasi (σ=0,75, C=1, ε≈22) yang diuji,
DP-SGD **belum layak** diadopsi untuk operasi produksi tanpa penyesuaian
lebih lanjut (mis. arsitektur atau strategi pelatihan yang lebih tahan
*noise*, atau anggaran privasi yang berbeda). Analisis XAI menegaskan
pentingnya inspeksi visual langsung -- bukan sekadar metrik AD/FRR agregat
-- sebelum mempercayai penjelasan visual model, khususnya pada model dengan
keyakinan prediksi yang rendah. Demonstrasi *deployment* yang berhasil
menunjukkan bahwa model federasi non-privat (B2) sudah layak secara
fungsional untuk uji coba operasional terbatas, meski konfigurasi produksi
penuh (ketahanan, keamanan jaringan, performa terukur) masih memerlukan
pekerjaan lanjutan (Subbab 5.3).
