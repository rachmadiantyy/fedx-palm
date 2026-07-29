<!--
CATATAN PENULISAN (hapus komentar ini sebelum submit):
Ditulis berdasarkan hasil NYATA B1/B2 pada Bab 4 (dikutip dari manuskrip
JUTIF, docs/thesis/manuscript/). Saran pada Subbab 5.3 memuat DP-SGD, XAI,
dan deployment sebagai ARAH LANJUTAN yang belum dikerjakan pada tahap ini
-- bukan diklaim sebagai bagian dari kontribusi bab ini.
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
   0{,}5$) mendekati utilitas pelatihan tersentral (B1) pada tugas ini:
   B1 mencapai mAP50 *held-out test* 0,8820 (mAP50-95 = 0,7309), sedangkan
   B2 mencapai mAP50 *validation* rata-rata 0,8775 (SD = 0,0157) di tiga
   *seed* pelatihan independen -- selisih indikatif hanya 0,0045 -- meski
   ketimpangan data antar klien substansial (8,67% hingga 59,36% dari
   *split train* per klien). Ini menjawab rumusan masalah 2 (Subbab 1.2)
   dan mendukung H1 (Subbab 1.6): FedAvg tanpa DP di bawah partisi Non-IID
   yang diuji menghasilkan model dengan utilitas yang mendekati *baseline*
   tersentral dan konvergensi yang relatif stabil lintas *seed*. Perlu
   dicatat bahwa perbandingan ini bersifat indikatif, karena B1 dilaporkan
   pada *split held-out test* sedangkan B2 pada *split validation*
   (Subbab 3.8, 4.6) -- perbandingan yang tersandingkan penuh ditunda ke
   pelaporan lanjutan.

Kesimpulan atas dampak DP-SGD terhadap utilitas dan validitas interpretasi
Grad-CAM++ -- yang menjadi bagian rumusan masalah kerangka FedX-Palm secara
lebih luas -- **tidak** ditarik pada tahap ini, karena kedua komponen
tersebut berada di luar cakupan penelitian tahap ini (Subbab 1.4).

## 5.2 Keterbatasan Penelitian

Simulasi FL dijalankan sekuensial pada satu GPU, bukan multi-*host* fisik,
sehingga hanya menangkap heterogenitas statistik klien, bukan faktor sistem
terdistribusi nyata (latensi, klien putus koneksi). Dataset berasal dari
satu sumber, sehingga generalisasi lintas perkebunan/kultivar/kamera belum
teruji. B2 direplikasi pada tiga *seed*, cukup untuk mengindikasikan
stabilitas namun belum memberi interval kepercayaan yang ketat. Keempat
klien adalah pecahan data tersimulasi, bukan perkebunan/afdeling fisik yang
berbeda. B2 pada tahap ini hanya dilaporkan pada *split validation*; belum
ada perbandingan B1-versus-B2 yang tersandingkan penuh pada *split
held-out test* yang sama. Rincian lengkap ada pada Subbab 4.5.

## 5.3 Saran untuk Penelitian Selanjutnya

Arah berikut membangun langsung di atas titik rujukan B1/B2 yang ditetapkan
pada penelitian ini, dan mencakup komponen yang secara sengaja ditunda dari
cakupan tahap ini (Subbab 1.4, 3.9):

- **Evaluasi B2 pada *split held-out test*.** Menjalankan evaluasi
  *held-out test* untuk ketiga *checkpoint seed* B2 yang sudah terkunci,
  agar perbandingan B1-versus-B2 tersandingkan penuh pada *split* yang
  sama (bukan lagi indikatif seperti Subbab 4.3).
- **Perluasan *Differential Privacy* (DP-SGD).** Menerapkan DP-SGD
  per-sampel pada *checkpoint* GroupNorm B1/B2 yang sudah kompatibel
  secara arsitektural (Subbab 2.5), pada konfigurasi *full* (seluruh
  parameter) dan *partial* (*backbone* beku), dengan akuntansi privasi ε
  per klien mengingat ketimpangan ukuran data antar klien (Tabel 3.2, dan
  implikasinya pada Subbab 4.4.2).
- **Integrasi *Explainable AI* (Grad-CAM++).** Menambahkan validasi
  interpretasi visual (*Average Drop*, *Focus Retention Rate*) atas model
  operasional yang dipilih dari B2, khususnya untuk menyelidiki kelas
  *Ripe* yang berperforma lemah pada B1 (Subbab 4.1).
- **Demonstrasi *deployment*.** Mengemas model operasional sebagai layanan
  inferensi (infrastruktur `deployment/` sudah tersedia pada repositori)
  untuk menguji portabilitas praktis di luar lingkungan pelatihan GPU.
- **Replikasi tambahan dan pelatihan FL terdistribusi nyata.** Menambah
  jumlah *seed* B2 untuk interval kepercayaan yang lebih ketat pada mAP,
  dan menguji pelatihan FL lintas-*host* fisik sungguhan untuk mengukur
  biaya komunikasi nyata, melengkapi simulasi satu-GPU pada tahap ini.
- **Investigasi kelas *Ripe*.** Analisis *confusion matrix* dan
  tingkat-dataset untuk mengidentifikasi penyebab performa AP50 yang jauh
  lebih rendah pada kelas *Ripe* dibanding kelas lain pada B1 (Subbab 4.1,
  4.4).

## 5.4 Implikasi Praktis untuk Industri Perkebunan

Selisih utilitas yang kecil antara B1 dan B2 pada penelitian ini
mengindikasikan bahwa, untuk tugas dan dataset ini, berpindah dari
pelatihan tersentral ke pelatihan federasi Non-IID membawa biaya utilitas
yang relatif kecil -- sehingga operator perkebunan yang menghadapi kendala
berbagi-data atau bandwidth jaringan tidak perlu mengorbankan akurasi
deteksi secara signifikan untuk menghindari pemusatan citra kebun ke satu
server. Temuan ini juga memotivasi penempatan B2, bukan B1, sebagai titik
rujukan utama ketika mengukur biaya utilitas tambahan yang diperkenalkan
oleh *Differential Privacy* pada penelitian lanjutan (Subbab 4.4.2), karena
biaya federasi itu sendiri sudah relatif kecil terhadap *baseline*
tersentral. Implikasi praktis yang lebih konkret (mis. kelayakan
operasional model dan demonstrasi antarmuka *deployment*) menunggu
komponen yang ditunda pada Subbab 5.3.
