<!--
CATATAN PENULISAN (hapus komentar ini sebelum submit):
Bab ini diisi dengan angka NYATA hasil eksperimen B1/B2 yang sudah
dijalankan penulis di GPU (NVIDIA GeForce RTX 4080, 16 GB VRAM), dikutip
langsung dari docs/thesis/manuscript/JUTIF_Manuscript_FedXPalm_B1B2.docx.
Bagian E1/E2 (DP-SGD), XAI (Grad-CAM++), dan deployment yang ada pada draf
template sebelumnya DIHAPUS karena di luar cakupan tahap ini (Subbab 1.4).
Item [TODO: ...] yang masih ada di manuskrip (mis. penyebab pasti kelas
Ripe lemah, code/data availability statement) dipertahankan sebagai
[TODO] di sini juga -- JANGAN diisi dengan tebakan.
-->

# CHAPTER 4 -- HASIL DAN PEMBAHASAN

Bab ini melaporkan hasil untuk *baseline* tersentral (B1) dan *baseline*
federasi Non-IID tanpa DP (B2). Hasil untuk perluasan DP-SGD (*Full* dan
*Partial*) yang direncanakan berada di luar cakupan bab ini dan akan
dilaporkan terpisah pada tahap penelitian lanjutan (Subbab 3.9, 5.3).

## 4.1 B1: Hasil *Baseline* Tersentral

Tabel 4.1 melaporkan hasil akhir B1 pada *split held-out test* (1.051
citra, Subbab 3.3.2), setelah *checkpoint* dipilih berdasarkan performa
*validation* (Subbab 3.8).

Tabel 4.1. B1 *baseline* tersentral -- hasil *held-out test*

| Kelas | AP50 | AP50-95 |
|---|---|---|
| Abnormal | 0,9924 | 0,8462 |
| Empty Bunch | 0,9817 | 0,7521 |
| Overripe | 0,9406 | 0,6968 |
| Ripe | 0,5571 | 0,4757 |
| Underripe | 0,8275 | 0,7382 |
| Unripe | 0,9929 | 0,8766 |
| **mAP keseluruhan** | **0,8820** | **0,7309** |

*Precision* keseluruhan = 0,8745; *Recall* keseluruhan = 0,8653.

*Ripe* adalah kelas dengan performa terlemah pada B1, baik dari sisi AP50
maupun AP50-95, dengan selisih besar terhadap kelas lain. [TODO: Selidiki
dan laporkan penyebabnya begitu analisis *confusion matrix*/dataset
tersedia -- tidak ada penyebab pasti yang diklaim di sini.] Konsisten
dengan struktur kelas pada Subbab 2.1.2, dua kelas yang secara struktural
paling berbeda (*Abnormal*, *Empty Bunch*) serta tahap kematangan yang
paling tidak ambigu secara visual (*Unripe*) mencapai nilai AP50 tertinggi
(seluruhnya $\ge 0{,}9817$), sedangkan *Ripe* -- tahap tengah yang diapit
*Underripe* dan *Overripe* -- mencapai AP50 terendah (0,5571). *Underripe*
(0,8275), juga tahap tengah, menunjukkan nilai AP50 menengah yang secara
umum konsisten dengan pola ini, meski *Overripe* (0,9406) tidak sepenuhnya
sesuai dengan pola tersebut; penjelasan atas ketidaksesuaian ini
diserahkan pada analisis *confusion matrix* yang disebutkan di atas.

**Gambar 4.1** dan **4.1b** BUKAN grafik batang gambar-tangan: keduanya
diambil langsung dari `confusion_matrix.png` dan `PR_curve.png` yang
dihasilkan Ultralytics sendiri saat mengevaluasi B1 pada *split held-out
test* (jalankan `python scripts/42_generate_b1_test_plots.py --weights
runs/b1_centralized/train/weights/best.pt --data data/splits_v2/data.yaml`,
lihat Subbab 3.6/3.8) -- angka yang ditampilkan dijamin sama persis dengan
Tabel 4.1 karena berasal dari evaluasi yang sama, hanya dengan `plots=True`
dinyalakan. **Gambar 4.2**: 2-4 citra *held-out test* dengan kotak deteksi,
label kelas, dan skor keyakinan hasil prediksi B1 (`model.predict(source=...,
save=True)` pada `best.pt`), termasuk minimal satu contoh dari kelas Ripe --
juga keluaran langsung model, bukan ilustrasi buatan.

## 4.2 B2: Hasil Federasi Tanpa DP -- Tiga *Seed*

Tabel 4.2 melaporkan hasil *validation* B2 pada tiga *seed* pelatihan.
Karena B1 dilaporkan pada *split held-out test* sedangkan B2 pada *split
validation*, selisih performa B1-versus-B2 secara langsung tidak dihitung
di sini; perbandingan indikatif diberikan pada Subbab 4.3, dan perbandingan
*held-out test* yang tersandingkan penuh ditunda ke pelaporan berikutnya
(Subbab 3.8, 4.6).

Tabel 4.2. B2 (FedAvg, tanpa DP) -- hasil *validation* tiga *seed*

| Seed | Ronde terbaik | mAP50 | mAP50-95 | Precision / Recall |
|---|---|---|---|---|
| 42 | 9 | 0,8850 | 0,7431 | 0,8717 / 0,8616 |
| 123 | 11 | 0,8594 | 0,7145 | 0,8139 / 0,8422 |
| 2026 | 40 | 0,8880 | 0,7544 | 0,8868 / 0,9104 |
| **Rata-rata** | -- | **0,8775** | **0,7374** | 0,8574 / 0,8714 |
| **SD** | -- | **0,0157** | **0,0206** | 0,0385 / 0,0352 |

Ronde *checkpoint* terbaik-*validation* bervariasi cukup jauh antar *seed*:
ronde 9 untuk *seed* 42, ronde 11 untuk *seed* 123, dan ronde 40 (ronde
terakhir) untuk *seed* 2026. Ini mengindikasikan bahwa kecepatan konvergensi
di bawah partisi Non-IID ini sensitif terhadap trayektori stokastik
pelatihan, meski performa *validation* akhir tetap sebanding di ketiga
*seed* (rentang mAP50: 0,8594-0,8880).

## 4.3 Perbandingan Indikatif Tersentral-versus-Federasi

Meski B1 dan B2 dievaluasi pada *split* berbeda (*held-out test* versus
*validation*) dan karenanya tidak dapat dibandingkan secara langsung dalam
arti yang tersandingkan penuh, selisih antara mAP50 *held-out test* B1
(0,8820) dan mAP50 *validation* rata-rata tiga-*seed* B2 (0,8775, SD =
0,0157) tergolong kecil (0,0045), yang mengindikasikan performa operasional
yang sebanding antara konfigurasi tersentral dan federasi Non-IID di bawah
partisi yang diuji. Salah satu kemungkinan penyebab selisih kecil ini
adalah bahwa FedAvg membobotkan pembaruan lokal tiap klien berdasarkan
jumlah sampel lokalnya (Persamaan 2.3); karena Klien 2 sendiri menyumbang
59,36% data latih (Tabel 3.2), sinyal gradiennya yang didukung banyak
sampel lokal dapat mendominasi pembaruan teragregasi, sebagian mengimbangi
heterogenitas yang diperkenalkan oleh ketiga klien yang lebih kecil.
Perbandingan yang lebih ketat berbasis *held-out test* ditunda ke pelaporan
berikutnya, setelah *checkpoint* B2 dikunci dan dievaluasi di bawah
protokol *held-out* yang sama dengan B1.

**Gambar 4.3** (`docs/thesis/manuscript/figure_b2_convergence.png`) bukan
grafik batang tiga angka akhir Tabel 4.2, melainkan kurva konvergensi mAP50
*validation* sungguhan per ronde untuk ketiga *seed*, diplot langsung dari
`history.json` tiap *run* (dicatat setiap ronde oleh
`run_federated_training`, Subbab 3.7.2; ketiga berkas sumbernya disertakan
sebagai `docs/thesis/manuscript/b2_k4_seed{42,123,2026}_history.json`)
memakai `scripts/43_plot_b2_convergence.py`. Bentuk kurva ini -- bukan
sekadar tiga titik akhir -- yang menunjukkan mengapa ronde *checkpoint*
terbaik bisa berbeda jauh antar *seed* (9 / 11 / 40, lihat di atas): ketiga
*seed* naik cepat dan hampir berhimpit hingga ronde ~5, lalu berosilasi di
kisaran 0,81-0,89 tanpa satupun *seed* yang stabil monoton meningkat --
*seed* 42 dan 2026 kebetulan menyentuh puncak lokalnya masing-masing di
ronde 9 dan 40, sedangkan *seed* 123 secara konsisten berosilasi sedikit
lebih rendah sepanjang pelatihan. Sesuatu yang tidak terlihat dari
ringkasan batang tunggal per *seed*.

## 4.4 Diskusi

Hasil B1 dan B2 pada bab ini masing-masing menetapkan titik rujukan
tersentral dan titik rujukan federasi Non-IID (non-privat) untuk deteksi
kematangan TBS sawit enam kelas menggunakan YOLOv11n dengan GroupNorm.
Hasil *held-out test* B1 menunjukkan performa deteksi keseluruhan yang
kuat (mAP50 = 0,8820, mAP50-95 = 0,7309), dengan *Ripe* saat ini menjadi
kelas dengan performa terlemah dengan selisih besar (AP50 = 0,5571
dibanding $\ge 0{,}8275$ untuk seluruh kelas lain); penyebab pasti tidak
dapat diklaim tanpa analisis *confusion matrix* atau tingkat-dataset,
namun faktor yang mungkin berkontribusi antara lain transisi visual yang
bertahap masuk dan keluar dari tahap *Ripe*, serta kemungkinan tumpang
tindih visual dengan kelas tetangga *Underripe* dan *Overripe*, yang perlu
diselidiki langsung pada penelitian lanjutan. Hasil *validation* tiga-
*seed* B2 (rata-rata mAP50 = 0,8775, SD = 0,0157) mengindikasikan bahwa
FedAvg konvergen ke titik operasi yang stabil lintas realisasi stokastik
pelatihan independen di bawah partisi Non-IID ini (K = 4, Dirichlet
$\alpha = 0{,}5$), meskipun Klien 1 hanya memegang 8,67% data latih
berbanding porsi 59,36% Klien 2. Stabilitas ini konsisten dengan ekspektasi
bahwa GroupNorm mengurangi sensitivitas pelatihan federasi terhadap
statistik *batch* kecil dan heterogen per klien (Subbab 2.5).

### 4.4.1 Selisih Utilitas Tersentral-versus-Federasi

Perbandingan indikatif pada Subbab 4.3 menunjukkan selisih utilitas yang
kecil antara pelatihan tersentral dan federasi Non-IID pada dataset dan
partisi ini, meski perbandingannya belum tersandingkan penuh karena B1 dan
B2 saat ini dilaporkan pada *split* berbeda (*held-out test* versus
*validation*).

### 4.4.2 Implikasi bagi Perluasan *Differential Privacy* yang Direncanakan

Titik rujukan B1/B2 pada bab ini mengukur titik rujukan utilitas federasi
Non-IID sebelum penambahan *noise* privasi formal. Karena DP-SGD
menyuntikkan *noise* gradien per-sampel di atas proses optimisasi federasi,
penurunan utilitas yang akan teramati pada perluasan DP-SGD (konfigurasi
*Full* dan *Partial*, Subbab 3.9) semestinya ditafsirkan relatif terhadap
performa federasi non-privat B2 -- bukan semata-mata relatif terhadap
performa tersentral B1 -- karena federasi itu sendiri sudah memperkenalkan
sumber variasi utilitas yang independen dari *noise* privasi, di sini
teramati kecil. Secara khusus, karena ukuran sampel antar klien bervariasi
substansial (Tabel 3.2), laju *sampling* per klien yang dipakai *accountant*
privasi DP-SGD juga akan bervariasi per klien; klien yang lebih kecil
(Klien 0 dan Klien 1) diperkirakan mengakumulasi pembelanjaan privasi lebih
cepat dibanding klien yang lebih besar di bawah *noise multiplier* yang
sama, yang melatari pelaporan ε per klien -- bukan sebagai satu nilai
teragregasi tunggal -- pada hasil DP mendatang.

## 4.5 Keterbatasan

Beberapa keterbatasan perlu dipertimbangkan dalam menafsirkan hasil ini.
Pertama, pelatihan federasi disimulasikan secara sekuensial pada satu GPU;
penelitian ini karenanya mengevaluasi heterogenitas klien secara statistik
(Non-IID), bukan faktor sistem terdistribusi seperti latensi jaringan,
klien yang putus koneksi, atau kegagalan komunikasi. Kedua, dataset berasal
dari satu sumber, sehingga generalisasi ke perkebunan, kultivar, kamera,
atau kondisi pencahayaan lain belum dievaluasi langsung. Ketiga, B2
direplikasi memakai tiga *seed* pelatihan, yang mengurangi namun tidak
menghilangkan ketergantungan pada varians stokastik pelatihan; jumlah
replikasi yang lebih besar akan memberikan estimasi varians yang lebih
ketat. Keempat, keempat klien adalah pecahan data tersimulasi yang ditarik
dari satu *split train*, bukan representasi empat perkebunan, afdeling,
atau organisasi yang secara fisik berbeda; kesimpulan tentang *federated
learning* di sini terbatas pada heterogenitas statistik tersimulasi yang
ditangkap oleh partisi Dirichlet. Kelima, B2 saat ini hanya dilaporkan pada
*split validation*; evaluasi *held-out test* B2, dan karenanya perbandingan
B1-versus-B2 yang tersandingkan penuh, ditunda ke pelaporan berikutnya
setelah *checkpoint* yang bersangkutan dikunci.

## 4.6 Cakupan yang Ditunda pada Bab Ini

Sesuai batasan masalah (Subbab 1.4) dan cakupan metodologi (Subbab 3.9),
bab ini tidak melaporkan: (a) hasil DP-SGD *Full*/*Partial* (E1/E2) dan
kurva privasi-utilitas ε-versus-mAP; (b) evaluasi *Explainable AI*
(Grad-CAM++, *Average Drop*, *Focus Retention Rate*); (c) demonstrasi
operasional *deployment* layanan inferensi. Ketiganya direncanakan sebagai
Bab 4 pada laporan penelitian lanjutan yang dibangun di atas titik rujukan
B1/B2 di sini (lihat Subbab 5.3).
