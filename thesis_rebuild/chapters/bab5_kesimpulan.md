# BAB 5 KESIMPULAN DAN SARAN

> **STATUS PENULISAN — SKELETON.** Beberapa angka final menunggu Bab 4
> tuntas; tandai `{TBD: ...}` diisi setelah `aggregate_results.py` selesai.

## 5.1 Kesimpulan

Penelitian ini merancang dan mengevaluasi sebuah kerangka kerja deteksi
kematangan tandan buah segar (TBS) kelapa sawit yang **memadukan
*Federated Learning*, *Differential Privacy* per-sampel, dan *Explainable AI*
ke dalam satu pipa yang dapat dijalankan**. Mengacu pada empat rumusan
hipotesis pada Bab 1, temuan utama dapat diringkas sebagai berikut.

**Pertama (H1 terkonfirmasi)**, *baseline* YOLOv11n-GN tersentral mencapai
mAP@0.5 = **0,787** pada himpunan validasi global anti-kebocoran (re-split
*bunch_id*-stratified), memenuhi syarat sebagai *upper bound* utilitas yang
sah. Konversi seluruh 81 lapisan BatchNorm menjadi GroupNorm — prasyarat
arsitektural agar gradien per-sampel terdefinisi (Bab 2.6) — tidak menurunkan
utilitas secara signifikan.

**Kedua (H2 `{TBD: terkonfirmasi/sebagian/tidak}`)**, DP-SGD per-sampel
melalui Opacus 1.5.4 menghasilkan trade-off privasi-utilitas yang
**`{TBD: gradual / cliff}`** pada rentang $\sigma \in [0{,}5; 3{,}0]$.
Titik *sweet spot* operasional ditemukan pada `{TBD: K=?, σ=?, ε≈?, mAP≈?}`,
yang masuk kategori `{TBD: acceptable/excellent}` (Bab 4.0).

**Ketiga (H2-K `{TBD: terkonfirmasi/tidak}`)**, arah pengaruh jumlah klien
$K$ pada DP-SGD per-sampel **berkebalikan** dengan intuisi DP-FedAvg
level-klien: $K$ besar memperburuk utilitas karena samples-per-klien yang
menyusut membuat *noise* mendominasi sinyal lokal sebelum agregasi global
sempat memperbaikinya. Pada $K=16$ (klien terkecil hanya 270 citra), model
`{TBD: collapse / masih trainable}`; pada $K=2$ (klien terbesar 7.938 citra),
utilitas tertinggi tercapai.

**Keempat (kontribusi metodologis utama)**, perbandingan lintas mekanisme
menunjukkan bahwa **letak penyuntikan *noise* DP lebih menentukan
keberhasilan daripada nilai $\varepsilon$ itu sendiri**: DP-FedAvg
level-klien gagal (*collapse*) pada seluruh $\varepsilon$ yang diuji,
sedangkan DP-SGD per-sampel berhasil mempertahankan utilitas berguna pada
$\varepsilon \le 8$. Diagnosa teoretis (Bab 2.4.4) — *noise* di gradien
ter-*dampen* oleh momentum SGD sementara *noise* di pembaruan agregat
bersifat permanen — terkonfirmasi secara empiris.

**Kelima (H3 `{TBD: terkonfirmasi/tidak}`)**, evaluasi kuantitatif
Grad-CAM++ via Average Drop dan FRR menunjukkan bahwa kualitas penjelasan
model DP-SGD `{TBD: tetap sebanding/menurun signifikan}` dibanding *baseline*
non-DP. Dengan kata lain, privasi formal **`{TBD: tidak merusak/sedikit
mempengaruhi}`** kemampuan model untuk menunjukkan *di mana* ia melihat
buah saat memprediksi kematangan.

## 5.2 Kontribusi Penelitian

Kontribusi penelitian ini dibagi menjadi empat lapis.

1. **Metodologis.** Demonstrasi empiris bahwa DP-SGD per-sampel pada
   YOLOv11n-GN dapat dilatih secara federasi tanpa *collapse*, sementara
   DP-FedAvg level-klien pada arsitektur yang sama gagal. Identifikasi tiga
   inkompatibilitas konkret Opacus×YOLO (SiLU *in-place*, signature *loss*
   vektor, Conv+BN *fusion*) beserta *patch* sederhana yang dapat
   di-reproduksi oleh peneliti lain.

2. **Empiris.** Karakterisasi trade-off privasi-utilitas pada object
   detection sawit di seluruh *grid* $5K \times 5\sigma = 25$ konfigurasi DP
   penuh + 25 DP parsial, dengan akuntansi PRV yang ketat ($\delta = 10^{-5}$).
   Pengamatan baru: **arah $K$ terbalik** pada DP-SGD per-sampel — pengaruh
   yang tidak dilaporkan secara eksplisit dalam literatur DP-FL berbasis
   level-klien.

3. **Domain.** Pipa siap-pakai untuk perkebunan: model terlatih
   (`best.pt`) plus blueprint deployment Docker CPU-only (`thesis_rebuild/deploy/`)
   yang memungkinkan deteksi 6-kelas kematangan TBS dijalankan di edge
   (mini-PC pabrik) atau VPS murah tanpa konflik dependensi.

4. **Reproducibility.** Seluruh kode, partisi Dirichlet ($K \in \{2,4,8,12,16\}$),
   serta hasil mentah (`runs_master.csv`) terbuka di repo, dengan skrip
   `aggregate_results.py` yang merekonstruksi seluruh tabel dan plot Bab 4
   dari hasil mentah.

## 5.3 Keterbatasan

Penelitian ini memiliki tiga keterbatasan utama yang harus disampaikan jujur.

**Pertama, sumber data tunggal.** Seluruh 10.814 citra berasal dari satu
versi dataset Roboflow (varietas dan kondisi pengambilan terbatas). Hasil
yang dilaporkan berlaku pada distribusi data tersebut; generalisasi ke
perkebunan lain (varietas Tenera vs Dura vs Pisifera, iklim Indonesia
Timur vs Sumatera, kamera CCTV vs drone) belum diuji.

**Kedua, simulasi FL bukan deployment riil.** Privasi dijamin oleh mekanisme
DP-SGD per-sampel yang sama secara matematis dengan deployment terdistribusi;
namun aspek operasional FL nyata (latensi jaringan, *drop-out* klien,
kegagalan parsial agregasi, otentikasi server) tidak disimulasikan.

**Ketiga, $\delta$ tetap.** Pelaporan $\delta = 10^{-5}$ memudahkan
perbandingan tetapi tidak menyesuaikan skala $1/n_k$ per-klien. Analisis
sensitivitas terhadap pilihan $\delta$ dapat memperketat klaim privasi.

## 5.4 Saran Penelitian Lanjut

Berdasarkan temuan dan keterbatasan di atas, lima arah lanjut diajukan.

1. **Multi-source dataset.** Validasi pada dataset perkebunan riil lintas
   varietas dan lokasi untuk menguji generalisasi.

2. **Adaptive DP per klien.** Kalibrasi $\sigma$ atau $C$ secara individual
   per klien berdasarkan $n_k$, sehingga klien kecil yang "membayar"
   privasi lebih mahal (lihat Bab 4.3.3) tidak terpaksa mengorbankan
   utilitas global. Pendekatan seperti *per-client noise scaling* layak
   dieksplorasi.

3. **DP-SGD adaptif (DP-FTRL atau Differentially Private Adam).** Optimizer
   yang lebih canggih dari SGD murni dapat memperbaiki utilitas pada
   $\varepsilon$ rendah; perlu uji apakah keunggulan empiris tetap pada
   YOLOv11n-GN.

4. **Defense against gradient leakage in practice.** Walaupun DP-SGD
   memberi jaminan teoretis, demonstrasi serangan rekonstruksi gradien
   nyata (DLG/iDLG) pada model B2 versus E1/E2 akan memperkuat narasi
   praktis bahwa privasi formal benar-benar menutup celah serangan ini.

5. **Deployment edge nyata + benchmark inferensi.** Pengukuran latensi
   dan akurasi pada *hardware* edge nyata (mis. Jetson Nano, Raspberry Pi 5)
   dari image Docker yang sudah disiapkan; menentukan apakah *quantization*
   pasca-pelatihan (INT8) menjaga utilitas.
