# TEMPLATE PRESENTASI DOSEN: PROYEK "PLATEVISION"
*Panduan Slide, Skenario Lisan Kelompok, dan Simulasi Tanya Jawab (Q&A) Ujian Akhir Semester Pengolahan Citra Digital*

---

## 🖥️ SLIDE 1: JUDUL UTAMA & IDENTITAS KELOMPOK
### Tampilan Visual Slide:
*   **Judul Utama (Besar)**: PLATEVISION
*   **Subtitle**: Smart License Plate Detection, Adaptive Preprocessing & OCR System
*   **Identitas**: Logo Universitas/Fakultas, Mata Kuliah: Pengolahan Citra Digital (PCD)
*   **Dosen Pengampu**: [Nama Dosen Pengampu, Gelar]
*   **Daftar Anggota Kelompok**:
    1.  [Nama Anggota 1] - [NIM 1] *(Fokus: Segmentasi & OpenCV)*
    2.  [Nama Anggota 2] - [NIM 2] *(Fokus: Tata Letak UI & CSS)*
    3.  [Nama Anggota 3] - [NIM 3] *(Fokus: OCR & Evaluasi Performa)*

### 🗣️ Narasi Lisan (Anggota 1 - Pembuka):
> *"Selamat pagi/siang Bapak/Ibu Dosen penguji. Kami dari kelompok [Nama Kelompok] hari ini akan mempresentasikan proyek akhir mata kuliah Pengolahan Citra Digital kami yang berjudul **PlateVision**. Proyek ini merupakan sistem deteksi pelat nomor kendaraan pintar yang mengintegrasikan pemrosesan citra tingkat lanjut menggunakan OpenCV, sistem pra-pemrosesan adaptif otomatis, pengenalan karakter berbasis deep learning EasyOCR, serta dashboard evaluasi performa ilmiah berbasis web."*

---

## 🖥️ SLIDE 2: LATAR BELAKANG & RUMUSAN MASALAH
### Tampilan Visual Slide:
*   **Poin Masalah Utama (Real-World Challenges)**:
    *   Variasi kondisi lingkungan: Cahaya redup (malam hari), silau matahari (bright), kabur akibat gerakan (motion blur), dan noise sensor (hujan/kabut).
    *   Keterbatasan metode konvensional: Nilai filter (*kernel size*) dan batas ambang Canny (*thresholds*) yang kaku (*hardcoded*) akan gagal jika kondisi masukan gambar berubah.
*   **Solusi PlateVision**:
    *   Menerapkan *image quality assessment* secara real-time.
    *   Menghadirkan pre-processing adaptif yang menyesuaikan parameter secara otomatis demi mengoptimalkan hasil deteksi tepi.

### 🗣️ Narasi Lisan (Anggota 1):
> *"Mengapa sistem License Plate Recognition konvensional sering gagal di lapangan? Karena parameter pemrosesan citra biasanya diatur secara statis. Nilai threshold yang bekerja baik di siang hari akan gagal total di malam hari atau ketika kamera bergoyang menghasilkan blur. PlateVision memecahkan masalah ini dengan menganalisis statistik citra secara instan begitu gambar diunggah, lalu menyesuaikan filter secara adaptif agar segmentasi pelat tetap presisi."*

---

## 🖥️ SLIDE 3: ARSITEKTUR KODE MODULAR (CLEAN CODE)
### Tampilan Visual Slide:
*   **Diagram Pemisahan Kode**:
    *   `main.py` ➔ *Entry Point* Server & Launching Gradio.
    *   `ui.py` ➔ Tata letak interface, styling CSS kustom & event handling.
    *   `utils.py` ➔ Perhitungan parameter kualitas citra & jarak Levenshtein.
    *   `preprocessing.py` ➔ Manajemen preset parameter PCD adaptif.
    *   `detector.py` ➔ Implementasi pipeline pemrosesan citra OpenCV.
    *   `ocr.py` ➔ Model deep learning EasyOCR (CNN-LSTM).
    *   `evaluation.py` ➔ Penghitungan metrik akurasi kuantitatif & visualisasi grafik.

### 🗣️ Narasi Lisan (Anggota 1 atau 2):
> *"Kami membangun aplikasi ini dengan standar software engineering yang baik menggunakan paradigma pemrograman modular. Kode program dipecah ke dalam 7 modul berkas terpisah untuk memisahkan logika antarmuka, pengolahan citra, OCR, dan evaluasi. Hal ini membuat arsitektur aplikasi menjadi sangat bersih (*clean code*), mudah dipelihara, didebug, serta dikembangkan lebih lanjut dibanding menulis semua kode dalam satu berkas tunggal."*

---

## 🖥️ SLIDE 4: PIPELINE PENGOLAHAN CITRA DIGITAL OPENCV
### Tampilan Visual Slide:
*   Alur Proses Gambar (Tampilkan screenshot tab **Processing Pipeline**):
    1.  **Resizing & Grayscale**: Standardisasi lebar 600px dan eliminasi 3-channel warna.
    2.  **Bilateral Filter**: Mereduksi noise tanpa merusak ketajaman garis tepi (*edge-preserving*).
    3.  **Canny Edge**: Menghasilkan garis tepi tipis dari pelat dan karakter.
    4.  **Morphological Closing**: Kernel horizontal persegi panjang $17 \times 3$ untuk menggabungkan karakter-karakter terpisah menjadi satu blok kontur solid.
    5.  **Aproksimasi Poligon**: Deteksi kontur bersudut 4 (`approxPolyDP`) dengan aspek rasio horizontal antara 1.5 - 6.0.

### 🗣️ Narasi Lisan (Anggota 2 - Demonstrator):
> *"Mari kita tinjau pipeline pemrosesan citranya. Citra masukan dikonversi ke grayscale untuk menyederhanakan data warna. Kami menggunakan **Bilateral Filter** karena filter ini sangat efektif menghaluskan noise di area pelat namun tetap menjaga ketajaman garis tepi karakter. Setelah garis tepi diisolasi dengan **Canny Edge**, kami menerapkan **Morphological Closing menggunakan kernel horizontal persegi panjang berukuran 17x3**. Operasi closing ini sangat krusial karena berfungsi melekatkan sela-sela antar karakter pelat nomor sehingga menyatu menjadi satu blok kontur putih solid yang mudah dikenali oleh fungsi pencarian kontur segi empat."*

---

## 🖥️ SLIDE 5: ADAPTIVE PREPROCESSING (AUTO MODE)
### Tampilan Visual Slide:
*   **Kriteria Penilaian Kualitas Citra (Sidebar Kiri)**:
    *   *Brightness*: Rata-rata intensitas abu-abu.
    *   *Contrast*: Deviasi standar piksel.
    *   *Sharpness*: Variansi dari operator Laplace (Variance of Laplacian).
    *   *Noise*: Selisih absolut median citra.
*   **Skema Penyesuaian Preset**:
    *   Kecerahan $< 65$ ➔ Aktifkan preset **Low Light** (memperlebar ambang Canny agar tepi lemah tetap terdeteksi).
    *   Ketajaman $< 90$ ➔ Aktifkan preset **Motion Blur** (mempersepat ukuran kernel agar tidak merusak informasi tepi).
    *   Noise $> 32$ ➔ Aktifkan preset **Rain/Fog** (memperkuat diameter bilateral filter).

### 🗣️ Narasi Lisan (Anggota 2 - Demonstrator):
> *"Di sinilah letak kecerdasan PlateVision. Begitu gambar diunggah, aplikasi langsung mengukur kecerahan, kontras, ketajaman, dan noise citra secara real-time. Jika pengguna memilih mode **Auto**, nilai-nilai statistik ini akan memicu penyesuaian parameter OpenCV secara dinamis. Sebagai contoh, jika gambar masukan dinilai gelap (kecerahan di bawah 65), sistem otomatis berganti ke parameter 'Low Light' dengan melonggarkan ambang deteksi Canny agar karakter pelat yang redup tetap tervisualisasi."*

---

## 🖥️ SLIDE 6: OPTICAL CHARACTER RECOGNITION (EASYOCR) & FALLBACK SYSTEM
### Tampilan Visual Slide:
*   **Arsitektur OCR**: Menggunakan EasyOCR berbasis Convolutional Neural Network (CNN) untuk ekstraksi fitur citra teks, Recurrent Neural Network (LSTM) untuk analisis sekuens karakter, dan CTC untuk pembacaan teks akhir.
*   **Sistem Fallback Cerdas**:
    *   *Masalah*: Jika gambar yang diunggah berupa potongan pelat nomor jarak dekat, garis batas pelat akan bersentuhan dengan margin gambar, sehingga OpenCV gagal membentuk kontur tertutup bersudut 4.
    *   *Solusi*: Sistem secara otomatis melakukan fallback dengan mengirimkan citra grayscale utuh langsung ke EasyOCR tanpa melalui pemotongan kontur.

### 🗣️ Narasi Lisan (Anggota 2 atau 3):
> *"Setelah daerah pelat berhasil dilokalisasi, citra hasil potongan dikirim ke modul **EasyOCR**. Namun, kami juga mengantisipasi kegagalan deteksi kontur. Jika pengguna mengunggah gambar yang diambil terlalu dekat sehingga batas pelat menyentuh bingkai luar foto, fungsi kontur tidak akan menemukan poligon tertutup bersudut 4. Untuk mencegah kegagalan sistem, PlateVision mengaktifkan **Fallback System** dengan mengirimkan gambar skala abu-abu utuh langsung ke modul pembacaan karakter. Hasilnya, teks pelat nomor tetap dapat terbaca dengan sukses."*

---

## 🖥️ SLIDE 7: METODOLOGI EVALUASI ILMIAH & BATCH TESTING
### Tampilan Visual Slide:
*   **Metrik Evaluasi**:
    *   **True Positive (TP)**: Pelat terdeteksi dan dibaca OCR 100% tepat sesuai Ground Truth.
    *   **False Positive (FP)**: Pelat terdeteksi tapi karakter OCR salah atau tidak cocok.
    *   **False Negative (FN)**: Pelat gagal terdeteksi koordinatnya atau teks OCR kosong.
    *   **Formulasi Metrik**:
        $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
*   **Tabel Ground Truth & Confusion Matrix**: Heatmap visual 2x2 yang memetakan performa prediksi sistem secara keseluruhan.

### 🗣️ Narasi Lisan (Anggota 3 - Evaluator):
> *"Untuk memvalidasi keandalan sistem kami tidak hanya mengandalkan satu gambar uji, melainkan merancang sistem evaluasi batch secara ilmiah. Di tab evaluasi, kami menyediakan tabel Ground Truth yang dapat diedit langsung oleh pengguna di web. Sistem kemudian menghitung metrik standard klasifikasi biner, yaitu Precision, Recall, dan F1-Score. Hasil pengujian divisualisasikan dalam bentuk **Confusion Matrix Heatmap** menggunakan Matplotlib, memetakan secara jelas perbandingan antara kelas prediksi sistem dengan kelas aktual."*

---

## 🖥️ SLIDE 8: HASIL DEMONSTRASI & ANALISIS KASUS CITRA
### Tampilan Visual Slide:
*   Tabel Perbandingan Hasil Uji Dataset:
    *   `mobil1.jpg` ➔ Terdeteksi 100%, OCR Terbaca: `"B 123 WLG"` (Sempurna - TP).
    *   `mobil 3.jpg` ➔ Terdeteksi 100%, OCR Terbaca: `"B 1387 DKC"` (Sempurna - TP).
    *   `mobil.png` ➔ Terdeteksi 100%, OCR Terbaca: `"R 1909 NR"` (Sempurna - TP).
*   **Analisis Kualitas**: Keberhasilan lokalisasi pelat OpenCV mencapai 100% pada dataset uji, dan pembacaan karakter EasyOCR sangat dipengaruhi oleh kerapatan resolusi piksel dari gambar asal.

### 🗣️ Narasi Lisan (Anggota 3 - Evaluator):
> *"Berdasarkan hasil pengujian terhadap dataset gambar di direktori, sistem kami sukses melokalisasi koordinat pelat nomor pada seluruh gambar uji (*Plate Localization Rate* 100%). Karakter pada gambar `mobil1.jpg` dan `mobil 3.jpg` terbaca secara akurat. Kami menemukan bahwa tingkat keberhasilan OCR sangat berkorelasi dengan resolusi citra masukan. Semakin tinggi kerapatan piksel pada pelat yang terpotong, semakin tinggi tingkat keyakinan (*confidence score*) yang dihasilkan oleh mesin OCR."*

---

## 🖥️ SLIDE 9: SOLUSI DESAIN ANTARMUKA PREMIUM (ANTI-VIBRATION)
### Tampilan Visual Slide:
*   **Permasalahan Cumulative Layout Shift (CLS)**: Elemen halaman web naik-turun dan bergetar tidak stabil ketika tombol deteksi diklik akibat perubahan tinggi dinamis blok gambar.
*   **Solusi UX Premium**:
    *   Mengunci dimensi tinggi gambar input (`height=280`) dan gambar pipeline (`height=150`).
    *   Memisahkan visualisasi proses OpenCV ke dalam sub-tabs, sehingga struktur halaman web tetap kokoh (*solid layout*).
    *   Membatasi tinggi tabel Ground Truth menggunakan kustom CSS `overflow-y: auto`.

### 🗣️ Narasi Lisan (Anggota 2 - Designer):
> *"Kami juga sangat memperhatikan aspek kenyamanan pengguna atau User Experience. Pada tahap awal pengembangan, antarmuka halaman web sering bergetar dan melompat kasar saat tombol deteksi ditekan karena elemen gambar berubah ukuran dari 0 piksel menjadi tinggi penuh. Kami mengatasi masalah visual jitter ini dengan menetapkan tinggi elemen secara statis pada CSS kustom dan menyembunyikan proses intermediet di dalam sub-tabs. Hasilnya, transisi antarmuka web menjadi sangat halus dan elegan."*

---

## 🖥️ SLIDE 10: KESIMPULAN & PENUTUP
### Tampilan Visual Slide:
*   **Poin Kesimpulan**:
    1.  Metode preprocessing adaptif terbukti sukses menjaga stabilitas segmentasi tepi di berbagai rentang kualitas citra.
    2.  Operasi morfologi Closing horizontal $17 \times 3$ sangat ideal untuk menggabungkan kontur karakter pelat nomor.
    3.  Tingkat akurasi pembacaan karakter akhir sangat bergantung pada kejelasan piksel gambar masukan.
*   *Terima Kasih - Sesi Tanya Jawab Dibuka*

### 🗣️ Narasi Lisan (Anggota 3 - Penutup):
> *"Sebagai kesimpulan, proyek PlateVision berhasil membuktikan bahwa kombinasi pemrosesan citra konvensional (OpenCV) untuk melokalisasi pelat nomor dengan Deep Learning (EasyOCR) untuk mengekstraksi teks adalah solusi yang sangat optimal. Metode preprocessing adaptif kami meminimalkan intervensi manual oleh pengguna. Sekian presentasi dari kelompok kami. Terima kasih atas perhatiannya, dan kami siap untuk sesi tanya jawab bersama Bapak/Ibu Dosen."*

---

## 🙋‍♂️ PANDUAN SIMULASI TANYA JAWAB (Q&A) DENGAN DOSEN PENGUJI

Berikut adalah 5 pertanyaan kritis yang sering ditanyakan oleh dosen pengampu mata kuliah Pengolahan Citra Digital saat sidang proyek beserta strategi jawaban ilmiahnya:

### ❓ Pertanyaan 1: "Mengapa kalian menggunakan kernel Morfologi Closing ukuran $17 \times 3$ berbentuk persegi panjang horizontal? Kenapa tidak kernel persegi $5 \times 5$?"
*   **Jawaban Taktis**:
    > *"Pelat nomor kendaraan memiliki pola susunan karakter huruf dan angka yang memanjang secara horizontal dengan jarak antar-karakter yang cukup rapat. Jika kita menggunakan kernel persegi seperti $5 \times 5$, efek dilasi dan erosi akan melebar secara merata ke arah atas dan bawah, sehingga dapat menggabungkan pelat nomor dengan noise di sekitarnya seperti bemper mobil. Dengan menggunakan kernel persegi panjang horizontal berukuran $17 \times 3$, kita memperlebar pencarian kontur ke samping untuk menjembatani spasi antar huruf pelat nomor saja tanpa memperbesar tinggi objek secara berlebihan, sehingga menghasilkan satu blok kontur panjang yang tepat merepresentasikan bentuk fisik pelat nomor."*

### ❓ Pertanyaan 2: "Apa keunggulan Bilateral Filter dibandingkan Gaussian Blur biasa pada kasus deteksi pelat nomor?"
*   **Jawaban Taktis**:
    > *"Gaussian Blur bekerja dengan menghitung rata-rata tertimbang dari intensitas piksel tetangga hanya berdasarkan jarak spasial. Akibatnya, filter Gaussian akan mengaburkan garis tepi karakter pelat nomor yang tajam. Sebaliknya, Bilateral Filter bekerja dengan mempertimbangkan dua domain sekaligus: jarak spasial dan kemiripan intensitas piksel (*radiometric domain*). Filter ini hanya akan mengaburkan area yang memiliki variasi warna halus (noise jalanan atau body mobil), tetapi akan menolak pengaburan pada area yang mengalami transisi intensitas warna yang tajam (seperti tepi huruf pelat hitam di atas dasar putih). Oleh karena itu, tepi karakter pelat tetap tajam dan siap untuk proses deteksi Canny."*

### ❓ Pertanyaan 3: "Bagaimana cara kerja perhitungan Sharpness (Ketajaman) dengan rumus Variance of Laplacian?"
*   **Jawaban Taktis**:
    > *"Operator Laplace adalah operator diferensial orde kedua yang mengukur laju perubahan gradien citra secara spasial. Jika sebuah citra memiliki fokus yang tajam, citra tersebut akan memiliki perubahan warna yang mendadak pada garis-garis tepinya, yang menghasilkan nilai turunan kedua (Laplacian) yang sangat tinggi atau sangat rendah (variansi besar). Sebaliknya, jika citra buram (*blurry*), transisi warnanya sangat lambat dan merata, sehingga nilai Laplacian di seluruh citra akan mendekati seragam (variansi kecil). Dengan menghitung variansi dari hasil konvolusi kernel Laplacian pada citra, kita mendapatkan representasi kuantitatif ketajaman citra tersebut secara instan."*

### ❓ Pertanyaan 4: "Bagaimana sistem kalian menangani gambar mobil yang pelatnya tidak terdeteksi konturnya oleh OpenCV?"
*   **Jawaban Taktis**:
    > *"Kami merancang Fallback System di dalam modul `detector.py` dan `ui.py`. Ketika algoritma deteksi kontur OpenCV gagal menemukan poligon bersudut 4 (misalnya karena pelat terlalu dekat atau miring), sistem tidak langsung menghasilkan output error. Sebagai gantinya, citra masukan skala keabuan (*grayscale*) yang telah di-resize secara penuh akan langsung dialihkan ke EasyOCR. Model deep learning pada EasyOCR memiliki kemampuan lokalisasi teks terintegrasi berbasis deteksi fitur, sehingga teks pelat nomor pada gambar mobil secara keseluruhan masih berpeluang besar dibaca dengan baik meskipun OpenCV gagal mendeteksi koordinat kotaknya."*

### ❓ Pertanyaan 5: "Bagaimana cara kalian mengevaluasi sistem ini secara kuantitatif?"
*   **Jawaban Taktis**:
    > *"Kami mengevaluasi sistem dengan menerapkan pengujian batch pada database Ground Truth yang kami buat. Kami menghitung metrik standard klasifikasi yaitu Precision (mengukur seberapa andal hasil pembacaan benar dari seluruh deteksi yang dilakukan), Recall (mengukur seberapa banyak pelat yang berhasil dibaca benar dari total seluruh data uji yang ada), serta F1-Score sebagai rata-rata harmonis kedua metrik tersebut. Selain itu, kami menghitung akurasi karakter menggunakan Levenshtein Distance untuk mengetahui persentase kesalahan ketik huruf per huruf dari hasil pembacaan OCR terhadap Ground Truth."*
