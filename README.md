---
title: Sistem Lampu Lalu Lintas Adaptif YOLOv8s + SA
emoji: 🚦
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# YOLOv8s + Simulated Annealing untuk Lampu Lalu Lintas Adaptif

## Cara menjalankan di Hugging Face Spaces
1. Buat Space baru dengan SDK **Gradio**.
2. Unggah `app.py`, `requirements.txt`, dan `README.md` ke repositori Space.
3. Tunggu instalasi dependency selesai. Pertama kali menjalankan `yolov8s.pt`, Ultralytics akan mengunduh bobot COCO pralatih.
4. Untuk model penelitian yang sudah dilatih, unggah `best.pt` ke Space dan isi kolom bobot dengan path file tersebut (misalnya `/home/user/app/best.pt` setelah file berada pada path itu).

## Isi aplikasi
- Upload video CCTV per pendekat.
- Deteksi kelas kendaraan motor, mobil, bus, truk.
- Penghitungan berbasis ID tracking saat centroid melintasi garis virtual.
- Konversi jumlah terhitung ke smp menggunakan koefisien proposal: MC=0.25, LV=1.0, bus/truk=1.3.
- Optimasi pembagian durasi hijau dengan Simulated Annealing.
- Tabel dan grafik metrik simulasi.

## Batasan yang wajib dinyatakan dalam skripsi
Purwarupa ini menggunakan ByteTrack bawaan Ultralytics untuk tracking (bukan DeepSORT). Metrik tundaan, antrean, derajat kejenuhan dan rasio kendaraan terhenti yang ditampilkan merupakan aproksimasi simulasi, bukan keluaran PKJI/MKJI tervalidasi. Sistem hanya menghasilkan rekomendasi simulasi, bukan mengendalikan perangkat lampu nyata. Untuk evaluasi akademik, gunakan bobot YOLO hasil pelatihan dataset penelitian dan validasi metrik secara terpisah.
