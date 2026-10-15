import os
import math
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Proposal-aligned classes and MKJI-style passenger-car equivalents (smp).
CLASS_MAP = {
    0: ("motor", "Sepeda Motor (MC)", 0.25),
    2: ("mobil", "Mobil (LV)", 1.00),
    5: ("bus", "Bus (HV)", 1.30),
    7: ("truk", "Truk (HV)", 1.30),
}
APPROACHES = ["Raya Langsep", "Dieng", "Galunggung", "Terusan Dieng"]
MODEL_CACHE = {}

def load_model(weights):
    weights = (weights or "").strip() or "yolov8s.pt"
    if weights not in MODEL_CACHE:
        MODEL_CACHE[weights] = YOLO(weights)
    return MODEL_CACHE[weights]

def process_video(video_path, weights, conf, line_percent, frame_stride, approach_name):
    if not video_path:
        return None, pd.DataFrame(), "Video belum diunggah."
    model = load_model(weights)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Video tidak dapat dibuka. Coba unggah MP4/AVI yang valid.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    line_y = int(height * float(line_percent) / 100.0)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    tmp.close()
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    counts = {v[0]: 0 for v in CLASS_MAP.values()}
    seen_crossings = set()
    previous_y = {}
    processed = 0
    frame_index = 0
    # Built-in Ultralytics tracker (ByteTrack) assigns persistent IDs to avoid
    # counting the same vehicle repeatedly. This is not DeepSORT.
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % max(1, int(frame_stride)) != 0:
                frame_index += 1
                continue

            result = model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                conf=float(conf), imgsz=640, verbose=False
            )[0]
            annotated = result.plot()
            cv2.line(annotated, (0, line_y), (width - 1, line_y), (0, 255, 255), 2)
            cv2.putText(annotated, f"Garis hitung - {approach_name}", (15, max(25, line_y - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(xyxy)
                for box, cls_id, track_id in zip(xyxy, cls_ids, ids):
                    if cls_id not in CLASS_MAP:
                        continue
                    x1, y1, x2, y2 = box
                    cy = float((y1 + y2) / 2)
                    if track_id >= 0:
                        old_y = previous_y.get(track_id)
                        # Count once when the tracked centroid crosses the line in either direction.
                        if old_y is not None and ((old_y < line_y <= cy) or (old_y > line_y >= cy)):
                            key = (approach_name, int(track_id))
                            if key not in seen_crossings:
                                counts[CLASS_MAP[cls_id][0]] += 1
                                seen_crossings.add(key)
                        previous_y[track_id] = cy

            y_text = 28
            for _, (key, label, _) in CLASS_MAP.items():
                cv2.putText(annotated, f"{label}: {counts[key]}", (15, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (50, 255, 50), 2)
                y_text += 26
            writer.write(annotated)
            processed += 1
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    # Save only annotated video frames that were sampled; output FPS is adjusted so it plays near original duration.
    if processed and int(frame_stride) > 1:
        cap2 = cv2.VideoCapture(out_path)
        # Video output is intentionally kept at original FPS; frame skipping is a speed/quality trade-off.
        cap2.release()

    rows = []
    for _, (key, label, smp) in CLASS_MAP.items():
        rows.append({"Pendekat": approach_name, "Kelas": label, "Jumlah terhitung": counts[key],
                     "Koefisien smp": smp, "Total smp": round(counts[key] * smp, 2)})
    total_smp = sum(counts[k] * CLASS_MAP[next(i for i,v in CLASS_MAP.items() if v[0] == k)][2] for k in counts)
    duration_min = (total_frames / fps / 60.0) if fps > 0 else 0
    # Convert observed crossings to smp/hour; clearly marked as an estimate if clip is short.
    flow_smp_h = total_smp * 60.0 / max(duration_min, 1/60)
    summary = (
        f"**Pendekat:** {approach_name}  \n"
        f"**Frame diproses:** {processed:,} dari {total_frames:,}  \n"
        f"**Durasi video:** {duration_min:.2f} menit  \n"
        f"**Kendaraan terhitung melintasi garis:** {sum(counts.values())}  \n"
        f"**Arus estimasi:** {flow_smp_h:.1f} smp/jam  \n\n"
        "Catatan: penghitungan berbasis garis virtual hanya menghitung ID yang terdeteksi melintasi garis. "
        "Kualitas hasil dipengaruhi sudut CCTV, oklusi, confidence, dan model."
    )
    return out_path, pd.DataFrame(rows), summary

def objective(greens, flows, cycle, saturation):
    # Approximate delay objective for simulation: arrival flow versus green-proportional capacity.
    greens = np.asarray(greens, dtype=float)
    flows = np.asarray(flows, dtype=float)
    cycle = max(float(cycle), 4.0 * 3.0 + 1.0)
    effective_green = np.maximum(greens, 1.0)
    capacity = float(saturation) * effective_green / cycle
    degree = np.divide(flows, np.maximum(capacity, 1e-6))
    # Uniform-delay approximation plus a strong penalty for oversaturation.
    green_ratio = effective_green / cycle
    delay = (0.5 * cycle * np.square(1.0 - green_ratio)) / np.maximum(1.0 - np.minimum(degree, 0.99) * green_ratio, 0.05)
    overflow = np.maximum(degree - 0.85, 0.0)
    per_approach = delay + 90.0 * overflow**2
    return float(np.sum(flows * per_approach) / max(np.sum(flows), 1.0)), capacity, degree, delay

def run_sa(flow_values, cycle, saturation, t0, tfinal, alpha, iterations, gmin, gmax, seed):
    flows = np.array([max(0.0, float(x)) for x in flow_values], dtype=float)
    rng = np.random.default_rng(int(seed))
    n = len(flows)
    # Keep each phase green within bounds and ensure the total green budget fits the cycle.
    clearance = 3.0 * n
    if float(cycle) < n * float(gmin) + clearance:
        raise gr.Error(f"Waktu siklus minimal untuk {n} fase adalah {n * float(gmin) + clearance:.0f} detik (durasi hijau minimum + clearance 3 detik/fase).")
    green_budget = float(cycle) - clearance
    if green_budget > n * float(gmax) or green_budget < n * float(gmin):
        raise gr.Error("Jumlah durasi hijau minimum/maksimum tidak sesuai dengan waktu siklus.")
    x = rng.uniform(float(gmin), float(gmax), n)
    x = x / x.sum() * green_budget
    x = np.clip(x, float(gmin), float(gmax))
    # Rebalance while respecting bounds.
    for _ in range(100):
        diff = green_budget - x.sum()
        if abs(diff) < 1e-6: break
        free = np.where((x < gmax - 1e-6) if diff > 0 else (x > gmin + 1e-6))[0]
        if not len(free): break
        x[free] += diff / len(free)
        x = np.clip(x, gmin, gmax)

    current = x.copy()
    best = x.copy()
    current_e = objective(current, flows, cycle, saturation)[0]
    best_e = current_e
    history = [best_e]
    temp = float(t0)
    iters = int(iterations)
    while temp > float(tfinal):
        for _ in range(iters):
            candidate = current.copy()
            i = int(rng.integers(0, n))
            candidate[i] = np.clip(candidate[i] + rng.uniform(-5, 5), gmin, gmax)
            # Preserve total green budget by distributing the change among other phases.
            diff = current.sum() - candidate.sum()
            others = [j for j in range(n) if j != i]
            if others:
                candidate[others] += diff / len(others)
                candidate[others] = np.clip(candidate[others], gmin, gmax)
            new_e = objective(candidate, flows, cycle, saturation)[0]
            delta = new_e - current_e
            if delta <= 0 or rng.random() < math.exp(-min(delta / max(temp, 1e-12), 700)):
                current, current_e = candidate, new_e
            if current_e < best_e:
                best, best_e = current.copy(), current_e
            history.append(best_e)
        temp *= float(alpha)

    best = np.round(best, 1)
    energy, capacity, degree, delays = objective(best, flows, cycle, saturation)
    # Approximate queue and stopped ratio for comparative simulation, not certified PKJI output.
    queue = np.maximum(flows - capacity, 0) / 60.0
    stopped_ratio = np.clip(1.0 - best / max(float(cycle), 1.0), 0, 1)
    table = pd.DataFrame({
        "Pendekat": APPROACHES,
        "Arus input (smp/jam)": np.round(flows, 1),
        "Hijau optimal (detik)": best,
        "Kapasitas estimasi (smp/jam)": np.round(capacity, 1),
        "Derajat kejenuhan estimasi": np.round(degree, 3),
        "Tundaan estimasi (detik/kendaraan)": np.round(delays, 2),
        "Panjang antrean relatif (kendaraan)": np.round(queue, 2),
        "Rasio kendaraan terhenti estimasi": np.round(stopped_ratio, 3),
    })
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(APPROACHES, best)
    ax.set_ylabel("Durasi hijau (detik)")
    ax.set_title("Rekomendasi Durasi Hijau - Simulated Annealing")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    return table, fig, f"**Nilai fungsi objektif (estimasi tundaan berbobot):** {energy:.3f}  \n**Siklus:** {cycle:.0f} detik (termasuk clearance kuning 3 detik/fase)."

def optimize_from_inputs(raya, dieng, galunggung, terusan, cycle, saturation, t0, tfinal, alpha, iterations, gmin, gmax, seed):
    flows = [raya, dieng, galunggung, terusan]
    return run_sa(flows, cycle, saturation, t0, tfinal, alpha, iterations, gmin, gmax, seed)

CSS = """
.gradio-container {max-width: 1250px !important; margin: auto !important;}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald"), css=CSS, title="Sistem Lampu Lalu Lintas Adaptif YOLOv8s + SA") as demo:
    gr.Markdown("""
    # Sistem Lampu Lalu Lintas Adaptif Berbasis CCTV
    **YOLOv8s + pelacakan ID + Simulated Annealing (SA)**  
    Purwarupa penelitian untuk rekaman CCTV Simpang Empat Dieng. Sistem ini menghasilkan *rekomendasi simulasi*,
    bukan kendali langsung perangkat lampu lalu lintas.
    """)
    with gr.Tab("1. Deteksi & Hitung Kendaraan"):
        gr.Markdown("Unggah video CCTV untuk satu pendekat. Ulangi pada tab ini untuk setiap pendekat. Model default `yolov8s.pt` adalah bobot COCO pralatih; untuk hasil penelitian final, unggah bobot kendaraan `best.pt` yang sudah dilatih.")
        with gr.Row():
            video = gr.Video(label="Video CCTV (MP4/AVI)", sources=["upload"], format="mp4")
            with gr.Column():
                approach = gr.Dropdown(choices=APPROACHES, value=APPROACHES[0], label="Nama pendekat")
                weights = gr.Textbox(value="yolov8s.pt", label="Path bobot model atau nama bobot", info="Contoh: yolov8s.pt atau /path/best.pt")
                conf = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Confidence threshold")
                line_percent = gr.Slider(10, 90, value=55, step=1, label="Posisi garis hitung (% tinggi video)")
                frame_stride = gr.Slider(1, 5, value=1, step=1, label="Proses setiap N frame (1 = semua frame)")
                detect_btn = gr.Button("Jalankan deteksi dan penghitungan", variant="primary")
        annotated = gr.Video(label="Video dengan anotasi")
        with gr.Row():
            count_table = gr.Dataframe(label="Volume kendaraan terhitung", interactive=False)
            count_summary = gr.Markdown()
        detect_btn.click(process_video, [video, weights, conf, line_percent, frame_stride, approach],
                         [annotated, count_table, count_summary])

    with gr.Tab("2. Optimasi Durasi Lampu (SA)"):
        gr.Markdown("Masukkan arus tiap pendekat dalam **smp/jam**. Nilai dapat diambil dari hasil penghitungan video setelah dikonversi menurut koefisien smp pada proposal: motor 0,25; mobil 1,00; bus/truk 1,30. Karena satu video hanya mewakili satu pendekat, arus empat pendekat dimasukkan pada panel ini.")
        with gr.Row():
            flow_raya = gr.Number(value=600, label="Raya Langsep (smp/jam)", minimum=0)
            flow_dieng = gr.Number(value=360, label="Dieng (smp/jam)", minimum=0)
            flow_gal = gr.Number(value=450, label="Galunggung (smp/jam)", minimum=0)
            flow_terusan = gr.Number(value=300, label="Terusan Dieng (smp/jam)", minimum=0)
        with gr.Accordion("Parameter optimasi dan asumsi simulasi", open=True):
            with gr.Row():
                cycle = gr.Slider(90, 180, value=100, step=5, label="Waktu siklus total (detik)")
                saturation = gr.Number(value=1800, label="Arus jenuh S (smp/jam-hijau)", minimum=100)
                seed = gr.Number(value=42, label="Seed (reproducibility)", precision=0)
            with gr.Row():
                t0 = gr.Number(value=1000, label="Suhu awal T0")
                tfinal = gr.Number(value=0.01, label="Suhu akhir Tfinal")
                alpha = gr.Slider(0.80, 0.99, value=0.95, step=0.01, label="Cooling rate α")
                iterations = gr.Slider(10, 200, value=100, step=10, label="Iterasi per suhu L")
            with gr.Row():
                gmin = gr.Slider(5, 30, value=20, step=1, label="Hijau minimum (detik)")
                gmax = gr.Slider(30, 60, value=45, step=1, label="Hijau maksimum (detik)")
        opt_btn = gr.Button("Optimasi dengan Simulated Annealing", variant="primary")
        opt_summary = gr.Markdown()
        opt_table = gr.Dataframe(label="Hasil optimasi dan metrik estimasi", interactive=False)
        opt_plot = gr.Plot(label="Grafik durasi hijau")
        opt_btn.click(optimize_from_inputs,
                      [flow_raya, flow_dieng, flow_gal, flow_terusan, cycle, saturation,
                       t0, tfinal, alpha, iterations, gmin, gmax, seed],
                      [opt_table, opt_plot, opt_summary])

    with gr.Tab("3. Petunjuk & Batasan"):
        gr.Markdown("""
        ### Alur penggunaan
        1. Unggah video CCTV satu pendekat dan pilih nama pendekat.
        2. Sesuaikan posisi garis hitung agar berada di area kendaraan bergerak, bukan pada tepi gambar.
        3. Jalankan deteksi. Periksa video anotasi dan jumlah kendaraan.
        4. Ulangi untuk pendekat lain, lalu masukkan arus hasil konversi ke smp/jam pada tab optimasi.
        5. Jalankan SA dan simpan tabel/grafik hasil untuk analisis skripsi.

        ### Catatan metodologis penting
        - `yolov8s.pt` standar COCO digunakan untuk uji awal. Proposal menyebut kelas motor, mobil, bus, dan truk; untuk evaluasi skripsi, gunakan `best.pt` hasil pelatihan dataset yang telah dianotasi dan evaluasi precision, recall, F1, dan mAP.
        - Untuk kompatibilitas ruang komputasi Hugging Face, versi awal ini memakai tracker **ByteTrack bawaan Ultralytics**, bukan DeepSORT seperti rancangan proposal. Jangan melaporkan hasilnya sebagai implementasi DeepSORT.
        - Fungsi objektif dan metrik antrean/tundaan di aplikasi ini adalah **aproksimasi simulasi**, bukan perhitungan PKJI/MKJI yang tervalidasi penuh. Untuk hasil ilmiah final, lengkapi data geometrik, arus jenuh, waktu siklus, rumus PKJI yang tepat, dan validasi manual.
        - Video dihitung saat ID kendaraan melintasi garis virtual. Jika kendaraan bergerak bolak-balik melewati garis, ID yang sama hanya dihitung sekali per video.
        - Aplikasi ini hanya memberi rekomendasi simulasi. Tidak terhubung dengan pengendali lampu lalu lintas nyata.
        """)

if __name__ == "__main__":
    demo.queue().launch()
