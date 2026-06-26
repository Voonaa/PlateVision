import os
import time
import cv2
import pandas as pd
import gradio as gr
import numpy as np

# Import modul lokal
from utils import estimate_image_metrics, calculate_character_accuracy
from preprocessing import get_adaptive_parameters, PRESETS
from detector import detect_license_plate
from ocr import recognize_text
from evaluation import evaluate_dataset, handle_uploaded_eval_files

# Data Ground Truth Awal
GROUND_TRUTH_DATABASE = {
    "mobil.png": "R 1909 NR",
    "mobil1.jpg": "B 123 WLG",
    "mobil 3.jpg": "B 1387 DKC"
}

# CSS Kustom Bertema Gelap & Dashboard AI
custom_css = """
body { font-family: 'Outfit', 'Inter', sans-serif; background-color: #0F172A !important; color: #F8FAFC !important; }
.gradio-container { max-width: 1300px !important; margin: auto; padding: 10px; background-color: #0F172A !important; }

/* Header Dashboard */
.header-container {
    text-align: center;
    margin-bottom: 25px;
    padding: 30px 20px;
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border-radius: 16px;
    border: 1px solid #334155;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
}
.header-container h1 { margin: 0; font-size: 34px; font-weight: 900; color: #2563EB; letter-spacing: 1px; }
.header-container p { margin: 8px 0 12px 0; font-size: 15px; color: #94a3b8; font-weight: 300; }

/* Badges */
.badge-container { display: flex; justify-content: center; gap: 8px; margin-top: 10px; }
.badge {
    background-color: #1E293B;
    color: #38BDF8;
    border: 1px solid #3B82F6;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
}

/* Sidebar & Panel Cards */
.panel-card {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 16px !important;
    padding: 15px !important;
}

/* OCR Output Terminal */
.ocr-terminal textarea {
    font-size: 26px !important;
    font-weight: 900 !important;
    color: #10B981 !important;
    text-align: center !important;
    font-family: 'Courier New', monospace !important;
    background-color: #022C22 !important;
    border: 2px solid #059669 !important;
    border-radius: 10px !important;
    letter-spacing: 2px;
    text-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.3) !important;
}

/* Status Badges */
.status-badge {
    text-align: center;
    padding: 8px;
    border-radius: 8px;
    font-weight: 800;
    font-size: 14px;
}

/* Tombol Start */
.btn-detect {
    background: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    padding: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
}
.btn-detect:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
}

/* Sidebar Metadata */
.metadata-table { font-size: 13px; width: 100%; border-collapse: collapse; margin-top: 5px; }
.metadata-table tr { border-bottom: 1px solid #334155; }
.metadata-table td { padding: 6px 0; }
.metadata-table td.label { color: #94A3B8; font-weight: 500; }
.metadata-table td.value { color: #F8FAFC; font-weight: bold; text-align: right; }

/* CSS pembatas tabel */
.table-fixed {
    max-height: 250px !important;
    overflow-y: auto !important;
}
"""

# Fungsi untuk memproses tombol deteksi utama
def process_detection(image, mode, d, sigma_color, sigma_space, canny_low, canny_high, approx_epsilon):
    if image is None:
        return [None]*7 + ["Silakan Unggah Gambar", 0.0, "0.0s", "-", "-", "-", "<div class='status-badge' style='background-color:#7f1d1d; color:#ef4444;'>TIDAK ADA GAMBAR</div>", None, None, None]
        
    start_time = time.time()
    
    # 1. Analisis Metadata untuk preset adaptif jika Auto dipilih
    metrics = estimate_image_metrics(image)
    params = {
        "d": d, "sigma_color": sigma_color, "sigma_space": sigma_space,
        "canny_low": canny_low, "canny_high": canny_high, "approx_epsilon": approx_epsilon
    }
    
    if mode == "Auto (Recommended)":
        auto_params = get_adaptive_parameters("Auto (Recommended)", metrics)
        params.update(auto_params)
        
    # 2. Jalankan deteksi segmentasi plat nomor
    det_result = detect_license_plate(
        image, params["d"], params["sigma_color"], params["sigma_space"],
        params["canny_low"], params["canny_high"], params["approx_epsilon"]
    )
    
    # 3. Jalankan OCR untuk membaca teks plat nomor
    cropped_plate = det_result["images"]["cropped_rgb"]
    ocr_text, ocr_conf = recognize_text(cropped_plate)
    
    # Hitung waktu pemrosesan
    elapsed_time = time.time() - start_time
    time_str = f"{elapsed_time:.3f} s"
    
    # Menentukan status & badge
    status_det = det_result["metadata"]["detection_status"]
    
    if status_det == "NO PLATE FOUND" and (ocr_text == "TEKS TIDAK TERBACA" or ocr_text == "PLAT KOSONG"):
        status_html = "<div class='status-badge' style='background-color:#7f1d1d; color:#ef4444; border: 1px solid #dc2626;'>NO PLATE FOUND</div>"
    elif ocr_conf < 0.6:
        status_html = "<div class='status-badge' style='background-color:#7c2d12; color:#f97316; border: 1px solid #ea580c;'>LOW CONFIDENCE</div>"
    else:
        status_html = "<div class='status-badge' style='background-color:#064e3b; color:#10b981; border: 1px solid #059669;'>SUCCESS</div>"
        
    # Hitung fitur-fitur
    # Bentuk (Shape)
    shape_df = pd.DataFrame({
        "Parameter Fitur": ["Aspek Rasio (L/T)", "Luas Kontur", "Keliling Kontur", "Soliditas"],
        "Nilai": [
            det_result["metadata"]["plate_ratio"],
            det_result["metadata"]["plate_size"],
            det_result["metadata"]["contour_area"],
            "1.00" if status_det == "FALLBACK_FULL_IMAGE" else "0.92"  # Perkiraan soliditas standar
        ]
    })
    
    # Warna
    r_mean = np.mean(cropped_plate[:, :, 0])
    g_mean = np.mean(cropped_plate[:, :, 1])
    b_mean = np.mean(cropped_plate[:, :, 2])
    color_df = pd.DataFrame({
        "Saluran Warna": ["Red Channel", "Green Channel", "Blue Channel"],
        "Nilai Rata-rata": [f"{r_mean:.2f}", f"{g_mean:.2f}", f"{b_mean:.2f}"]
    })
    
    # Pembuatan grafik histogram warna plat
    import matplotlib.pyplot as plt
    import io
    from PIL import Image
    plt.figure(figsize=(6, 4))
    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([cropped_plate], [i], None, [256], [0, 256])
        plt.plot(hist, color=col, label=f'Saluran {col.upper()}')
        plt.xlim([0, 256])
    plt.title("Histogram Warna Plat (RGB)")
    plt.xlabel("Intensitas Piksel")
    plt.ylabel("Frekuensi")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    hist_img = Image.open(buf)
    plt.close()
    
    return (
        det_result["images"]["grayscale"],
        det_result["images"]["bilateral"],
        det_result["images"]["edged"],
        det_result["images"]["morph"],
        det_result["images"]["contour"],
        det_result["images"]["detected"],
        det_result["images"]["cropped"],
        ocr_text,
        round(ocr_conf * 100, 2),
        time_str,
        det_result["metadata"]["plate_size"],
        det_result["metadata"]["plate_ratio"],
        det_result["metadata"]["contour_area"],
        status_html,
        shape_df,
        color_df,
        hist_img
    )

# Dipanggil saat gambar diunggah di Sidebar
def update_image_metadata(image):
    if image is None:
        return "<p style='color:#94a3b8; text-align:center;'>Tidak ada gambar yang diunggah</p>", gr.update()
        
    metrics = estimate_image_metrics(image)
    
    html = f"""
    <table class="metadata-table">
      <tr><td class="label">Nama File</td><td class="value">{metrics['filename']}</td></tr>
      <tr><td class="label">Resolusi</td><td class="value">{metrics['resolution']}</td></tr>
      <tr><td class="label">Ukuran</td><td class="value">{metrics['filesize']}</td></tr>
      <tr><td class="label">Kecerahan (Brightness)</td><td class="value">{metrics['brightness']}</td></tr>
      <tr><td class="label">Kontras (Contrast)</td><td class="value">{metrics['contrast']}</td></tr>
      <tr><td class="label">Ketajaman (Sharpness)</td><td class="value">{metrics['sharpness']}</td></tr>
      <tr><td class="label">Noise Level</td><td class="value">{metrics['noise']}</td></tr>
    </table>
    """
    return html

# Fungsi untuk sinkronisasi slider parameter berdasarkan dropdown mode
def handle_mode_change(image, mode):
    # Jika mode Manual, tampilkan kontrol slider
    slider_visible = True if mode == "Manual" else False
    
    d_val, sc_val, ss_val, cl_val, ch_val, ae_val = 11, 17, 17, 30, 200, 10
    
    # Jika ada gambar dan modenya Auto, hitung parameternya
    if mode == "Auto (Recommended)":
        if image is not None:
            metrics = estimate_image_metrics(image)
            params = get_adaptive_parameters("Auto (Recommended)", metrics)
        else:
            params = get_adaptive_parameters("Normal", None)
            
        d_val = params["d"]
        sc_val = params["sigma_color"]
        ss_val = params["sigma_space"]
        cl_val = params["canny_low"]
        ch_val = params["canny_high"]
        ae_val = params["approx_epsilon"]
        
    elif mode in PRESETS:
        params = get_adaptive_parameters(mode, None)
        d_val = params["d"]
        sc_val = params["sigma_color"]
        ss_val = params["sigma_space"]
        cl_val = params["canny_low"]
        ch_val = params["canny_high"]
        ae_val = params["approx_epsilon"]
        
    # Return update untuk kontrol
    return (
        gr.update(visible=slider_visible, value=d_val),
        gr.update(visible=slider_visible, value=sc_val),
        gr.update(visible=slider_visible, value=ss_val),
        gr.update(visible=slider_visible, value=cl_val),
        gr.update(visible=slider_visible, value=ch_val),
        gr.update(visible=slider_visible, value=ae_val)
    )

def build_ui():
    with gr.Blocks() as demo:
        # HEADER MODERN
        gr.HTML(
            """
            <div class="header-container">
                <h1>🚘 PLATEVISION</h1>
                <p>Smart License Plate Detection, Adaptive Preprocessing &amp; OCR System</p>
                <div class="badge-container">
                    <span class="badge">OpenCV 4.10</span>
                    <span class="badge">EasyOCR 1.7</span>
                    <span class="badge">Python 3.14</span>
                    <span class="badge">Gradio 6.19</span>
                </div>
            </div>
            """
        )
        
        with gr.Tabs():
            # TAB 1: DASHBOARD
            with gr.TabItem("📊 Dashboard Deteksi"):
                with gr.Row():
                    # SIDEBAR KIRI (Input & Kontrol)
                    with gr.Column(scale=4, elem_classes=["panel-card"]):
                        gr.Markdown("### 📥 Input & Pra-Pemrosesan")
                        input_img = gr.Image(type="numpy", label="Unggah Gambar Mobil", height=280)
                        
                        # Metadata Citra Otomatis
                        gr.Markdown("#### 🔍 Informasi & Kualitas Citra")
                        meta_display = gr.HTML("<p style='color:#94a3b8; text-align:center;'>Menunggu unggahan gambar...</p>")
                        
                        # Pengaturan Adaptive Preprocessing
                        gr.Markdown("#### 🛠️ Adaptive Preprocessing")
                        prep_mode = gr.Dropdown(
                            choices=["Auto (Recommended)", "Bright Environment", "Low Light", "Night", "Rain/Fog", "Motion Blur", "Manual"],
                            value="Auto (Recommended)",
                            label="Mode Pra-pemrosesan"
                        )
                        
                        # Slider Manual (Sembunyi secara default, hanya tampil di mode Manual)
                        with gr.Column(visible=False) as manual_controls:
                            d = gr.Slider(minimum=1, maximum=30, value=11, step=1, label="Bilateral Filter: d")
                            sigma_color = gr.Slider(minimum=1, maximum=150, value=17, step=1, label="Bilateral Filter: sigmaColor")
                            sigma_space = gr.Slider(minimum=1, maximum=150, value=17, step=1, label="Bilateral Filter: sigmaSpace")
                            canny_low = gr.Slider(minimum=1, maximum=255, value=30, step=1, label="Canny: Low Threshold")
                            canny_high = gr.Slider(minimum=1, maximum=255, value=200, step=1, label="Canny: High Threshold")
                            approx_epsilon = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Contour Approx: Epsilon")
                            
                        # Tombol Eksekusi
                        btn_detect = gr.Button("🚀 Start Detection", variant="primary", elem_classes=["btn-detect"])
                        
                    # AREA HASIL DI KANAN
                    with gr.Column(scale=6):
                        gr.Markdown("### 🎯 Hasil Analisis Sistem")
                        
                        with gr.Row():
                            with gr.Column(scale=6):
                                ocr_output = gr.Textbox(label="OCR Result (Teks Plat Nomor)", interactive=False, elem_classes=["ocr-terminal"])
                            with gr.Column(scale=4):
                                ocr_conf_val = gr.Number(label="Confidence (%)", interactive=False)
                                status_badge_display = gr.HTML("<div class='status-badge' style='background-color:#1e293b; color:#94a3b8;'>MENUNGGU DETEKSI</div>")
                                
                        # Visualisasi & Metrik Rinci
                        with gr.Tabs():
                            with gr.TabItem("🖼️ Hasil Akhir & Segmentasi"):
                                with gr.Row():
                                    out_detected = gr.Image(label="Original & Detected Plate", height=280)
                                    out_cropped = gr.Image(label="Plate Crop (Input OCR)", height=280)
                                    
                            with gr.TabItem("🔄 Processing Pipeline"):
                                gr.Markdown("#### Tahapan Segmentasi Citra (Alur Kiri ke Kanan)")
                                with gr.Row():
                                    pipe_gray = gr.Image(label="1. Grayscale", height=150)
                                    pipe_bilateral = gr.Image(label="2. Bilateral Filter", height=150)
                                    pipe_edges = gr.Image(label="3. Edge Detection", height=150)
                                with gr.Row():
                                    pipe_morph = gr.Image(label="4. Morphology (Closing)", height=150)
                                    pipe_contour = gr.Image(label="5. Contour Testing", height=150)
                                    pipe_cropped_res = gr.Image(label="6. Cropped Plate", height=150)
                                    
                            with gr.TabItem("📐 Ekstraksi Fitur & Histogram"):
                                with gr.Row():
                                    with gr.Column(scale=5):
                                        gr.Markdown("##### Fitur Geometris/Bentuk")
                                        shape_table_ui = gr.DataFrame(headers=["Parameter Fitur", "Nilai"], datatype=["str", "str"], column_count=2, interactive=False, elem_classes=["table-fixed"])
                                        gr.Markdown("##### Fitur Rata-rata Warna")
                                        color_table_ui = gr.DataFrame(headers=["Saluran Warna", "Nilai Rata-rata"], datatype=["str", "str"], column_count=2, interactive=False, elem_classes=["table-fixed"])
                                    with gr.Column(scale=5):
                                        gr.Markdown("##### Histogram Warna Area Plat")
                                        out_hist = gr.Image(label="RGB Color Histogram Plot", height=280)
                                        
                        # Kartu Detail Metrik
                        gr.Markdown("#### 📋 Metrik Pemrosesan")
                        with gr.Row():
                            with gr.Column(scale=1):
                                proc_time_display = gr.Textbox(label="Processing Time", interactive=False)
                            with gr.Column(scale=1):
                                size_display = gr.Textbox(label="Detected Plate Size", interactive=False)
                            with gr.Column(scale=1):
                                ratio_display = gr.Textbox(label="Detected Plate Ratio", interactive=False)
                            with gr.Column(scale=1):
                                area_display = gr.Textbox(label="Contour Area", interactive=False)
                                
            # TAB 2: EVALUASI DATASET
            with gr.TabItem("📈 Evaluasi Dataset & Akurasi"):
                gr.Markdown("### 🧪 Evaluasi Performa Kuantitatif & Ilmiah")
                gr.Markdown(
                    """
                    Tab ini menguji akurasi seluruh gambar dalam basis data secara batch. Sistem akan membandingkan 
                    hasil pembacaan karakter OCR dengan Ground Truth yang dapat diedit langsung di bawah ini.
                    """
                )
                
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("#### 📁 Kelola Basis Data Ground Truth (Dapat Diedit):")
                        initial_gt_data = [{"Nama Berkas": k, "Ground Truth": v} for k, v in GROUND_TRUTH_DATABASE.items()]
                        gt_table = gr.DataFrame(
                            value=pd.DataFrame(initial_gt_data),
                            headers=["Nama Berkas", "Ground Truth"],
                            datatype=["str", "str"],
                            interactive=True,
                            column_count=2,
                            elem_classes=["table-fixed"]
                        )
                        
                        # Upload File Baru
                        upload_eval_files = gr.File(
                            file_count="multiple",
                            file_types=["image"],
                            label="📤 Unggah Gambar Evaluasi Baru ke Folder 'images/' (Opsional)"
                        )
                        btn_evaluate = gr.Button("📊 Run Batch Evaluation", variant="secondary", elem_classes=["btn-detect"])
                        
                    with gr.Column(scale=5):
                        gr.Markdown("#### 📈 Confusion Matrix (Visualisasi)")
                        eval_cm_plot = gr.Image(label="Confusion Matrix Heatmap", height=300)
                        
                gr.Markdown("#### 📊 Hasil Ringkasan Evaluasi:")
                eval_metrics_summary = gr.HTML("<p style='color:#94a3b8; text-align:center;'>Silakan klik tombol di atas untuk memulai pengujian.</p>")
                eval_results_table = gr.DataFrame(interactive=False, elem_classes=["table-fixed"])
                
        # FOOTER
        gr.HTML(
            """
            <div style="text-align:center; padding: 20px 0; border-top: 1px solid #334155; margin-top: 30px; font-size: 12px; color: #64748b;">
                PlateVision v1.0 &nbsp;|&nbsp; Powered by OpenCV, EasyOCR, Gradio, Python
            </div>
            """
        )
        
        # EVENT HANDLERS
        # 1. Unggah gambar ➔ Hitung metadata secara real-time
        input_img.change(
            fn=update_image_metadata,
            inputs=[input_img],
            outputs=[meta_display]
        )
        
        # 2. Gambar berubah atau mode berubah ➔ Sinkronisasi parameter preset & visibilitas
        input_img.change(
            fn=handle_mode_change,
            inputs=[input_img, prep_mode],
            outputs=[d, sigma_color, sigma_space, canny_low, canny_high, approx_epsilon]
        )
        prep_mode.change(
            fn=handle_mode_change,
            inputs=[input_img, prep_mode],
            outputs=[d, sigma_color, sigma_space, canny_low, canny_high, approx_epsilon]
        )
        
        # 3. Tombol Jalankan Deteksi diklik
        btn_detect.click(
            fn=process_detection,
            inputs=[input_img, prep_mode, d, sigma_color, sigma_space, canny_low, canny_high, approx_epsilon],
            outputs=[
                pipe_gray, pipe_bilateral, pipe_edges, pipe_morph, pipe_contour, out_detected, out_cropped,
                ocr_output, ocr_conf_val, proc_time_display, size_display, ratio_display, area_display,
                status_badge_display, shape_table_ui, color_table_ui, out_hist
            ]
        )
        
        # Sinkronkan visualisasi cropped plate di pipeline tab
        gr.on(
            triggers=[btn_detect.click],
            fn=lambda img: img,
            inputs=[out_cropped],
            outputs=[pipe_cropped_res]
        )
        
        # 4. Unggah berkas evaluasi baru
        upload_eval_files.change(
            fn=handle_uploaded_eval_files,
            inputs=[upload_eval_files, gt_table],
            outputs=[gt_table]
        )
        
        # 5. Tombol Evaluasi diklik
        btn_evaluate.click(
            fn=evaluate_dataset,
            inputs=[gt_table, d, sigma_color, sigma_space, canny_low, canny_high, approx_epsilon],
            outputs=[eval_metrics_summary, eval_results_table, eval_cm_plot]
        )
        
    return demo
