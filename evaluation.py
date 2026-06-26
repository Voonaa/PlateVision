import os
import time
import cv2
import numpy as np
import pandas as pd
import imutils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import shutil
from PIL import Image

from utils import calculate_character_accuracy
from detector import detect_license_plate
from ocr import recognize_text

# Fungsi pembantu untuk meng-handle berkas evaluasi baru yang diunggah
def handle_uploaded_eval_files(files, current_gt_df):
    if not files:
        return current_gt_df
        
    images_dir = "images"
    os.makedirs(images_dir, exist_ok=True)
    
    new_rows = []
    # Ambil berkas yang sudah ada di tabel saat ini
    existing_files = set(current_gt_df["Nama Berkas"].tolist()) if not current_gt_df.empty else set()
    
    for f in files:
        file_path = f.name if hasattr(f, 'name') else str(f)
        filename = os.path.basename(file_path)
        
        # Salin ke folder images/
        dest_path = os.path.join(images_dir, filename)
        try:
            shutil.copy(file_path, dest_path)
            
            # Jika belum ada di tabel, tambahkan baris baru dengan default GT "B 1234 AB"
            if filename not in existing_files:
                new_rows.append({"Nama Berkas": filename, "Ground Truth": "B 1234 AB"})
                existing_files.add(filename)
        except Exception as e:
            print(f"Gagal menyalin file {filename}: {str(e)}")
            
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        updated_df = pd.concat([current_gt_df, df_new], ignore_index=True)
        return updated_df
        
    return current_gt_df

# Menghasilkan plot Confusion Matrix biner
def generate_confusion_matrix_plot(tp, fn, fp, tn):
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Matriks kebingungan biner
    cm = np.array([[tp, fn], [fp, tn]])
    
    # Gambar heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Atur label
    classes = ['Tepat (Match)', 'Salah/Gagal']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix (Hasil OCR)',
           ylabel='Label Aktual',
           xlabel='Label Prediksi')
    
    # Tambahkan angka di dalam sel
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, weight='bold')
                    
    fig.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img

# Mengevaluasi dataset secara ilmiah
def evaluate_dataset(gt_table_df, d, sigma_color, sigma_space, canny_low, canny_high, approx_epsilon):
    images_dir = "images"
    if not os.path.exists(images_dir):
        return "Folder 'images' tidak ditemukan!", None, None
        
    if gt_table_df is None or gt_table_df.empty:
        return "Tabel Ground Truth kosong! Silakan tambahkan minimal satu gambar.", None, None
        
    results = []
    total_images = len(gt_table_df)
    
    # Metrik Evaluasi
    detected_count = 0
    exact_match_count = 0
    total_char_accuracy = 0.0
    total_processing_time = 0.0
    total_ocr_confidence = 0.0
    
    # Variabel untuk Confusion Matrix
    tp = 0  # True Positive: Plat terdeteksi DAN OCR benar 100%
    fp = 0  # False Positive: Plat terdeteksi tapi OCR salah
    fn = 0  # False Negative: Plat gagal terdeteksi ATAU teks OCR kosong/error
    tn = 0  # True Negative: Disimulasikan 0 karena semua gambar uji bertipe plat
    
    for index, row in gt_table_df.iterrows():
        filename = row["Nama Berkas"]
        gt_text = str(row["Ground Truth"]).strip()
        
        filepath = os.path.join(images_dir, filename)
        if not os.path.exists(filepath):
            results.append({
                "Nama File": filename,
                "Ground Truth": gt_text,
                "Hasil OCR": "FILE TIDAK DITEMUKAN",
                "Status Deteksi": "Gagal",
                "Akurasi Karakter (%)": "0.0%",
                "Waktu Proses (s)": "0.0s",
                "Keyakinan OCR (%)": "0.0%"
            })
            fn += 1
            continue
            
        img = cv2.imread(filepath)
        if img is None:
            results.append({
                "Nama File": filename,
                "Ground Truth": gt_text,
                "Hasil OCR": "GAGAL BACA GAMBAR",
                "Status Deteksi": "Gagal",
                "Akurasi Karakter (%)": "0.0%",
                "Waktu Proses (s)": "0.0s",
                "Keyakinan OCR (%)": "0.0%"
            })
            fn += 1
            continue
            
        # Hitung waktu proses segmentasi + OCR
        start_time = time.time()
        
        # Jalankan detektor
        det_result = detect_license_plate(
            img, d, sigma_color, sigma_space, canny_low, canny_high, approx_epsilon
        )
        
        cropped_plate = det_result["images"]["cropped_rgb"]
        status_det = det_result["metadata"]["detection_status"]
        
        # Jalankan OCR
        pred_text, ocr_conf = recognize_text(cropped_plate)
        
        processing_time = time.time() - start_time
        total_processing_time += processing_time
        total_ocr_confidence += ocr_conf
        
        is_detected_bool = det_result["success"] or (status_det == "FALLBACK_FULL_IMAGE" and pred_text != "TEKS TIDAK TERBACA" and pred_text != "PLAT KOSONG")
        
        if is_detected_bool:
            detected_count += 1
            status_text = "Sukses"
        else:
            status_text = "Gagal"
            
        # Hitung akurasi karakter Levenshtein
        char_acc = calculate_character_accuracy(pred_text, gt_text)
        total_char_accuracy += char_acc
        
        # Cek Exact Match (100% sama)
        pred_clean = "".join([c.upper() for c in pred_text if c.isalnum()])
        gt_clean = "".join([c.upper() for c in gt_text if c.isalnum()])
        
        if pred_clean == gt_clean and pred_clean != "":
            exact_match_count += 1
            tp += 1
        else:
            if is_detected_bool:
                fp += 1
            else:
                fn += 1
                
        results.append({
            "Nama File": filename,
            "Ground Truth": gt_text,
            "Hasil OCR": pred_text,
            "Status Deteksi": status_text,
            "Akurasi Karakter (%)": f"{char_acc}%",
            "Waktu Proses (s)": f"{processing_time:.3f}s",
            "Keyakinan OCR (%)": f"{round(ocr_conf * 100, 2)}%"
        })
        
    # Hitung Persentase Metrik
    detection_rate = (detected_count / total_images) * 100
    avg_char_acc = total_char_accuracy / total_images
    exact_match_rate = (exact_match_count / total_images) * 100
    avg_proc_time = total_processing_time / total_images
    avg_ocr_conf = (total_ocr_confidence / total_images) * 100
    
    # Hitung Precision, Recall, F1-Score
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    summary_html = f"""
    <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px;'>
        <div style='background-color: #1e293b; padding: 15px; border-radius: 12px; text-align: center; border-left: 5px solid #3b82f6; border: 1px solid #334155;'>
            <p style='margin: 0; color: #94a3b8; font-size: 13px; font-weight: bold;'>Detection Success Rate</p>
            <p style='margin: 5px 0 0 0; color: #3b82f6; font-size: 26px; font-weight: bold;'>{detection_rate:.2f}%</p>
            <p style='margin: 0; color: #64748b; font-size: 11px;'>({detected_count}/{total_images} Gambar Berhasil)</p>
        </div>
        <div style='background-color: #1e293b; padding: 15px; border-radius: 12px; text-align: center; border-left: 5px solid #10b981; border: 1px solid #334155;'>
            <p style='margin: 0; color: #94a3b8; font-size: 13px; font-weight: bold;'>Precision (Keandalan OCR)</p>
            <p style='margin: 5px 0 0 0; color: #10b981; font-size: 26px; font-weight: bold;'>{precision:.2f}%</p>
            <p style='margin: 0; color: #64748b; font-size: 11px;'>TP / (TP + FP)</p>
        </div>
        <div style='background-color: #1e293b; padding: 15px; border-radius: 12px; text-align: center; border-left: 5px solid #8b5cf6; border: 1px solid #334155;'>
            <p style='margin: 0; color: #94a3b8; font-size: 13px; font-weight: bold;'>Recall (Ketercakupan)</p>
            <p style='margin: 5px 0 0 0; color: #8b5cf6; font-size: 26px; font-weight: bold;'>{recall:.2f}%</p>
            <p style='margin: 0; color: #64748b; font-size: 11px;'>TP / (TP + FN)</p>
        </div>
        <div style='background-color: #1e293b; padding: 15px; border-radius: 12px; text-align: center; border-left: 5px solid #dd6b20; border: 1px solid #334155;'>
            <p style='margin: 0; color: #94a3b8; font-size: 13px; font-weight: bold;'>F1-Score (Harmonis)</p>
            <p style='margin: 5px 0 0 0; color: #dd6b20; font-size: 26px; font-weight: bold;'>{f1_score:.2f}%</p>
            <p style='margin: 0; color: #64748b; font-size: 11px;'>(Rata-rata Precision & Recall)</p>
        </div>
    </div>
    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 25px;'>
        <div style='background-color: #1e293b; padding: 12px; border-radius: 10px; text-align: center; border: 1px solid #334155;'>
            <p style='margin: 0; color: #94a3b8; font-size: 12px;'>Rata-rata Waktu Proses</p>
            <p style='margin: 3px 0 0 0; color: #f8fafc; font-size: 18px; font-weight: bold;'>{avg_proc_time:.3f} detik</p>
        </div>
        <div style='background-color: #1e293b; padding: 12px; border-radius: 10px; text-align: center; border: 1px solid #334155;'>
            <p style='margin: 0; color: #94a3b8; font-size: 12px;'>Rata-rata Keyakinan OCR</p>
            <p style='margin: 3px 0 0 0; color: #f8fafc; font-size: 18px; font-weight: bold;'>{avg_ocr_conf:.2f}%</p>
        </div>
        <div style='background-color: #1e293b; padding: 12px; border-radius: 10px; text-align: center; border: 1px solid #334155;'>
            <p style='margin: 0; color: #94a3b8; font-size: 12px;'>Rata-rata Akurasi Karakter</p>
            <p style='margin: 3px 0 0 0; color: #f8fafc; font-size: 18px; font-weight: bold;'>{avg_char_acc:.2f}%</p>
        </div>
    </div>
    """
    
    # Generate Confusion Matrix Heatmap
    cm_plot_img = generate_confusion_matrix_plot(tp, fn, fp, tn)
    
    df_results = pd.DataFrame(results)
    return summary_html, df_results, cm_plot_img
