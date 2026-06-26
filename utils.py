import os
import cv2
import numpy as np

# Algoritma Levenshtein Distance untuk menghitung jarak antar string
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

# Menghitung akurasi karakter berdasarkan ground truth
def calculate_character_accuracy(pred, gt):
    if not pred or not gt:
        return 0.0
    pred_clean = "".join([c.upper() for c in pred if c.isalnum()])
    gt_clean = "".join([c.upper() for c in gt if c.isalnum()])
    
    if not gt_clean:
        return 100.0 if not pred_clean else 0.0
        
    distance = levenshtein_distance(pred_clean, gt_clean)
    max_len = max(len(pred_clean), len(gt_clean))
    
    accuracy = ((max_len - distance) / max_len) * 100.0
    return round(accuracy, 2)

# Mengukur kualitas citra secara otomatis (metadata & metrik citra)
def estimate_image_metrics(image, filepath=None):
    if image is None:
        return {}
        
    h, w = image.shape[:2]
    
    # Hitung ukuran file dalam KB
    file_size_kb = 0.0
    filename = "Unggahan Langsung"
    if filepath and os.path.exists(filepath):
        file_size_kb = round(os.path.getsize(filepath) / 1024, 2)
        filename = os.path.basename(filepath)
    else:
        # Perkiraan kasar dari ukuran numpy array jika tidak ada path berkas fisik
        file_size_kb = round((image.nbytes) / 1024, 2)
        
    # Konversi ke Grayscale untuk pemrosesan statistik
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        
    # Kecerahan (Mean)
    brightness = float(np.mean(gray))
    
    # Kontras (Std Dev)
    contrast = float(np.std(gray))
    
    # Ketajaman (Variance of Laplacian)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # Estimasi Tingkat Kebisingan (Noise Level)
    # Selisih antara gambar asli dengan median blur adalah estimasi kasar noise
    median_blur = cv2.medianBlur(gray, 3)
    noise_diff = cv2.absdiff(gray, median_blur)
    noise = float(np.std(noise_diff))
    
    return {
        "filename": filename,
        "resolution": f"{w}x{h} px",
        "filesize": f"{file_size_kb} KB",
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "noise": round(noise, 2)
    }
