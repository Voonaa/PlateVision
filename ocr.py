import easyocr
import numpy as np

_reader = None

def get_reader():
    global _reader
    if _reader is None:
        print("[INFO] Inisialisasi EasyOCR Model...")
        # Gunakan CPU untuk memastikan kompatibilitas universal
        _reader = easyocr.Reader(['en'], gpu=False)
        print("[INFO] Inisialisasi EasyOCR Berhasil.")
    return _reader

# Menjalankan OCR untuk membaca teks dari potongan plat nomor
def recognize_text(cropped_image):
    if cropped_image is None:
        return "PLAT KOSONG", 0.0
        
    reader = get_reader()
    try:
        ocr_result = reader.readtext(cropped_image)
        if len(ocr_result) > 0:
            # Menggabungkan teks jika ada beberapa baris atau potongan terpisah
            texts = [res[1] for res in ocr_result]
            confidences = [res[2] for res in ocr_result]
            
            ocr_text = " ".join(texts).upper()
            ocr_confidence = float(np.mean(confidences))
            return ocr_text, ocr_confidence
        else:
            return "TEKS TIDAK TERBACA", 0.0
    except Exception as e:
        return f"ERROR OCR: {str(e)}", 0.0
