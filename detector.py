import cv2
import numpy as np
import imutils

def detect_license_plate(image, d, sigma_color, sigma_space, canny_low, canny_high, approx_epsilon):
    if image is None:
        return None
        
    # 1. Resize gambar agar seragam (width=600) untuk pemrosesan piksel yang konsisten
    image_resized = imutils.resize(image, width=600)
    h_res, w_res = image_resized.shape[:2]
    
    # 2. Grayscale (Konversi warna RGB ke intensitas keabuan)
    gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
    
    # 3. Bilateral Filter (Edge-preserving smoothing untuk menghilangkan noise)
    bfilter = cv2.bilateralFilter(gray, int(d), float(sigma_color), float(sigma_space))
    
    # 4. Edge Detection (Deteksi tepi Canny)
    edged = cv2.Canny(bfilter, int(canny_low), int(canny_high))
    
    # 5. Morphology (Operasi Closing menggunakan kernel persegi panjang)
    # Ini membantu menutup kerenggangan antar karakter pelat sehingga terbentuk satu blok utuh
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morph = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, rect_kernel)
    
    # 6. Find Contours (Mendapatkan poligon terluar dari gambar morfologi)
    keypoints = cv2.findContours(morph.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    # Visualisasi top contours (Merah) pada gambar asli
    contour_image = image_resized.copy()
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)
    
    location = None
    # Cari kontur berbentuk segi empat (plat nomor)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, float(approx_epsilon), True)
        if len(approx) == 4:
            # Saring berdasarkan rasio aspek plat nomor standar (biasanya lebar > tinggi)
            x_b, y_b, w_b, h_b = cv2.boundingRect(contour)
            aspect_ratio = float(w_b) / h_b if h_b > 0 else 0
            if aspect_ratio > 1.5 and aspect_ratio < 6.0 and w_b > 40:
                location = approx
                break
            
    detected_image = image_resized.copy()
    cropped_gray = None
    cropped_rgb = None
    status_success = False
    
    metadata = {
        "plate_size": "-",
        "plate_ratio": "-",
        "contour_area": "-",
        "detection_status": "NO PLATE FOUND"
    }
    
    if location is not None:
        status_success = True
        # Gambar kotak hijau pada plat yang terdeteksi
        cv2.drawContours(detected_image, [location], -1, (0, 255, 0), 3)
        
        # Cari bounding box
        x, y, w, h = cv2.boundingRect(location)
        
        # Potong area plat
        cropped_gray = gray[y:y+h, x:x+w]
        cropped_rgb = image_resized[y:y+h, x:x+w]
        
        contour_area = cv2.contourArea(location)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        metadata = {
            "plate_size": f"{w}x{h} px",
            "plate_ratio": f"{aspect_ratio:.2f}",
            "contour_area": f"{int(contour_area)} px²",
            "detection_status": "SUCCESS"
        }
    else:
        # FALLBACK: Jika kontur tidak ditemukan, gunakan seluruh gambar sebagai crop (untuk close-up)
        # Kami set deteksi ke status FALLBACK
        cropped_gray = gray.copy()
        cropped_rgb = image_resized.copy()
        
        # Nilai metadata gambar penuh
        aspect_ratio = float(w_res) / h_res if h_res > 0 else 0
        metadata = {
            "plate_size": f"{w_res}x{h_res} px (Full)",
            "plate_ratio": f"{aspect_ratio:.2f} (Full)",
            "contour_area": "-",
            "detection_status": "FALLBACK_FULL_IMAGE"
        }
        
    # Konversi citra abu-abu ke format RGB agar Gradio dapat menampilkannya dengan benar
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    bfilter_rgb = cv2.cvtColor(bfilter, cv2.COLOR_GRAY2RGB)
    edged_rgb = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB)
    morph_rgb = cv2.cvtColor(morph, cv2.COLOR_GRAY2RGB)
    
    cropped_gray_rgb = cv2.cvtColor(cropped_gray, cv2.COLOR_GRAY2RGB)
    
    return {
        "success": status_success,
        "images": {
            "grayscale": gray_rgb,
            "bilateral": bfilter_rgb,
            "edged": edged_rgb,
            "morph": morph_rgb,
            "contour": contour_image,
            "detected": detected_image,
            "cropped": cropped_gray_rgb,
            "cropped_rgb": cropped_rgb
        },
        "metadata": metadata
    }
