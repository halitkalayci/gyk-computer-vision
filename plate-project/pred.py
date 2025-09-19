from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import glob
import easyocr

model = YOLO("runs/detect/train2/weights/best.pt")

# EasyOCR okuyucu oluştur (Türkçe ve İngilizce)
reader = easyocr.Reader(['tr', 'en'])

def preprocess_plate_image(image):
    """
    Plaka görüntüsünü OCR için optimize eder
    """
    # Görüntüyü BGR'den RGB'ye çevir (eğer gerekirse)
    if len(image.shape) == 3:
        # Mavi kanalı azalt (mavi plaka kısmını yok say)
        image_processed = image.copy()
        image_processed[:, :, 2] = image_processed[:, :, 2] * 0.2  # Mavi kanalı daha fazla azalt
        
        # Grayscale'e çevir
        gray = cv2.cvtColor(image_processed, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Görüntüyü büyüt (OCR performansı için)
    height, width = gray.shape
    scale_factor = 4  # Daha fazla büyüt
    new_height, new_width = height * scale_factor, width * scale_factor
    resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Histogram eşitleme (kontrastı artır)
    equalized = cv2.equalizeHist(resized)
    
    # Gaussian blur uygula (gürültüyü azalt)
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    
    # Otsu threshold ile binary görüntü oluştur (daha iyi karakter tanıma)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morfolojik işlemler (karakterleri doldur)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # Önce opening (gürültüyü temizle)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # Sonra closing (karakterleri doldur)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # Dilatation ile karakterleri kalınlaştır
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilated = cv2.dilate(closed, dilate_kernel, iterations=1)
    
    return dilated

def preprocess_plate_image_alternative(image):
    """
    Alternatif plaka görüntü işleme yöntemi
    """
    # Görüntüyü BGR'den RGB'ye çevir
    if len(image.shape) == 3:
        # Sadece kırmızı ve yeşil kanalları kullan (mavi kanalı tamamen yok say)
        image_processed = image.copy()
        image_processed[:, :, 2] = 0  # Mavi kanalı sıfırla
        
        # Grayscale'e çevir
        gray = cv2.cvtColor(image_processed, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Görüntüyü büyüt
    height, width = gray.shape
    scale_factor = 5  # Daha da büyük
    new_height, new_width = height * scale_factor, width * scale_factor
    resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(resized)
    
    # Median blur (gürültüyü azalt)
    blurred = cv2.medianBlur(enhanced, 3)
    
    # Basit threshold (manuel değer)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    # Morfolojik işlemler - karakterleri doldur
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Closing ile karakterleri doldur
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Erosion ile ince çizgileri temizle
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    eroded = cv2.erode(closed, erosion_kernel, iterations=1)
    
    return eroded

# data/test klasöründeki tüm fotoğrafları bul
test_folder = "data/test"
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
image_paths = []

for extension in image_extensions:
    image_paths.extend(glob.glob(os.path.join(test_folder, extension)))
    image_paths.extend(glob.glob(os.path.join(test_folder, extension.upper())))

print(f"Bulunan fotoğraf sayısı: {len(image_paths)}")

# Eğer fotoğraf yoksa uyarı ver
if not image_paths:
    print("data/test klasöründe fotoğraf bulunamadı!")
    exit()

# Toplam subplot sayısını hesapla (orijinal görüntüler + crop edilen plakalar)
total_plates = 0
all_results = []

# Önce tüm sonuçları topla
for image_path in image_paths:
    print(f"İşleniyor: {image_path}")
    
    # Fotoğrafı oku
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # YOLO ile tahmin yap
    result = model.predict(image)
    
    print(f"Tespit edilen nesne sayısı: {len(result[0].boxes)}")
    
    boxes = result[0].boxes
    valid_boxes = []
    
    # Tespit edilen plakaları işle
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        
        if conf > 0.75:
            # Orijinal görüntüde kutu çiz
            cv2.rectangle(image_rgb, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)
            label = f"Plaka - {conf:.2f}"
            cv2.putText(image_rgb, label, (int(x1),int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            
            # Crop edilen plaka
            x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
            cropped_plate = image_rgb[y1_int:y2_int, x1_int:x2_int]
            
            # Plaka görüntüsü ön işleme - birden fazla yöntem dene
            processed_plate = preprocess_plate_image(cropped_plate)
            processed_plate_alt = preprocess_plate_image_alternative(cropped_plate)
            
            # EasyOCR ile plaka metnini oku - önce ana yöntem
            plate_text = ""
            confidence = 0
            
            try:
                ocr_results = reader.readtext(processed_plate)
                if ocr_results:
                    best_result = max(ocr_results, key=lambda x: x[2])
                    plate_text = best_result[1]
                    confidence = best_result[2]
                    print(f"Ana yöntem - Plaka metni: {plate_text} (Güven: {confidence:.2f})")
                
                # Eğer ana yöntem başarısızsa alternatif yöntemi dene
                if confidence < 0.5 or len(plate_text) < 3:
                    ocr_results_alt = reader.readtext(processed_plate_alt)
                    if ocr_results_alt:
                        best_result_alt = max(ocr_results_alt, key=lambda x: x[2])
                        if best_result_alt[2] > confidence:
                            plate_text = best_result_alt[1]
                            confidence = best_result_alt[2]
                            print(f"Alternatif yöntem - Plaka metni: {plate_text} (Güven: {confidence:.2f})")
                
                if confidence < 0.3:
                    print("Plaka metni okunamadı")
                    plate_text = "Okunamadı"
                    
            except Exception as e:
                print(f"OCR hatası: {e}")
                plate_text = "Hata"
            
            valid_boxes.append((cropped_plate, conf, plate_text, processed_plate))
    
    all_results.append((image_rgb, os.path.basename(image_path), valid_boxes))
    total_plates += len(valid_boxes)

# Subplot düzenini hesapla (orijinal + işlenmiş plakalar)
num_images = len(image_paths)
total_subplots = num_images + (total_plates * 2)  # Her plaka için 2 subplot (orijinal + işlenmiş)
cols = min(5, total_subplots)  # Maksimum 5 sütun
rows = (total_subplots + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
if total_subplots == 1:
    axes = [axes]
elif rows == 1:
    axes = axes.reshape(1, -1)
else:
    axes = axes.flatten()

# Sonuçları göster
subplot_idx = 0

for image_rgb, filename, valid_boxes in all_results:
    # Orijinal görüntüyü göster
    axes[subplot_idx].imshow(image_rgb)
    axes[subplot_idx].set_title(f"Orijinal: {filename}")
    axes[subplot_idx].axis('off')
    subplot_idx += 1
    
    # Crop edilen plakaları göster
    for i, (cropped_plate, conf, plate_text, processed_plate) in enumerate(valid_boxes):
        if cropped_plate.size > 0:  # Boş crop kontrolü
            # Orijinal crop
            axes[subplot_idx].imshow(cropped_plate)
            axes[subplot_idx].set_title(f"Orijinal Crop {i+1}: {conf:.2f}")
            axes[subplot_idx].axis('off')
            subplot_idx += 1
            
            # İşlenmiş crop (eğer yer varsa)
            if subplot_idx < len(axes):
                axes[subplot_idx].imshow(processed_plate, cmap='gray')
                axes[subplot_idx].set_title(f"İşlenmiş: {plate_text}")
                axes[subplot_idx].axis('off')
                subplot_idx += 1

# Kullanılmayan subplot'ları gizle
for idx in range(subplot_idx, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.suptitle("Plaka Tespit Sonuçları ve Crop Edilen Plakalar", fontsize=16, y=0.98)
plt.show()

# xywh => x,y width,height