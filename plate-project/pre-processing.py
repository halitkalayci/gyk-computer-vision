import os
import cv2 
import numpy as np

IMAGE_DIR = "data/images"
LABEL_DIR = "data/labels"
IMG_SIZE = 640
BATCH_SIZE = 100  # Veriyi parçalar halinde işle

def process_batch(start_idx, end_idx, filenames):
    """Bir grup resmi işle ve kaydet"""
    X_batch = []
    y_batch = []
    
    for i in range(start_idx, min(end_idx, len(filenames))):
        filename = filenames[i]
        
        if not filename.endswith((".jpg",".png",".jpeg")):
            continue

        img_path = os.path.join(IMAGE_DIR, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w, _ = img.shape

        label_name = filename.rsplit(".",1)[0]
        label_path = os.path.join(LABEL_DIR, f"{label_name}.txt")

        if not os.path.exists(label_path):
            continue

        # Resmi resize et ve float32 kullan (float64 yerine)
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_resized = img_resized.astype(np.float32) / 255.0  # float32 kullan
        
        # Label dosyasını oku
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Her bir satır için işle
        labels = []
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    labels.append([class_id, x_center, y_center, width, height])
        
        # Veri setine ekle
        if labels:
            X_batch.append(img_resized)
            y_batch.append(labels[0])
    
    return np.array(X_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32)

# Ana işlem
print("Veri işleme başlıyor...")
filenames = [f for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg",".png",".jpeg"))]
total_files = len(filenames)
print(f"Toplam {total_files} resim dosyası bulundu.")

# İlk batch'i işle
X_all = []
y_all = []

num_batches = (total_files + BATCH_SIZE - 1) // BATCH_SIZE
print(f"Veri {num_batches} parça halinde işlenecek (her parçada {BATCH_SIZE} resim)")

for batch_idx in range(num_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = start_idx + BATCH_SIZE
    
    print(f"İşleniyor: Parça {batch_idx + 1}/{num_batches} ({start_idx}-{min(end_idx, total_files)})")
    
    X_batch, y_batch = process_batch(start_idx, end_idx, filenames)
    
    if len(X_batch) > 0:
        X_all.append(X_batch)
        y_all.append(y_batch)
    
    print(f"Bu parçada {len(X_batch)} resim işlendi")

# Tüm batch'leri birleştir
if X_all:
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    
    print(f"Toplam {len(X)} resim yüklendi.")
    print(f"Toplam {len(y)} etiket yüklendi.")
    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"y shape: {y.shape}, dtype: {y.dtype}")
    
    # Bellek kullanımını hesapla
    memory_gb = (X.nbytes + y.nbytes) / (1024**3)
    print(f"Yaklaşık bellek kullanımı: {memory_gb:.2f} GB")
    
    # Dosyalara kaydet
    print("Dosyalar kaydediliyor...")
    np.save("X.npy", X)
    np.save("y.npy", y)
    print("Kaydetme tamamlandı!")
else:
    print("İşlenecek geçerli veri bulunamadı!")
