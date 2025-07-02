# Transfer Learning => Eğitilmiş bir modeli kullanarak yeni bir model oluşturmak.

# imageları process
# label dosyalarını oku.
# imagelardan ve labellardan veri seti oluştur.
import os
import cv2 
IMAGE_DIR = "data/images"
LABEL_DIR = "data/labels"

IMG_SIZE = 640

X, y = [], []

for filename in os.listdir(IMAGE_DIR):
    if not filename.endswith((".jpg",".png",".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(img_path)

    if img is None:
        continue # döngüde sonraki adıma geç.

    h, w, _ = img.shape # orjinal boyut

    label_name = filename.rsplit(".",1)[0] # [1,jpg]
    label_path = os.path.join(LABEL_DIR, f"{label_name}.txt")

    if not os.path.exists(label_path):
        continue

    # Label dosyasını oku, X içerisine image'ın  resize edilmiş hali, y içerisine labeli ekle.
    
    # Resmi resize et
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Label dosyasını oku
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Her bir satır için işle (bir resimde birden fazla obje olabilir)
    labels = []
    for line in lines:
        line = line.strip()
        if line:  # Boş satır değilse
            parts = line.split()
            if len(parts) >= 5:  # YOLO formatı: class_id x_center y_center width height
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                labels.append([class_id, x_center, y_center, width, height])
    
    # Veri setine ekle
    if labels:  # Eğer label varsa
        X.append(img_resized)
        y.append(labels)

print(f"Toplam {len(X)} resim yüklendi.")
print(f"Toplam {len(y)} etiket yüklendi.")
