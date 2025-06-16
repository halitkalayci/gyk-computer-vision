import cv2 #opencv-python wrapper


# RGB
# BGR

# OpenCV imageları BGR olarak okur, ama matplotlib ve çoğu yeni kütüphane RGB ile çalışır..
img = cv2.imread('kedi.jpg')

height, width = img.shape[:2]

# Resize
resized_img = cv2.resize(img, (width // 2, height // 2)) # (100,100) => width, height

# Rotate (Döndürme)
center = (width // 2, height // 2) ## o anki yükseklik ve genişlik ortası.
rotation_matrix = cv2.getRotationMatrix2D(center, 90, 1.0)
rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))

# Flip (Yansıma)
flipped_img = cv2.flip(img, -1) # 1=> Yatay yansıma, 0=>Dikey Yansıma, -1 => Hem yatay hem de dikey yansıma

# Histogram Eşitleme - (Kontrastı artırır) Görüntülerdeki netliği artırmak. - Grayscale için kullanılır.
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equalized_img = cv2.equalizeHist(grayscale)

# Blur (Bulanıklık) => Ek araştırma (Opsiyonel)
blurred_img = cv2.GaussianBlur(img, (15,15), 0)

# Thresholding (Eşikleme) =>  Nesne-Arka plan ayrımı için kullanılır.
# Gürültü Temizleme
_, thresholded_img = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY) # 127 => Eşik değeri, 255 => Maksimum değer, cv2.THRESH_BINARY => Binary eşikleme




#Boilerplate Kod => Basmakalıp kod
cv2.imshow('Kedi', thresholded_img)
cv2.waitKey(0)
cv2.destroyAllWindows() # Bir tuşa basılana kadar imageı göster.
