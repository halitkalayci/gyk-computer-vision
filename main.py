import cv2 #opencv-python wrapper
import numpy as np

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

# Canny Edge Detection (Kenar tespiti)
edges = cv2.Canny(thresholded_img, 100, 200)

# Bireysel ödev => Seçtiğiniz herhangi bir görüntüde seçtiğiniz
# bir objeyi kenarları düzgünce çizilecek şekilde çiziniz.

# Morfolojik işlemler ve segmentasyon.

# Erosion ve Dilation => Aşındırma ve Genişletme
# Erosion => Beyaz bölgeleri küçültür, gürültüyü temizler.
# Dilation => Beyaz bölgeleri büyütür, boşlukları doldurur. 

kernel = np.ones((3,3), np.uint8)

erosion = cv2.erode(thresholded_img, kernel, iterations=1)

dilation = cv2.dilate(thresholded_img, kernel, iterations=1)

# Opening => Erosion sonrası dilation işlemi. (Kağıttaki leke örneği => Küçük noktaları temizler, yazıyı bozmadan gürültüyü azaltır.)
opening = cv2.morphologyEx(thresholded_img, cv2.MORPH_OPEN, kernel)

# Closing => Dilation sonrası erosion işlemi. (Kağıttaki leke örneği => Küçük noktaları temizler, yazıyı bozmadan gürültüyü azaltır.)
closing = cv2.morphologyEx(thresholded_img, cv2.MORPH_CLOSE, kernel)



# Kontur Bulma

# Mode parametresi = cv2.RETR_TREE, cv2.RETR_EXTERNAL, cv2.RETR_CCOMP, cv2.RETR_LIST
# Tree => Tüm konturları hiyerarşik olarak bulur. (iç içe nesnelerde çok kullanılır.)
# External => Sadece dış konturları bulur. (sadece dış nesneyi bulmak için.)

# CCOMP => İki seviyeli hiyerarşik konturları bulur. (daha nadir kullanılır.)
# List => Tüm konturları bulur ama hiyerarşi bilgisi vermez. ()

# Kontur noktaları nasıl belirlensin?
# Method parametresi = cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_TC89_KCOS, cv2.CHAIN_APPROX_TC89_L1
# Simple => Kontur noktalarını en az sayıda nokta ile temsil etmeye çalışır. (en çok kullanılır.)
# None => Tüm noktaları alır. (Çok yer kaplar.)
# TC89_KCOS, L1 => Daha gelişmiş Douglas-Peucker algoritmaları.
contours, _ = cv2.findContours(thresholded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


print(f"Kontur sayısı: {len(contours)}")
print(f"Konturler: {contours}")

contour_img = img.copy()
# -1 => Tüm konturları çiz., 0-1-2-3 vb => Belirtilen kontur numarasını (index) çiz.
# 0,0,255 BGR => Kırmızı renk
# 2 => Çizgi kalınlığı
cv2.drawContours(contour_img, contours, -1, (0,0,255), 2)

# Connected Components (Bağlı Bileşenler)
num_labels, labels_img = cv2.connectedComponents(opening)

print(f"Bağlı bileşen sayısı: {num_labels-1}")
print(labels_img)

# Bağlı bileşenleri görselleştirme

label_hue = np.uint8(179 * labels_img / (num_labels-1))

# label 0 => hue = 0
# label 1 => hue 60
# label 2 => hue 120

blank_ch = 255 * np.ones_like(label_hue)

# 255 * => HSV renk uzayında 255 değeri en büyük değerdir. (HSV) Saturation-Value

labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[labels_img == 0] = 0


cv2.imshow('Kedi - Connected Components', labeled_img)


# Watershed Segmentasyonu -> Farklı bir imagela yeni bir dosyada yapılacak.

# GrabCut Segmentasyonu -> 

# Bir maske oluşturuyorum, resmin her bir pikseli için etiket tutacak.
mask = np.zeros(img.shape[:2], dtype=np.uint8)

# GrabCut için gerekli olan modeller.
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)
# 


# Manuel Koordinat
# x,y
# width,height
rect = (50,50,800,800)
# 
# bu imagedeki, bu maskeyi güncelleyerek, şu kareyi incele.
cv2.grabCut(labeled_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
# 5 => Iterasyon sayısı


# ilk maskeyi sadeleştiriyoruz.
mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
result = img * mask2[:,:,np.newaxis]
cv2.imshow('Kedi - GrabCut', result)
cv2.imshow('Kedi - Contours', contour_img)
cv2.imshow('Kedi - Threshold', thresholded_img)
cv2.imshow('Kedi - Opening', opening)
cv2.imshow('Kedi - Closing', closing)
cv2.waitKey(0)
cv2.destroyAllWindows() # Bir tuşa basılana kadar imageı göster.


# Opsiyonel => Arayüz ile fotoğraf seçilen bir uygulama
# Bir fotoğraf seçilecek,
# seçilen fotoğraftaki arkaplan kaldırılıp (dekupe)
# transparan haliyle .png olarak kaydedilecek.

# Özellik Çıkarımı ve Nesne Tespiti
