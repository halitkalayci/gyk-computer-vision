import cv2
from matplotlib import pyplot as plt
import albumentations as A

image_path = "data/images/1.jpg"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



augmentations = [
    # Görseli rastgele -20 ile +20 arasında döndürür.
    ("Rotate 20 degrees" , A.Rotate(limit=20, p=1.0)),

    # Görseli yatay olarak çevirir. (Ayna efekti)
    ("Horizontal Flip" , A.HorizontalFlip(p=1.0)),

    # Parlaklık ve kontrastı rastgele değiştirir. + -
    ("Brightness", A.RandomBrightnessContrast(p=1.0)),

    # Görseli rastgele %20 oranına kadar büyütür veya küçültür.
    ("Zoom", A.RandomScale(scale_limit=0.2, p=1.0)),

    # Probability => 1.0
    # Focus problemi olan fotoğraf gibi..
    ("Gauissan Blur", A.GaussNoise(var_limit=(10.0,50.0), p=1.0)),

    # Rastgele 5 noktada 20x20 siyah diktörtgen ile resmin o bölümünü kapatır.
    ("Cutout", A.CoarseDropout(max_holes=5, max_height=20, max_width=20, p=1.0)),

    # Hareketli nesneler ile 
    ("Blur", A.OneOf(
        [
            A.MotionBlur(p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], 
        p=0.1
    ))
]


plt.figure(figsize=(18,12))
for i, (title,aug) in enumerate(augmentations,start=2):
    augmented_image = aug(image=image)["image"]
    plt.subplot(3,3,i)
    plt.imshow(augmented_image)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()

# Augmente görselleri kaydet.