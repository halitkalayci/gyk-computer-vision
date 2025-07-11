import cv2
from matplotlib import pyplot as plt
import albumentations as A
import os 
image_path = "data/images/1.jpg"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

bboxes = []
class_labels = []

with open("data/labels/1.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])
        bboxes.append([x_center, y_center, width, height])
        class_labels.append(class_id)

augmentations = [
    # Görseli rastgele -20 ile +20 arasında döndürür.
    ("Rotate 20 degrees" , A.Rotate(limit=20, p=1.0)),

    # Görseli yatay olarak çevirir. (Ayna efekti)
    ("Horizontal Flip" , A.HorizontalFlip(p=1.0)),

    # Parlaklık ve kontrastı rastgele değiştirir. + -
    ("Brightness", A.RandomBrightnessContrast(p=1.0)),

    # Görseli rastgele %20 oranına kadar büyütür veya küçültür.
    ("Zoom", A.RandomScale(scale_limit=0.4, p=1.0)),

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
    image_aug = cv2.imread(image_path)
    image_aug = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)

    transform = A.Compose(
        [aug],
        bbox_params=A.BboxParams(format='yolo', label_fields=["class_labels"])
    )

    #try:
    augmented = transform(image=image_aug, bboxes=bboxes, class_labels=class_labels)
    image_aug = augmented["image"]
    bboxes = augmented["bboxes"]
    labels = augmented["class_labels"]

    # 1.jpg -> Horizontal Flip
    # 1_Horizontal_Flip.jpg
    
    base_img_name = os.path.splitext(os.path.basename(image_path))[0]
    print( "BaseImgName:", base_img_name)

    img_name = f"data/images/{base_img_name}_{title.replace(' ', '_')}.jpg"
    label_name = f"data/labels/{base_img_name}_{title.replace(' ', '_')}.txt"


    cv2.imwrite(img_name, cv2.cvtColor(image_aug, cv2.COLOR_RGB2BGR))

    with open(label_name, "w") as f:
        for bbox in bboxes:
            print(f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
            f.write(f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

    h, w, _ = image_aug.shape

    for bbox in bboxes:
        print(bbox)
        x_center, y_center, width, height = bbox
        x1 = int((x_center - width/2) * w)
        x2 = int((x_center + width/2) * w)
        y1 = int((y_center - height/2) * h)
        y2 = int((y_center + height/2) * h)
        cv2.rectangle(image_aug, (x1,y1), (x2,y2), (255,0,0), 5)

    plt.subplot(3,3,i)
    plt.imshow(image_aug)
    plt.title(title)
    plt.axis("off")

    #except Exception as e:
       #print(f"Error applying {title}: {e}")
       # continue

plt.tight_layout()
plt.show()

# Augmente görselleri kaydet.

# 1.jpg -> 1_blur.jpg 1_zoom.jpg 1_rotate.jpg 1_horizontal_flip.jpg 1_brightness.jpg 1_gauss_noise.jpg 1_cutout.jpg 1_motion_blur.jpg 1_median_blur.jpg
# 1.txt -> 1_blur.txt 1_zoom.txt 1_rotate.txt 1_horizontal_flip.txt 1_brightness.txt 1_gauss_noise.txt 1_cutout.txt 1_motion_blur.txt 1_median_blur.txt