import cv2 #opencv-python wrapper
import matplotlib.pyplot as plt


# RGB
# BGR

# OpenCV imageları BGR olarak okur, ama matplotlib ve çoğu yeni kütüphane RGB ile çalışır..
img = cv2.imread('kedi.jpg')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(type(img_gray))
print(img_gray)
print(img_gray.ndim)
print(img_gray.shape)

plt.imshow(img_gray, cmap="gray")
plt.axis('off')
plt.show()
