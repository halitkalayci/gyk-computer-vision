# OCR -> Optical Character Recognition

# Tesseract OCR, EasyOCR, PaddleOCR -> Açık kaynaklı ocr kütüphaneleri

# AWS Textract, Google Cloud Vision OCR, Azure CS OCR -> Bulut tabanlı ücretli kütüphaneler.

import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image = cv2.imread("testocr.png")

text = pytesseract.image_to_string(image, lang="eng")

print(type(text))
print(text)

#
fatura_img = cv2.imread("fatura.png")
gray = cv2.cvtColor(fatura_img, cv2.COLOR_BGR2GRAY)
# denoising -> gürültü azaltma
denoised = cv2.fastNlMeansDenoising(gray, h=35)


# thresholding
# block size (11) => Eşik değeri hesaplanırken bakılacak komşu boyutu -> Tek sayı olmalı 11,13,15
# c (2) -> Hesaplanan ortalama/gaussian değerinden çıkarılan sabit. eşik = gaussian - C
thresholded_img = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
#

fatura_text = pytesseract.image_to_string(thresholded_img, config="--psm 6",lang="tur")
print(fatura_text)


cv2.imshow("grayscale",gray)
cv2.imshow("denoised",denoised)
cv2.imshow("threshold",thresholded_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#

#Page segmentation modes:
#  0    Orientation and script detection (OSD) only.
#  1    Automatic page segmentation with OSD.
#  2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
#  3    Fully automatic page segmentation, but no OSD. (Default)
#  4    Assume a single column of text of variable sizes.
#  5    Assume a single uniform block of vertically aligned text.
#  6    Assume a single uniform block of text.
#  7    Treat the image as a single text line.
#  8    Treat the image as a single word.
#  9    Treat the image as a single word in a circle.
# 10    Treat the image as a single character.
# 11    Sparse text. Find as much text as possible in no particular order.
# 12    Sparse text with OSD.
# 13    Raw line. Treat the image as a single text line,
#       bypassing hacks that are Tesseract-specific.
