# C:\Program Files (x86)\Tesseract-OCR
# opt/homebrew/Cellar/tesseract/5.4.01/bin/tesseract



import pytesseract
import cv2 

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image = cv2.imread('testocr.png')

text = pytesseract.image_to_string(image, lang='eng')

#print(text)

print("****")

fatura_img = cv2.imread('fatura.png')
fatura_img = cv2.resize(fatura_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

gray = cv2.cvtColor(fatura_img, cv2.COLOR_BGR2GRAY) # OCR için gri tonlama
denoised = cv2.fastNlMeansDenoising(gray, h=50) # h=filtre gücü

cv2.imshow('denoised', denoised)
cv2.waitKey(0)
cv2.destroyAllWindows()

#threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

threshold = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imshow('th', threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

# morfolojik işlemler -> OCR'da metni en iyi okunabilecek hale getirelim.
# ocr sonrası fatura tutarını bulalım.
# 10 farklı fatura resmi ile test edeceğiz.

fatura_text = pytesseract.image_to_string(threshold, config=r'--psm 6', lang='tur')

print(fatura_text)

# 20:30