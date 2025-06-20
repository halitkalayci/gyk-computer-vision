# C:\Program Files (x86)\Tesseract-OCR
# opt/homebrew/Cellar/tesseract/5.4.01/bin/tesseract

import pytesseract
import cv2 

image = cv2.imread('testocr.png')

text = pytesseract.image_to_string(image, lang='eng')

#print(text)

print("****")