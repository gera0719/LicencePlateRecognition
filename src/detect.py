import cv2
import pytesseract
from PIL import Image
import numpy as np


img_cv = cv2.imread("/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/result/platedetected5/crops/license_plate/4.jpg")

if img_cv is None:
    print("Error: Could not load image.")
else:
    
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    pil_img = Image.fromarray(thresh)
    
    custom_config = r'--psm 7'  
    text = pytesseract.image_to_string(pil_img, config=custom_config)
    
    print("OCR Output:")
    print("[" + text + "]")
