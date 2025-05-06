import pytesseract
from PIL import Image
import numpy as np
import preprocess_pipeline as pp
import cv2


img_path = "/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/result/platedetected6/crops/license_plate/102.jpg"

    
img = pp.preprocess_pipeline(img_path)
custom_config = r'--psm 13 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
img_rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
text = pytesseract.image_to_string(img_rgb, config=custom_config)

text = pytesseract.image_to_string(img, config=custom_config)
    
print("OCR Output:")
print("[" + text + "]")
