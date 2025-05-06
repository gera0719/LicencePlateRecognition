import pytesseract
from PIL import Image
import numpy as np
import preprocess_pipeline as pp


img_path = "/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/result/platedetected6/crops/license_plate/3.jpg"

    
img = pp.preprocess_pipeline(img_path)
    
custom_config = r'--psm 7'  
text = pytesseract.image_to_string(img, config=custom_config)
    
print("OCR Output:")
print("[" + text + "]")
