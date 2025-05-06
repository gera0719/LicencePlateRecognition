import os
import pytesseract
from PIL import Image
import preprocess_pipeline as pp


img_path = "/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/result/platedetected6/crops/license_plate"
custom_config = r'--psm 13 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def perform_ocr(folder_path):
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_path = os.path.join(folder_path, filename)

            try:
                img = pp.preprocess_pipeline(image_path)

                text = pytesseract.image_to_string(img, config=custom_config).strip()
                print(f"{filename}: [{text}]")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    perform_ocr(img_path)