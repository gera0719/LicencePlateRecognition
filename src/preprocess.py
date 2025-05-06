import cv2
import os

#BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = "/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/data/raw/firstsubset/images"
OUTPUT_DIR = "/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/data/processed/firstsubset/images"
#os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_image(image_path, output_folder):
    
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    equalized = cv2.equalizeHist(gray)

    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

    edges = cv2.Canny(blurred, 100, 200)

    os.makedirs(output_folder, exist_ok=True)

    
    cv2.imwrite(os.path.join(output_folder, "gray_" + os.path.basename(image_path)), gray)
    cv2.imwrite(os.path.join(output_folder, "equalized_" + os.path.basename(image_path)), equalized)
    cv2.imwrite(os.path.join(output_folder, "blurred_" + os.path.basename(image_path)), blurred)
    cv2.imwrite(os.path.join(output_folder, "edges_" + os.path.basename(image_path)), edges)

    

for subset in ["netherlands_day", "switzerland"]:
    subset_input_path = os.path.join(INPUT_DIR, subset)
    subset_output_path = os.path.join(OUTPUT_DIR, subset)

    if not os.path.exists(subset_input_path):
        print(f"Error: Input folder not found: {subset_input_path}")
        continue
    
    for image_name in os.listdir(subset_input_path):
        image_path = os.path.join(subset_input_path, image_name)

        if os.path.isdir(image_path):
            print(f"Skipping directory: {image_path}")
            continue

        print(f"Preprocessing image: {image_path}")
        preprocess_image(image_path, subset_output_path)

print(f"Preprocessing complete! Processed images saved in: {OUTPUT_DIR}")
