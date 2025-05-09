import cv2
import os
import shutil
from pathlib import Path

def clahe_enhancement(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def bilateral_denoise(image):
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def unsharp_mask(image):
    blur = cv2.GaussianBlur(image, (9, 9), 10.0)
    return cv2.addWeighted(image, 1.5, blur, -0.5, 0)

def preprocess_training_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    img = clahe_enhancement(img)
    img = bilateral_denoise(img)
    img = unsharp_mask(img)
    return img

def process_folder(images_input_dir, labels_input_dir, images_output_dir, labels_output_dir):
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    for filename in os.listdir(images_input_dir):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(images_input_dir, filename)
            processed = preprocess_training_image(img_path)
            cv2.imwrite(os.path.join(images_output_dir, filename), processed)

            label_filename = Path(filename).with_suffix(".txt").name
            label_src = os.path.join(labels_input_dir, label_filename)
            label_dst = os.path.join(labels_output_dir, label_filename)

            if os.path.exists(label_src):
                shutil.copy(label_src, label_dst)

    print(f"Processed images saved to: {images_output_dir}")
    print(f"Labels copied to: {labels_output_dir}")

if __name__ == "__main__":
    images_input = "/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/data/raw/stepzero/images"
    labels_input = "/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/data/processed/stepzero/annotations"
    images_output = "/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/data/processed/stepzero_imgproc/images"
    labels_output = "/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/data/processed/stepzero_imgproc/annotations"
    
    process_folder(images_input, labels_input, images_output, labels_output)
