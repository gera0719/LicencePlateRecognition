import cv2
import numpy as np
import math
import os
import pytesseract
from PIL import Image
from typing import Union


def resize_image(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

def detect_edges_and_compute_angle(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None:
        return 0.0

    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return 0.0

    median_angle = np.median(angles)
    return median_angle


def deskew_image_by_lines(image: np.ndarray) -> np.ndarray:
    angle = detect_edges_and_compute_angle(image)
    if abs(angle) < 0.1:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return deskewed

# def is_upside_down(image: np.ndarray) -> bool:
#     normal = pytesseract.image_to_string(image, config='--psm 7')
#     flipped = pytesseract.image_to_string(cv2.rotate(image, cv2.ROTATE_180), config='--psm 7')

#     score_normal = sum(c.isalnum() for c in normal)
#     score_flipped = sum(c.isalnum() for c in flipped)
#     return score_flipped > score_normal

def denoise_image(image: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(image, 9, 75, 75)

def grayscale_image(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

def threshold_image(image: np.ndarray) -> np.ndarray:
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def crop_largest_white_rectangle(image: np.ndarray) -> np.ndarray:
    if len(image.shape) != 2:
        raise ValueError("Expected a binary grayscale image.")

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)

    max_area = 0
    best_idx = 0
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > max_area:
            max_area = area
            best_idx = i

    if max_area == 0:
        return image

    x, y, w, h, _ = stats[best_idx]
    cropped = image[y:y+h, x:x+w]
    return cropped


def preprocess_pipeline(image_input: Union[str, np.ndarray]) -> np.ndarray:
    image = cv2.imread(image_input) if isinstance(image_input, str) else image_input
    if image is None:
        raise ValueError(f"Image not found: {image_input}")

    image = resize_image(image, scale=2.0)
    image = deskew_image_by_lines(image)

    # if is_upside_down(image):
    #     image = cv2.rotate(image, cv2.ROTATE_180)

    image = denoise_image(image)
    image = grayscale_image(image)
    image = threshold_image(image)
    image = crop_largest_white_rectangle(image)


    return image

if __name__ == "__main__":
    input_path = "/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/result/platedetected6/crops/license_plate/5.jpg"
    output_dir = "/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/result/preprocess_ocr_output/"
    os.makedirs(output_dir, exist_ok=True)

    processed = preprocess_pipeline(input_path)

    output_path = os.path.join(output_dir, "processed_image.jpg")
    cv2.imwrite(output_path, processed)
    print(f"Processed image saved to: {output_path}")
