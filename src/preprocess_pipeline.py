import cv2
import numpy as np
from skimage.morphology import skeletonize

def normalize_image(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def deskew(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    gray_inv = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return image  
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def scale_image(image, width=640, height=640):
    return cv2.resize(image, (width, height))

def remove_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def convert_to_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return image.copy()

def threshold_image(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def skeletonize_image(binary_img):
    binary_bool = binary_img > 0
    skeleton_bool = skeletonize(binary_bool)
    return (skeleton_bool.astype(np.uint8)) * 255

def preprocess_pipeline(image_or_path):
    if isinstance(image_or_path, str):
        image = cv2.imread(image_or_path)
        if image is None:
            raise ValueError(f"Could not read image from path: {image_or_path}")
    else:
        image = image_or_path

    norm_img = normalize_image(image)
    deskewed_img = deskew(norm_img)
    scaled_img = scale_image(deskewed_img, 640, 640)
    denoised_img = remove_noise(scaled_img)
    gray_img = convert_to_grayscale(denoised_img)
    binary_img = threshold_image(gray_img)
    skeleton_img = skeletonize_image(binary_img)
    return skeleton_img
