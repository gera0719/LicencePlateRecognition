import os
import shutil
import random

# Get the project root directory (one level up from 'src')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths for raw images and labels (adjust as needed)
RAW_IMAGES_DIR = os.path.join(BASE_DIR, "data", "processed", "stepzero", "images")
RAW_LABELS_DIR = os.path.join(BASE_DIR, "data", "processed", "stepzero", "annotations")

# Define output base directory for the split dataset under stepzero
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "data", "split", "stepzero")
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_BASE_DIR, "images")
OUTPUT_LABELS_DIR = os.path.join(OUTPUT_BASE_DIR, "labels")

# Split percentages: 70% train, 20% validation, and the rest for test.
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2

# Create output directories for images and labels for each split:
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_IMAGES_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_LABELS_DIR, split), exist_ok=True)

# List image files from the raw images directory (assumes .jpg, .jpeg, or .png)
all_images = [f for f in os.listdir(RAW_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
random.shuffle(all_images)  # Shuffle images randomly

total_images = len(all_images)
train_count = int(total_images * TRAIN_RATIO)
val_count = int(total_images * VAL_RATIO)
test_count = total_images - train_count - val_count

print(f"Total images: {total_images}")
print(f"Splitting as: {train_count} train, {val_count} val, {test_count} test")

# Split the image list accordingly
train_images = all_images[:train_count]
val_images = all_images[train_count:train_count + val_count]
test_images = all_images[train_count + val_count:]

def copy_files(image_list, split):
    """Copy image files and their corresponding annotation files into split directories."""
    for img_filename in image_list:
        # Copy image file
        src_img_path = os.path.join(RAW_IMAGES_DIR, img_filename)
        dst_img_path = os.path.join(OUTPUT_IMAGES_DIR, split, img_filename)
        shutil.copy(src_img_path, dst_img_path)
        
        # Determine corresponding annotation file (assumes same base name with .txt)
        annotation_filename = os.path.splitext(img_filename)[0] + ".txt"
        src_label_path = os.path.join(RAW_LABELS_DIR, annotation_filename)
        dst_label_path = os.path.join(OUTPUT_LABELS_DIR, split, annotation_filename)
        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dst_label_path)
        else:
            print(f"⚠️ Annotation not found for image: {img_filename}")

# Copy files into the respective split folders
copy_files(train_images, "train")
copy_files(val_images, "val")
copy_files(test_images, "test")

print("✅ Dataset splitting complete!")
