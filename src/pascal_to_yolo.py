import os
import xml.etree.ElementTree as ET

# Directory paths (adjust these based on your folder structure)
annotations_dir = "/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/data/raw/stepzero/annotations"
output_labels_dir = "/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/data/processed/stepzero/annotations"
os.makedirs(output_labels_dir, exist_ok=True)

def convert_voc_to_yolo(xml_file, output_file):
    """Convert a single VOC XML file to a YOLO format text file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get image dimensions from the XML
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)
    
    # Open the output file to write in YOLO format
    with open(output_file, "w") as f_out:
        for obj in root.iter("object"):
            # Get class name. In your annotation, it is "licence".
            class_name = obj.find("name").text.strip()
            
            # Map the class to an integer ID.
            # If you only have one class ("licence"), you can set class_id = 0.
            class_id = 0
            
            # Get bounding box coordinates
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            
            # Convert to YOLO coordinates: x_center, y_center, width, height
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # Write the annotation in YOLO format: class_id x_center y_center width height
            f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Process all XML files in the annotations directory
for file_name in os.listdir(annotations_dir):
    if file_name.endswith(".xml"):
        xml_path = os.path.join(annotations_dir, file_name)
        # Create corresponding txt file with same name
        txt_filename = file_name.replace(".xml", ".txt")
        output_path = os.path.join(output_labels_dir, txt_filename)
        convert_voc_to_yolo(xml_path, output_path)
        print(f"Converted {xml_path} to {output_path}")

print("✅ Annotation conversion complete!")
