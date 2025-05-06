import os
import xml.etree.ElementTree as ET

annotations_dir = "/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/data/raw/stepzero/annotations"
output_labels_dir = "/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/data/processed/stepzero/annotations"
os.makedirs(output_labels_dir, exist_ok=True)

def convert_voc_to_yolo(xml_file, output_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)
    
    with open(output_file, "w") as f_out:
        for obj in root.iter("object"):
            class_name = obj.find("name").text.strip()
            
            class_id = 0
            
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

for file_name in os.listdir(annotations_dir):
    if file_name.endswith(".xml"):
        xml_path = os.path.join(annotations_dir, file_name)
        txt_filename = file_name.replace(".xml", ".txt")
        output_path = os.path.join(output_labels_dir, txt_filename)
        convert_voc_to_yolo(xml_path, output_path)
        print(f"Converted {xml_path} to {output_path}")

print("Annotation conversion complete!")