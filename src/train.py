
import os

# vmi nem jo, nem erni el
os.environ['LD_LIBRARY_PATH'] = "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

from ultralytics import YOLO
# eleg lenne csak terminalban meghivni a sima yolos traininget
def main():
    model = YOLO("yolov8n.pt")
    
    model.train(
        data="/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/data/split/stepzero/dataset.yaml",
        epochs=60,
        imgsz=640,
        batch=16,
        project="/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/models",  
        name="stepzerotest" 
    )
    
    model.val()

if __name__ == "__main__":
    main()
