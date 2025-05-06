from ultralytics import YOLO
import os

def detect_plates():
    model_path = "/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/models/stepzerotest/weights/best.pt"

    source_path = "/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/data/test"

    project_path = "/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/result"
    name = "platedetected"

    model = YOLO(model_path)

    results = model.predict(
        source=source_path,
        project=project_path,
        name=name,
        conf=0.5,
        save=True,
        save_crop=True
    )
    save_dir = model.predictor.save_dir
    return save_dir

if __name__ == "__main__":
    detect_plates()
