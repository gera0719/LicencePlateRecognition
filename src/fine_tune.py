from ultralytics import YOLO

def main():
    
    model = YOLO("/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/models/stepzerotest/weights/best.pt")

    model.train(
        data="/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/data/split/stepzero_imgproc/dataset.yaml",
        epochs=30,
        imgsz=640,
        batch=16,
        lr0=0.001,
        project="/mnt/g/Linux/UbuntuWSLws/plate_recognition_project/models",
        name="stepzeroftune",
        pretrained=True
    )

    model.val()

if __name__ == "__main__":
    main()