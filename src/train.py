from ultralytics import YOLO
def train_model():
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
    train_model()
