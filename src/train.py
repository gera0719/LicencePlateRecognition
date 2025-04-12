from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="/mnt/g/Linux/UbuntuWSWws/plate_recognition_project/data/split/stepzero/dataset.yaml",
    epochs=5,
    imgsz=640,
    batch=16,
    project="/mnt/g/Linux/UbuntuWSWws/plate_recognition_project/runs/train",  # default project folder
    name="stepzerotest"  # creates a folder called "my_experiment" inside runs/train
)


