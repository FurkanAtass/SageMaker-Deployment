from ultralytics import YOLO
import os
print(os.getcwd())
model = YOLO("yolo26n.pt")

results = model.train(
    data="src/train/deepfashion2_yolo/dataset.yaml", 
    batch=16,
    epochs=100,
    imgsz=640,
    name="deepfashion2_yolo26n",
    project=f"{os.getcwd()}/src/train/runs",    
)
