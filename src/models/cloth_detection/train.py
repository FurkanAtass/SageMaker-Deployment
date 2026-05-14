import os
from dotenv import load_dotenv
from src.models.cloth_detection.model import ClothDetectionYOLO

load_dotenv()
env = os.getenv("ENV").lower()
PROJECT_DIR = f"{os.getcwd()}/src/models/cloth_detection/runs"

model = ClothDetectionYOLO(
    env=env,
    base_weights="yolo26n.pt",
    data="src/models/cloth_detection/deepfashion2_yolo/dataset.yaml",
    dataset="deepfashion2-1k",
    batch=16,
    epochs=10,
    imgsz=640,
    project_dir=PROJECT_DIR,
)

model.fit_and_log()
