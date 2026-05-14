from src.mlflow_service.mlflow_functions import load_model
import os
from dotenv import load_dotenv

from src.mlflow_service.yolo_wrapper import YOLOWrapper

load_dotenv()

def main():
    REGISTERED_MODEL_NAME = "cloth-detection"
    ENV = os.getenv("ENV").lower()

    model = load_model(REGISTERED_MODEL_NAME, ENV)

    results = model.predict(
        ["/home/furkan/SageMaker-Deployment/src/train/deepfashion2_yolo/train/images/000121.jpg"],
        # params={"imgsz": 640, "conf": 0.01},
    )
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.save(filename="result.jpg")  # save to disk


if __name__ == "__main__":
    main()