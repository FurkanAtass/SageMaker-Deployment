from src.mlflow_service.mlflow_functions import load_model
import os
from dotenv import load_dotenv

from src.mlflow_service.yolo_wrapper import YOLOWrapper

load_dotenv()

def main():
    REGISTERED_MODEL_NAME = "cloth-detection"
    ENV = os.getenv("ENV").lower()

    model = load_model(REGISTERED_MODEL_NAME, ENV)

    # To use model params, use model.unwrap_python_model() to access the underlying YOLOWrapper instance directly
    results = model.predict(
        ["/home/furkan/SageMaker-Deployment/src/train/deepfashion2_yolo/train/images/000121.jpg"],
    )
    for i, result in enumerate(results):
        result.save(filename=f"result-{i}.jpg")  # save to disk


if __name__ == "__main__":
    main()