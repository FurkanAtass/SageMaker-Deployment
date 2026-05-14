import os
from dotenv import load_dotenv

from src.models.cloth_detection.model import ClothDetectionYOLO

load_dotenv()
env = os.getenv("ENV").lower()

def main():
    model = ClothDetectionYOLO(env=env)

    results = model.predict(
        ["/home/furkan/SageMaker-Deployment/src/models/cloth_detection/deepfashion2_yolo/train/images/000121.jpg"],
    )
    for i, result in enumerate(results):
        result.save(filename=f"result-{i}.jpg")


if __name__ == "__main__":
    main()
