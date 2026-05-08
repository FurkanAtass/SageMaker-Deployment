import os
from ultralytics import YOLO
from src.mlflow_service.mlflow import (
    setup_experiment,
    start_run,
    log_training_params,
    log_training_metrics,
    log_model_artifact,
)

EXPERIMENT_NAME = "deepfashion2-yolo"

setup_experiment(EXPERIMENT_NAME)

model = YOLO("yolo26n.pt")

with start_run():
    log_training_params({
        "model": "yolo26n",
        "batch": 16,
        "epochs": 100,
        "imgsz": 640,
        "dataset": "deepfashion2",
    })

    results = model.train(
        data="src/train/deepfashion2_yolo/dataset.yaml",
        batch=16,
        epochs=100,
        imgsz=640,
        name="deepfashion2_yolo26n",
        project=f"{os.getcwd()}/src/train/runs",
    )

    log_training_metrics(results.results_dict)
    log_model_artifact(f"{os.getcwd()}/src/train/runs/deepfashion2_yolo26n/weights/best.pt")
