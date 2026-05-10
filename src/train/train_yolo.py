import os
from ultralytics import YOLO
from src.mlflow_service.mlflow_functions import (
    setup_experiment,
    start_run,
    log_training_params,
    log_training_metrics,
    log_run_artifacts,
)

EXPERIMENT_NAME = "YOLO 26 Training"
RUN_NAME = "deepfashion2-1k-yolo26n"

BATCH_SIZE = 16
EPOCHS = 3
IMGSZ = 640

setup_experiment(EXPERIMENT_NAME)

model = YOLO("yolo26n.pt")

with start_run(run_name=RUN_NAME):
    log_training_params({
        "model": "yolo26n",
        "batch": BATCH_SIZE,
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "dataset": "deepfashion2",
    })

    results = model.train(
        data="src/train/deepfashion2_yolo/dataset.yaml",
        batch=BATCH_SIZE,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        name=RUN_NAME,
        project=f"{os.getcwd()}/src/train/runs",
    )

    log_training_metrics(results.results_dict)
    log_run_artifacts(f"{os.getcwd()}/src/train/runs", RUN_NAME)
