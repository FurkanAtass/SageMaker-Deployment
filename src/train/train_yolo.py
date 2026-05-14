import os
from ultralytics import YOLO
from src.mlflow_service.mlflow_functions import (
    setup_experiment,
    start_run,
    log_training_params,
    log_training_metrics,
    log_run_artifacts,
    make_epoch_callback,
    log_model,
)
from ultralytics import settings

EXPERIMENT_NAME = "Cloth Detection"
RUN_NAME = "deepfashion2-1k-yolo26n"
MODEL_NAME = "yolo26n-deepfashion2"
DATASET_NAME = "deepfashion2-1k"
BATCH_SIZE = 16
EPOCHS = 10

IMGSZ = 640
PROJECT_DIR = f"{os.getcwd()}/src/train/runs"

setup_experiment(EXPERIMENT_NAME)

model = YOLO("yolo26n.pt")
model.add_callback("on_train_epoch_end", make_epoch_callback())

with start_run(run_name=RUN_NAME):
    log_training_params({
        "model": "yolo26n",
        "batch": BATCH_SIZE,
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "dataset": DATASET_NAME,
    })

    results = model.train(
        data="src/train/deepfashion2_yolo/dataset.yaml",
        batch=BATCH_SIZE,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        name=RUN_NAME,
        project=PROJECT_DIR,
    )

    log_training_metrics(results.results_dict)
    log_run_artifacts(PROJECT_DIR, RUN_NAME)
    log_model(MODEL_NAME, RUN_NAME, PROJECT_DIR)
