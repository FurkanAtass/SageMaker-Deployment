import os
from typing import Any

import mlflow
from ultralytics import YOLO

from src.models.base_model import BaseModel
from src.utils.mlflow_utils import (
    get_model_version,
    log_training_params,
)

WRAPPER_PATH = "src/models/cloth_detection/yolo_wrapper.py"


def _sanitize_metric_name(name: str) -> str:
    return name.replace("(", "_").replace(")", "")


def _find_latest_run_dir(project_dir: str, run_name: str) -> str | None:
    if not os.path.isdir(project_dir):
        return None
    candidates = [run_name]
    for entry in os.scandir(project_dir):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith(run_name + "-"):
            suffix = name[len(run_name) + 1:]
            if suffix.isdigit():
                candidates.append(name)
    candidates.sort(key=lambda n: int(n[len(run_name) + 1:]) if n != run_name else 0)
    for candidate in reversed(candidates):
        path = os.path.join(project_dir, candidate)
        if os.path.isdir(path):
            return path
    return None


class ClothDetectionYOLO(BaseModel):
    experiment_name = "Cloth Detection"
    model_name = "yolo26n-deepfashion2"          # artifact path inside the run
    registered_model_name = "cloth-detection"    # name in the MLflow Model Registry
    run_name = "deepfashion2-1k-yolo26n"

    def __init__(self, **config: Any) -> None:
        super().__init__(**config)
        self.model: YOLO | None = None
        self.current_model_version: str | None = None

    def _make_epoch_callback(self):
        def on_train_epoch_end(trainer):
            epoch = trainer.epoch + 1
            metrics = {
                **trainer.label_loss_items(trainer.tloss, prefix="train"),
                **trainer.metrics,
            }
            self.log_training_metrics(metrics, step=epoch)
        return on_train_epoch_end

    def build(self) -> None:
        self.model = YOLO(self.config["base_weights"])
        self.model.add_callback("on_train_epoch_end", self._make_epoch_callback())

    def log_training_params(self) -> None:
        log_training_params({
            "model": self.config["base_weights"],
            "batch": self.config["batch"],
            "epochs": self.config["epochs"],
            "imgsz": self.config["imgsz"],
            "dataset": self.config["dataset"],
        })

    def train(self) -> Any:
        if self.model is None:
            raise RuntimeError("Call build() before train().")
        return self.model.train(
            data=self.config["data"],
            batch=self.config["batch"],
            epochs=self.config["epochs"],
            imgsz=self.config["imgsz"],
            name=self.run_name,
            project=self.config["project_dir"],
        )

    def log_training_metrics(self, results: Any, step: int | None = None) -> None:
        if hasattr(results, "results_dict"):
            metrics = results.results_dict
        else:
            metrics = results
        mlflow.log_metrics(
            {_sanitize_metric_name(k): float(v) for k, v in metrics.items()},
            step=step,
        )

    def log_run_artifacts(self) -> None:
        project_dir = self.config["project_dir"]
        latest_run_dir = _find_latest_run_dir(project_dir, self.run_name)
        if latest_run_dir is None:
            print(f"Warning: No run directory found for '{self.run_name}' in {project_dir}. Skipping artifact logging.")
            return
        print(f"Logging artifacts from: {latest_run_dir}")
        for root, dirs, files in os.walk(latest_run_dir):
            dirs[:] = [d for d in dirs if d != "weights"]
            for file in files:
                file_path = os.path.join(root, file)
                rel_dir = os.path.relpath(root, latest_run_dir)
                artifact_path = None if rel_dir == "." else rel_dir
                mlflow.log_artifact(file_path, artifact_path=artifact_path)

    def log_model(self) -> None:
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run.")
        project_dir = self.config["project_dir"]
        latest_run_dir = _find_latest_run_dir(project_dir, self.run_name)
        if latest_run_dir is None:
            raise RuntimeError(
                f"No run directory found for '{self.run_name}' in {project_dir}. Cannot log model."
            )
        weights_path = f"{latest_run_dir}/weights/best.pt"
        mlflow.pyfunc.log_model(
            name=self.model_name,
            python_model=WRAPPER_PATH,
            artifacts={"weights": weights_path},
        )

    def load_model(self) -> Any:
        env = self.config["env"]
        version = get_model_version(self.registered_model_name, env)
        self.current_model_version = version
        model_uri = f"models:/{self.registered_model_name}/{version}"
        return mlflow.pyfunc.load_model(model_uri=model_uri)

    def update_model_version(self):
        env = self.config["env"]
        version = get_model_version(self.registered_model_name, env)
            
        if version != self.current_model_version:
            print(f"Model version updated from {self.current_model_version} to {version}. Reloading model.")
            self.current_model_version = version
            self.model = self.load_model()
            self.model = self.model.unwrap_python_model().model


    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        if self.model is None:
            self.model = self.load_model()
            self.model = self.model.unwrap_python_model().model
        return self.model.predict(inputs, **kwargs)
