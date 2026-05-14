import os
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from typing import Any

load_dotenv()

_tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
mlflow.set_tracking_uri(_tracking_uri)


def get_client() -> MlflowClient:
    return MlflowClient(tracking_uri=_tracking_uri)


def get_or_create_experiment(name: str) -> str:
    client = get_client()
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        return client.create_experiment(name)
    return experiment.experiment_id


def setup_experiment(name: str) -> None:
    get_or_create_experiment(name)
    mlflow.set_experiment(name)


def log_training_params(params: dict) -> None:
    mlflow.log_params(params)


def _sanitize_metric_name(name: str) -> str:
    return name.replace("(", "_").replace(")", "")


def log_training_metrics(results_dict: dict, step: int | None = None) -> None:
    mlflow.log_metrics(
        {_sanitize_metric_name(k): float(v) for k, v in results_dict.items()},
        step=step,
    )


def make_epoch_callback():
    def on_train_epoch_end(trainer):
        epoch = trainer.epoch + 1
        metrics = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics}
        log_training_metrics(metrics, step=epoch)

    return on_train_epoch_end


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


def log_run_artifacts(project_dir: str, run_name: str) -> None:
    latest_run_dir = _find_latest_run_dir(project_dir, run_name)
    if latest_run_dir is None:
        print(f"Warning: No run directory found for '{run_name}' in {project_dir}. Skipping artifact logging.")
        return
    print(f"Logging artifacts from: {latest_run_dir}")
    for root, dirs, files in os.walk(latest_run_dir):
        dirs[:] = [d for d in dirs if d != "weights"]
        for file in files:
            file_path = os.path.join(root, file)
            rel_dir = os.path.relpath(root, latest_run_dir)
            artifact_path = None if rel_dir == "." else rel_dir
            mlflow.log_artifact(file_path, artifact_path=artifact_path)


def log_model(model_name: str, run_name: str, project_dir: str) -> None:
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("No active MLflow run.")

    latest_run_dir = _find_latest_run_dir(project_dir, run_name)
    if latest_run_dir is None:
        raise RuntimeError(f"No run directory found for '{run_name}' in {project_dir}. Cannot log model.")
    
    weights_path = f"{latest_run_dir}/weights/best.pt"
    mlflow.pyfunc.log_model(
        name=model_name,
        python_model="src/mlflow_service/yolo_wrapper.py",
        artifacts={"weights": weights_path},
    )


def start_run(run_name: str = ""):
    return mlflow.start_run(run_name=run_name)


def get_model_version(model_name: str, env: str) -> str:
    client = get_client()
    versions = client.search_model_versions(f"name='{model_name}'")

    model_version = [version for version in versions if version.tags.get(env) == "True"]
    if len(model_version) == 0:
        raise ValueError(f"No model version found for model '{model_name}' with tag '{env}=True'.")
    if len(model_version) > 1:
        print(f"Multiple model versions found for model '{model_name}' with tag '{env}=True'.")
    
    return model_version[0].version


def load_model(model_name: str, env: str):
    version = get_model_version(model_name, env)
    model_uri = f"models:/{model_name}/{version}"
    return mlflow.pyfunc.load_model(model_uri=model_uri)