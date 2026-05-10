import os
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

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


def log_training_metrics(results_dict: dict) -> None:
    mlflow.log_metrics({_sanitize_metric_name(k): float(v) for k, v in results_dict.items()})


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
    # Sort: base name first, then by numeric suffix
    candidates.sort(key=lambda n: int(n[len(run_name) + 1:]) if n != run_name else 0)
    for candidate in reversed(candidates):
        path = os.path.join(project_dir, candidate)
        if os.path.isdir(path):
            return path
    return None


def log_run_artifacts(project_dir: str, run_name: str) -> None:
    run_dir = _find_latest_run_dir(project_dir, run_name)
    if run_dir is None:
        print(f"Warning: No run directory found for '{run_name}' in {project_dir}. Skipping artifact logging.")
        return
    print(f"Logging artifacts from: {run_dir}")
    for root, _, files in os.walk(run_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_dir = os.path.relpath(root, run_dir)
            artifact_path = None if rel_dir == "." else rel_dir
            mlflow.log_artifact(file_path, artifact_path=artifact_path)

def start_run(run_name: str = ""):
    return mlflow.start_run(run_name=run_name)