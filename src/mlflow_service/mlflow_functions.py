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


def log_training_metrics(results_dict: dict) -> None:
    mlflow.log_metrics({k: float(v) for k, v in results_dict.items()})


def log_model_artifact(weights_path: str) -> None:
    if os.path.exists(weights_path):
        mlflow.log_artifact(weights_path, artifact_path="weights")
    else:
        print(f"Warning: Weights file {weights_path} does not exist. Skipping artifact logging.")
