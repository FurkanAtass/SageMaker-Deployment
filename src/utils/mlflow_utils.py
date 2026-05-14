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