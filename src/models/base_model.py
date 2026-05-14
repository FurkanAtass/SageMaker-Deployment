from abc import ABC, abstractmethod
from typing import Any

from src.utils.mlflow_utils import setup_experiment, start_run


class BaseModel(ABC):
    experiment_name: str
    model_name: str
    run_name: str

    def __init__(self, **config: Any) -> None:
        self.config = config

    @abstractmethod
    def build(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_training_params(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def train(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def log_training_metrics(self, results: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_run_artifacts(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_model(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def fit_and_log(self) -> None:
        setup_experiment(self.experiment_name)
        with start_run(run_name=self.run_name):
            self.log_training_params()
            self.build()
            results = self.train()
            self.log_training_metrics(results)
            self.log_run_artifacts()
            self.log_model()
