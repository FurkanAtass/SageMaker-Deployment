from typing import Any
import mlflow

class YOLOWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from ultralytics import YOLO
        self.model = YOLO(context.artifacts["weights"])

    def predict(self, model_input: list[str], params: dict[str, Any] | None = None):
        return self.model.predict(model_input, **(params or {}))


mlflow.models.set_model(YOLOWrapper())