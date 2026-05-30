## Cloth Detection Inference on SageMaker with MLflow

This repository trains a YOLOv26n object detection model on the DeepFashion2 dataset, tracks experiments and manages model versions with SageMaker-managed MLflow, containerizes a FastAPI inference service, and deploys it to an AWS SageMaker endpoint. The running service polls MLflow on a configurable interval and hot-reloads the model when a new version is promoted to the active environment.

### Key Features
- **YOLO-based detection**: YOLOv26n nano trained on DeepFashion2 (13 clothing categories)
- **MLflow tracking**: per-epoch metrics, artifacts, and model versioning via SageMaker managed MLflow
- **Environment-tagged promotion**: model versions are promoted per environment using MLflow tags (`dev: True`, `prod: True`, etc.)
- **Hot-reload without restart**: running service detects a tag change in MLflow and swaps the model in-place
- **GPU inference**: CUDA 12.4 container deployed on `ml.g4dn.xlarge`

---

## Architecture Overview
- **Training** (`src/models/cloth_detection/train.py`): builds a `ClothDetectionYOLO` model, trains it with `ultralytics` YOLO, and logs params, per-epoch metrics, artifacts, and the model to MLflow.
- **Model Registry**: after training, the model artifact is registered in the MLflow Model Registry as `cloth-detection`. A version is promoted to an environment by setting a tag `<env>: True` on it.
- **Service** (`src/services/cloth_detection/app.py`): FastAPI app that loads the active model version for the configured `ENV`, exposes `/ping` and `/invocations`, and polls MLflow every `MODEL_UPDATE_INTERVAL` seconds for a version change.
- **Container** (`deploy.sh` + `Dockerfile`): builds a `linux/amd64` CUDA image and pushes it to ECR.
- **SageMaker endpoint** (`src/services/cloth_detection/sagemaker.ipynb`): creates a SageMaker model, endpoint config, and endpoint using the ECR image.

---

## Prerequisites
- AWS account with:
  - A SageMaker execution IAM role with SageMaker, ECR, and S3 permissions
  - A SageMaker-managed MLflow tracking server (Studio → MLflow)
  - GPU quota for `ml.g4dn.xlarge` in your region
- macOS/Linux shell with:
  - `aws` CLI configured with appropriate credentials
  - `docker` with `buildx` support
  - `uv` (installed in the steps below)
- DeepFashion2 dataset — download from [switchablenorms/DeepFashion2](https://github.com/switchablenorms/DeepFashion2)

---

## Repository Structure
- `src/models/cloth_detection/`: model class, training script, inference script, dataset converter, MLflow YOLO wrapper
- `src/services/cloth_detection/`: FastAPI app, `Dockerfile`, `sagemaker.ipynb` deployment notebook
- `src/utils/mlflow_utils.py`: MLflow client helpers (experiment setup, model version lookup by env tag)
- `deploy.sh`: builds and pushes the Docker image to ECR

---

## Environment Setup

Install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install project dependencies:
```bash
uv sync
```

Copy the environment template and fill in the values:
```bash
cp .template.env .env
```

`.env` variables:

| Variable | Description |
|---|---|
| `ENV` | Deployment environment (`dev`, `staging`, `prod`) |
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_DEFAULT_REGION` | AWS region (e.g. `us-east-1`) |
| `SAGEMAKER_EXECUTION_ROLE_ARN` | IAM role ARN for SageMaker |
| `MLFLOW_TRACKING_URI` | Tracking URI from your SageMaker-managed MLflow server |

To get the `MLFLOW_TRACKING_URI`, open SageMaker Studio → MLflow → select your tracking server → copy the tracking server URI.

---

## Prepare the Dataset

Download DeepFashion2 and place it at `src/models/cloth_detection/deepfashion2/` with the standard split structure (`train/`, `validation/`). Then convert it to YOLO format:

```bash
uv run python -m src.models.cloth_detection.deepfashion_to_yolo
```

This creates `src/models/cloth_detection/deepfashion2_yolo/` with YOLO-formatted labels and `dataset.yaml`. Training is capped at 5 000 train images and 500 validation images by default.

---

## Train the Model

```bash
uv run python -m src.models.cloth_detection.train
```

This starts a new MLflow run under the `Cloth Detection` experiment, logs training hyperparameters, per-epoch metrics, run artifacts (plots, confusion matrix), and the YOLO model as an MLflow pyfunc artifact.

---

## Register the Model

After training completes, open the MLflow UI (SageMaker Studio → MLflow → Experiments → `Cloth Detection`), find the run, and click **Register Model** on the `yolo26n-deepfashion2` artifact. Register it under the name `cloth-detection`.

Once registered, navigate to **Models → cloth-detection → `<version>`** and add a tag to promote it to your environment:

| Tag key | Tag value |
|---|---|
| `dev` | `True` |
| `prod` | `True` |

The service resolves the active model version by looking for the version tagged `<ENV>: True`. Only one version per environment should carry this tag at a time — set the new version's tag before removing the old one to avoid a gap during transitions.

---

## Build and Push the Inference Image

```bash
bash deploy.sh
```

The script:
1. Resolves your AWS account ID and region from the environment
2. Creates the `cloth-detection` ECR repository if it does not exist
3. Builds a `linux/amd64` image from `src/services/cloth_detection/Dockerfile`
4. Tags and pushes it as `<account>.dkr.ecr.<region>.amazonaws.com/cloth-detection:latest`

---

## Deploy to SageMaker

Open and run `src/services/cloth_detection/sagemaker.ipynb` cell by cell.

The notebook:
1. Creates a SageMaker model pointing to the ECR image and injects `ENV`, `MLFLOW_TRACKING_URI`, and AWS credentials as container environment variables
2. Creates an endpoint configuration on `ml.g4dn.xlarge`
3. Creates the endpoint and waits until it is `InService`
4. Runs a test inference against the live endpoint using a sample image
5. (Optional) contains a cleanup cell to delete the endpoint, config, and model

Ensure `SAGEMAKER_EXECUTION_ROLE_ARN` is set in your `.env` before running the notebook.

---

## Inference API

The service exposes two endpoints expected by SageMaker:

| Method | Path | Description |
|---|---|---|
| `GET` | `/ping` | Health check — returns `{"status": "healthy"}` |
| `POST` | `/invocations` | Runs cloth detection on an uploaded image |

Example request using the SageMaker runtime client:
```python
import boto3, json, uuid

runtime = boto3.client("sagemaker-runtime", region_name="<region>")

with open("/path/to/image.jpg", "rb") as f:
    file_bytes = f.read()

boundary = f"----Boundary{uuid.uuid4().hex}"
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="image.jpg"\r\n'
    f"Content-Type: image/jpeg\r\n\r\n"
).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()

response = runtime.invoke_endpoint(
    EndpointName="cloth-detection",
    ContentType=f"multipart/form-data; boundary={boundary}",
    Body=body,
)

result = json.loads(response["Body"].read().decode())
print(result)
```

Response:
```json
{
  "results": [
    {
      "boxes": { "...": "..." },
      "names": { "...": "..." }
    }
  ]
}
```

---

## Model Auto-Update

The service checks MLflow for a version change every `MODEL_UPDATE_INTERVAL` seconds (default: `300`). When it detects that the version tagged `<ENV>: True` has changed, it loads the new model in the background without restarting the container.

To promote a new model version:
1. Set `<env>: True` on the new version in the MLflow UI
2. Remove the tag from the old version

The running endpoint picks up the change on the next polling cycle. To adjust the poll interval, set `MODEL_UPDATE_INTERVAL` (in seconds) in the container environment when creating the SageMaker model in the notebook.

