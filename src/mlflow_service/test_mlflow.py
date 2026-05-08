import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from mlflow_service.mlflow_functions import get_client

load_dotenv()
_tracking_uri = os.environ["MLFLOW_TRACKING_URI"]


def test_mlflow_connection():
    print(f"Connecting to MLflow")
    try:
        client = get_client()
        print(f"Connection successful.")
    except Exception as e:
        print(f"Connection failed: {e}")


if __name__ == "__main__":
    test_mlflow_connection()
    
