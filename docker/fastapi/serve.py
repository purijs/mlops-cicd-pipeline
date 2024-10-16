import ray
import os
from ray import serve
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from typing import List

RAY_ADDRESS = os.getenv("RAY_ADDRESS")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Initialize Ray
ray.init(address=RAY_ADDRESS, ignore_reinit_error=True,
         _system_config={"allow_multiple": True},
         runtime_env={
             "pip": ["mlflow", "joblib", "scikit-learn", "pandas", "s3fs"]
         })
# Start Ray Serve
serve.start()

# Define the Ray Serve deployment for inference


@serve.deployment(
    name="ModelInference",
    ray_actor_options={
        "runtime_env": {
            "env_vars": {
                "AWS_ACCESS_KEY_ID": "virtualminds",
                "AWS_SECRET_ACCESS_KEY": "virtualminds",
                "MLFLOW_S3_ENDPOINT_URL": "http://minio.minio.svc.cluster.local:9000",
                "MLFLOW_S3_IGNORE_TLS": "true",
                "MLFLOW_ARTIFACTS_DESTINATION": "s3://mlops/models"
            },
            "pip": [
                "mlflow",
                "joblib",
                "scikit-learn",
                "pandas",
                "s3fs",
                "redis",
                "fastapi",
                "uvicorn",
                "botocore"
            ]
        }
    }
)
class ModelInference:
    def __init__(self, MLFLOW_TRACKING_URI: str):
        # Set MLflow tracking URI
        self.mlflow_tracking_uri = MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        self.mlflow_client = MlflowClient(self.mlflow_tracking_uri)
        self.model = None
        # Load the model upon initialization
        self.load_latest_model()

    def load_latest_model(self):
        latest_versions = self.mlflow_client.get_latest_versions("mlops_model")
        model_version = latest_versions[0].version
        model_uri = f"models:/mlops_model/{model_version}"
        self.model = mlflow.pyfunc.load_model(model_uri)
        print(f"Loaded model version: {model_version}")

    async def __call__(self, request):
        input_df = pd.DataFrame([request])
        prediction = self.model.predict(input_df)
        return {"prediction": prediction.tolist()}


# Use `serve.run()` to deploy the application
app = ModelInference.bind(MLFLOW_TRACKING_URI)
serve.run(app)

if __name__ == "__main__":
    print("Ray Serve app is running.")
    while True:
        pass
