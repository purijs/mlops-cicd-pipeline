import os
import logging
from typing import Optional, Dict, List, Any

import ray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlflow.tracking import MlflowClient

import mlflow
import joblib, time, json, redis
import pandas as pd
from ray.util.joblib import register_ray
from ray import serve

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi.responses import StreamingResponse
from datetime import datetime, timedelta
import uuid
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from minio import Minio
from minio.error import S3Error

app = FastAPI(title="MLOps FastAPI with Ray Jobs and MLflow")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Environment Variables
RAY_ADDRESS = os.getenv("RAY_ADDRESS")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = "mlops"
DATA_PATH = "s3://mlops/data/iris.csv"

ray.init(address=RAY_ADDRESS, ignore_reinit_error=True,
         _system_config={"allow_multiple": True},
         runtime_env={
             "pip": ["mlflow", "joblib", "scikit-learn", "pandas", "s3fs"]
         })

# Pydantic Models
class ScheduleTrainingRequest(BaseModel):
    minutes: int
    hyperparameters: Dict[str, Any]

class InferenceRequest(BaseModel):
    input_data: List[float]  # Define according to your model's input
    model_version: Optional[str] = None  # e.g., "1"
    retries: Optional[int] = 3
    sla_seconds: Optional[int] = 60

class TrainingRequest(BaseModel):
    hyperparameters: Dict[str, Any]

class WatchModelRequest(BaseModel):
    minutes: int

scheduler = BackgroundScheduler()
scheduler.start()

# Shared message queue for SSE notifications
redis_client = redis.StrictRedis(host='redis-master.db.svc.cluster.local', port=6379, db=0, decode_responses=True, password='redis')

# SSE endpoint
@app.get("/webhook")
async def stream_webhook():
    async def event_generator():
        while True:
            # Check if there are any messages in the queue
            message = redis_client.lpop('message_queue')
            if message:
                yield f"data: {message}\n\n"
            await asyncio.sleep(1)  # Prevent busy-waiting

    # Return a streaming response for Server-Sent Events (SSE)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# MinIo Init
client = Minio(
    endpoint=MINIO_ENDPOINT,
    secure=False,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY
  )

def init_mlflow(EXPERIMENT_NAME):
    mlflow_client = MlflowClient(MLFLOW_TRACKING_URI)
    experiment = mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow_client.create_experiment(EXPERIMENT_NAME, artifact_location='s3://mlops/models')
        logger.info(f"Created MLflow experiment: {EXPERIMENT_NAME}")
    elif experiment.lifecycle_stage == "deleted":
        mlflow_client.restore_experiment(experiment.experiment_id)
        logger.info(f"Restored MLflow experiment: {EXPERIMENT_NAME}")
    else:
        logger.info(f"MLflow experiment '{EXPERIMENT_NAME}' already exists.")

@ray.remote(num_cpus=1, memory=1000 * 1024 * 1024, runtime_env={
        "env_vars": {
            "AWS_ACCESS_KEY_ID": "virtualminds",
            "AWS_SECRET_ACCESS_KEY": "virtualminds",
            "MLFLOW_S3_ENDPOINT_URL": "http://minio.minio.svc.cluster.local:9000",
            "MLFLOW_S3_IGNORE_TLS": "true",
            "MLFLOW_ARTIFACTS_DESTINATION":"s3://mlops/models"
        }
    })
def train_model_remote(hyperparameters: Dict[str, Any], MLFLOW_TRACKING_URI: str, DEFINED_THRESHOLD: int):
    # Set MLflow tracking URI
    redis_client = redis.StrictRedis(host='redis-master.db.svc.cluster.local', port=6379, db=0, decode_responses=True, password='redis')

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow_client = MlflowClient(MLFLOW_TRACKING_URI)
    
    # Ensure the experiment exists and is active
    EXPERIMENT_NAME = "mlops_experiment"
    experiment = mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id

    # Set the experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load data from MinIO
    try:
        df = pd.read_csv('s3://mlops/data/iris.csv', storage_options={'key': 'virtualminds','secret': 'virtualminds','client_kwargs': {'endpoint_url': 'http://minio.minio.svc.cluster.local:9000'}})
        logger.info("Data loaded successfully from MinIO.")
    except Exception as e:
        logger.error(f"Failed to load data from MinIO: {e}")
        raise

    # Define target column
    target_column = 'variety'

    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id):
        # Log hyperparameters
        mlflow.log_params(hyperparameters)
        logger.info(f"Logged hyperparameters: {hyperparameters}")

        # Prepare data
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
        )

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        logger.info("Data preprocessing completed.")

        # Train the model using Ray's Joblib backend
        register_ray()
        with joblib.parallel_backend('ray'):
            model = RandomForestClassifier(**hyperparameters)
            model.fit(X_train, y_train)
            logger.info("Model training completed.")

        # Make predictions and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy >= DEFINED_THRESHOLD:
            message = json.dumps({
                "text": f"Latest model accuracy was: {accuracy}"
            })
            redis_client.rpush('message_queue', message)

        mlflow.log_metric("accuracy", accuracy)
        logger.info(f"Model accuracy: {accuracy}")

        # Log the model to MLflow Model Registry
        registered_model = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="mlops_model"
        )
        logger.info(f"Model registered response: {registered_model}")

        # Get the latest version of the registered model
        latest_versions = mlflow_client.get_latest_versions("mlops_model")
        if not latest_versions:
            raise mlflow.exceptions.MlflowException(f"No versions found for model 'mlops_model'")
        model_version_registered = latest_versions[0].version
        logger.info(f"Model version registered: {model_version_registered}")

'''
@ray.remote(num_cpus=0.25, memory=500 * 1024 * 1024)
def perform_inference_remote(input_data: List[float], MLFLOW_TRACKING_URI: str):
    """
    Remote function to perform inference using a specified model version.
    """
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow_client = MlflowClient(MLFLOW_TRACKING_URI)

    latest_versions = mlflow_client.get_latest_versions("mlops_model")
    model_version = latest_versions[0].version
    print('Model version: ', model_version)
    # Load model from MLflow Model Registry
    model_uri = f"models:/mlops_model/{model_version}"
    
    model = mlflow.pyfunc.load_model(model_uri)
    prediction = model.predict(pd.DataFrame([input_data]))

    return {"prediction": prediction.tolist()}
'''

@ray.remote(num_cpus=0.25, memory=100 * 1024 * 1024)
def watch_model(minutes: int, MLFLOW_TRACKING_URI: str):
    """
    Remote function to watch MLFlow models
    """

    redis_client = redis.StrictRedis(host='redis-master.db.svc.cluster.local', 
                                     port=6379, db=0, decode_responses=True, 
                                     password='redis')
    
    # Set MLflow tracking URI
    mlflow_client = MlflowClient(MLFLOW_TRACKING_URI)

    latest_versions = mlflow_client.get_latest_versions("mlops_model")
    timestamp_s = latest_versions[0].creation_timestamp / 1000

    timestamp_dt = datetime.utcfromtimestamp(timestamp_s)
    now = datetime.utcnow()

    # Calculate the time 'n' minutes ago
    time_n_minutes_ago = now - timedelta(minutes=minutes)

    # Check if the timestamp is within the last 'n' minutes
    print(timestamp_dt >= time_n_minutes_ago)
    if not timestamp_dt >= time_n_minutes_ago:
        message = json.dumps({
            "text": f"Model has not been updated in the last {minutes} minutes."
        })
        redis_client.rpush('message_queue', message)
        print({"status": "Notification added to Redis"})

    return {"status": f"Model updated within the last {minutes} minutes."}

@app.post("/trigger_training")
async def trigger_training(request: TrainingRequest):
    try:
        print('INPUTS ARE',request.hyperparameters)

        train_model_remote.remote(request.hyperparameters, MLFLOW_TRACKING_URI, 0.9)
        return {"status": "Training completed"}
    except Exception as e:
        logger.error(f"Training job failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference")
async def inference(request: InferenceRequest):
    retries = request.retries or 3
    sla = request.sla_seconds or 60
    attempt = 0
    start_time = time.time()

    while attempt < retries:
        try:
            handle = serve.get_app_handle("default")
            result = await handle.remote(request.input_data)
            return result
        except ray.exceptions.GetTimeoutError:
            attempt += 1
            logger.warning(f"Inference attempt {attempt} timed out.")
        except Exception as e:
            attempt += 1
            logger.warning(f"Inference attempt {attempt} failed with error: {e}")

        # Check if SLA has been exceeded
        elapsed_time = time.time() - start_time
        if elapsed_time > sla:
            break

    logger.error("Inference failed after multiple retries.")
    raise HTTPException(
        status_code=500, detail="Inference failed after multiple retries")

@app.post("/schedule_training")
async def schedule_training(request: ScheduleTrainingRequest):
    """
    Schedule a training request using APScheduler every x minutes.
    """
    DEFINED_THRESHOLD = 0.9

    def scheduled_training():
        logger.info(f"Triggering scheduled training with hyperparameters: {request.hyperparameters}")
        train_model_remote.remote(request.hyperparameters, MLFLOW_TRACKING_URI, DEFINED_THRESHOLD)
    
    run_id = str(uuid.uuid4())
    # Schedule the job to run every x minutes
    scheduler.add_job(scheduled_training, 'interval', minutes=request.minutes, id=run_id)
    logger.info(f"Scheduled training every {request.minutes} minutes")
    return {"status": f"Training scheduled every {request.minutes} minutes", "Cancellation ID ":run_id}

@app.post("/watch_model")
async def watch_mlflow(request: WatchModelRequest):
    """
    Schedule a watch job that monitors MLFlow models
    """
    def scheduled_watch_job():
        logger.info(f"Triggering watch job")
        watch_model.remote(request.minutes, MLFLOW_TRACKING_URI)
    
    run_id = str(uuid.uuid4())
    
    # Schedule the job to run every x minutes
    scheduler.add_job(scheduled_watch_job, 'interval', minutes=request.minutes, id=run_id)

    logger.info(f"Scheduled watch job for every {request.minutes} minutes")
    return {"status": f"Watch Job scheduled every {request.minutes} minutes", "Cancellation ID ":run_id}

@app.delete("/kill_scheduled_job/{job_id}")
async def kill_scheduled_job(job_id: str):
    """
    Kill or remove the scheduled job by its job ID.
    """
    try:
        scheduler.remove_job(job_id)
        logger.info(f"Job {job_id} successfully removed.")
        return {"status": f"Job {job_id} successfully removed."}
    except Exception as e:
        logger.error(f"Failed to remove job {job_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found or could not be removed.")

@app.get("/get_metrics")
async def get_metrics():
    """
    Get the metrics for the latest model and the average metrics for the last 7 days.
    """
    try:
        print(MLFLOW_TRACKING_URI)
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Get the latest model's accuracy
        runs_log = mlflow.search_runs(experiment_names=["mlops_experiment"], order_by=["start_time DESC"])
        runs_log['start_time'] = pd.to_datetime(runs_log['start_time'])
        runs_log['start_time'] = runs_log['start_time'].dt.tz_localize(None)
        
        # Filter to get runs in the last 7 days
        seven_days_ago = datetime.now() - timedelta(days=7)
        filtered_runs = runs_log[runs_log['start_time'] >= seven_days_ago]

        # Calculate the average accuracy for the last 7 days
        avg_accuracy_last_7_days = filtered_runs['metrics.accuracy'].mean()

        # Calculate average accuracy per day
        filtered_runs['date'] = filtered_runs['start_time'].dt.date
        avg_accuracy_per_day = filtered_runs.groupby('date')['metrics.accuracy'].mean().to_dict()

        # Formatting the datetime objects as string
        avg_accuracy_per_day = {str(date): acc for date, acc in avg_accuracy_per_day.items()}

        return {
            "weekly_average": avg_accuracy_last_7_days,
            "avg_accuracy_last_7_days": avg_accuracy_per_day
        }
    
    except Exception as e:
        logger.error(f"Failed to retrieve metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "Gesund!"}

init_mlflow("mlops_experiment")