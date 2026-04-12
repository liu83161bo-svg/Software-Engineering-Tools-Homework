# scripts/mlflow_setup.py
import mlflow


def setup_mlflow():
    mlflow.set_tracking_uri("file:./mlruns")  # 本地存储

    with mlflow.start_run():
        mlflow.log_param("data_version", "v1.0")
        mlflow.log_metric("num_samples", 320)

    print("MLflow tracking configured")