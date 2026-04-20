#!/usr/bin/env python3
"""
MLflow Demo for HW3 - Simple tracking example
This demonstrates MLflow integration without heavy dependencies
"""

import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
import os


def log_data_quality_metrics():
    """Log data quality metrics to MLflow"""

    # Set MLflow tracking URI (local directory)
    mlflow.set_tracking_uri("file:./mlruns")

    # Start a new MLflow run
    with mlflow.start_run(run_name="HW3_Data_Quality"):

        # Log parameters (data generation info)
        mlflow.log_param("dataset_name", "EEG_Age_Classification")
        mlflow.log_param("data_version", "v1.0")
        mlflow.log_param("samples_per_age", 20)
        mlflow.log_param("total_samples", 320)

        # Read the generated data
        data_path = Path(__file__).parent.parent / "data" / "sample_eeg_data.csv"
        if data_path.exists():
            df = pd.read_csv(data_path)

            # Calculate and log metrics
            mlflow.log_metric("num_samples", len(df))
            mlflow.log_metric("num_features", len(df.columns))
            mlflow.log_metric("unique_subjects", df['subject_hash'].nunique())
            mlflow.log_metric("mean_age", df['age'].mean())
            mlflow.log_metric("age_std", df['age'].std())

            # Log data quality metrics
            missing_values = df.isnull().sum().sum()
            mlflow.log_metric("missing_values", missing_values)

            # Count signal columns
            signal_cols = [col for col in df.columns if 'signal' in col]
            mlflow.log_metric("num_signal_points", len(signal_cols))

            # Log split information
            splits_dir = Path(__file__).parent.parent / "data" / "splits"
            if splits_dir.exists():
                for split_file in splits_dir.glob("*.txt"):
                    with open(split_file, 'r') as f:
                        num_subjects = len(f.readlines())
                        mlflow.log_metric(f"subjects_in_{split_file.stem}", num_subjects)

            # Log an artifact (data summary)
            summary_path = "data_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(f"Dataset: EEG Age Classification\n")
                f.write(f"Total samples: {len(df)}\n")
                f.write(f"Age range: {df['age'].min()} - {df['age'].max()}\n")
                f.write(f"Unique ages: {df['age'].nunique()}\n")
                f.write(f"Columns: {', '.join(df.columns)}\n")

            mlflow.log_artifact(summary_path)
            os.remove(summary_path)  # Clean up

            print("✓ Logged data quality metrics to MLflow")

            # Print run info
            run_id = mlflow.active_run().info.run_id
            print(f"MLflow Run ID: {run_id}")
            print(f"View results: mlflow ui --backend-store-uri ./mlruns")

        else:
            print("⚠ Data file not found, skipping detailed metrics")


if __name__ == "__main__":
    print("MLflow Demo - HW3 Data Quality Tracking")
    print("=" * 50)

    try:
        log_data_quality_metrics()
        print("\n MLflow demo completed successfully")
    except Exception as e:
        print(f"\n MLflow demo failed: {e}")
        # Don't fail the CI if MLflow demo fails
        print("This is a demo, continuing with CI...")