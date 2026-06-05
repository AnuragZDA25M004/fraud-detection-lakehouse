"""
dags/fraud_detection_dag.py
----------------------------
Airflow DAG — orchestrates the full fraud detection pipeline daily.

Schedule: runs every day at 02:00 UTC
Tasks:
  1. feature_engineering  — Spark batch job (raw → features Delta table)
  2. ml_training          — Spark MLlib GBT training + MLflow logging
  3. check_model_registry — confirms best model is registered in MLflow

Place this file in the dags/ folder.
Airflow picks it up automatically within 30 seconds.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import mlflow

# ── Default args ──────────────────────────────────────────────────────────────
default_args = {
    "owner": "fraud-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ── Spark submit command template ─────────────────────────────────────────────
SPARK_SUBMIT = """
docker exec spark-master spark-submit \
  --master spark://spark-master:7077 \
  --packages io.delta:delta-spark_2.12:3.1.0,org.apache.hadoop:hadoop-aws:3.3.4 \
  {script_path}
"""

FEATURE_SCRIPT = "/opt/bitnami/spark/work-dir/spark_jobs/spark_feature_engineering.py"
TRAINING_SCRIPT = "/opt/bitnami/spark/work-dir/ml/spark_ml_training.py"

# ── Model registry check ──────────────────────────────────────────────────────
def check_model_registered():
    """Confirms the best model run is registered in MLflow Model Registry."""
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.get_latest_versions("FraudDetection-GBT")
        if not versions:
            raise ValueError("No model versions found in registry!")
        latest = versions[-1]
        print(f"[OK] Model registered — version {latest.version}, "
              f"stage: {latest.current_stage}")
        return latest.version
    except Exception as e:
        raise RuntimeError(f"Model registry check failed: {e}")

# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="fraud_detection_pipeline",
    description="Daily fraud detection pipeline: feature engineering + ML training",
    default_args=default_args,
    schedule_interval="0 2 * * *",  # 02:00 UTC daily
    start_date=datetime(2026, 3, 1),
    catchup=False,
    tags=["fraud", "bigdata", "mlops"],
) as dag:

    # Task 1 — Feature Engineering
    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command=SPARK_SUBMIT.format(script_path=FEATURE_SCRIPT),
        dag=dag,
    )

    # Task 2 — ML Training
    ml_training = BashOperator(
        task_id="ml_training",
        bash_command=SPARK_SUBMIT.format(script_path=TRAINING_SCRIPT),
        dag=dag,
    )

    # Task 3 — Check model registry
    check_registry = PythonOperator(
        task_id="check_model_registry",
        python_callable=check_model_registered,
        dag=dag,
    )

    # ── Pipeline order ────────────────────────────────────────────────────────
    feature_engineering >> ml_training >> check_registry
