"""
spark_ml_training.py
Trains GBTClassifier on the feature table and logs 5 runs to MLflow.
Uses only core mlflow (compatible with mlflow-skinny — no mlflow.spark needed).
"""

import sys
import mlflow


from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# ── Config ────────────────────────────────────────────────────────────────────
MINIO_ENDPOINT   = "http://minio:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "bigdata123"
MLFLOW_TRACKING  = "http://mlflow:5000"
FEATURES_PATH    = "s3a://warehouse/features/transactions"
MODEL_SAVE_PATH  = "s3a://warehouse/models/gbt_fraud"

# 5 experiment configurations
EXPERIMENT_CONFIGS = [
    {"maxDepth": 3, "maxIter": 10, "stepSize": 0.1, "subsamplingRate": 0.8},
    {"maxDepth": 5, "maxIter": 20, "stepSize": 0.1, "subsamplingRate": 0.8},
    {"maxDepth": 5, "maxIter": 20, "stepSize": 0.05, "subsamplingRate": 0.9},
    {"maxDepth": 7, "maxIter": 30, "stepSize": 0.1, "subsamplingRate": 0.8},
    {"maxDepth": 7, "maxIter": 30, "stepSize": 0.05, "subsamplingRate": 1.0},
]

# ── Spark Session ─────────────────────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("FraudDetection-GBT-Training")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.hadoop.fs.s3a.endpoint",          MINIO_ENDPOINT)
    .config("spark.hadoop.fs.s3a.access.key",        MINIO_ACCESS_KEY)
    .config("spark.hadoop.fs.s3a.secret.key",        MINIO_SECRET_KEY)
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print("[INFO] Spark session created ✅")

# ── Load Feature Table ─────────────────────────────────────────────────────────
print(f"[INFO] Reading feature table from {FEATURES_PATH} ...")
df = spark.read.format("delta").load(FEATURES_PATH)
total = df.count()
fraud = df.filter(F.col("isFraud") == 1).count()
print(f"[INFO] Total rows: {total:,}  |  Fraud: {fraud:,}  |  Legit: {total-fraud:,}")

# ── Feature Columns ────────────────────────────────────────────────────────────
feature_cols = [c for c in df.columns if c not in ("isFraud", "TransactionID", "TransactionDT")]
print(f"[INFO] Using {len(feature_cols)} features")

# ── MLflow Setup ───────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_TRACKING)
mlflow.set_experiment("FraudDetection-GBT")
print(f"[INFO] MLflow tracking URI: {MLFLOW_TRACKING}")

# ── Train/Test Split ───────────────────────────────────────────────────────────
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
test_df.cache()
print(f"[INFO] Train: {train_df.count():,} rows  |  Test: {test_df.count():,} rows")

# ── Evaluators ────────────────────────────────────────────────────────────────
auc_evaluator  = BinaryClassificationEvaluator(labelCol="isFraud", metricName="areaUnderROC")
aupr_evaluator = BinaryClassificationEvaluator(labelCol="isFraud", metricName="areaUnderPR")
f1_evaluator   = MulticlassClassificationEvaluator(labelCol="isFraud", metricName="f1")

# ── Run 5 Experiments ──────────────────────────────────────────────────────────
best_auc   = 0.0
best_run_id = None

for i, cfg in enumerate(EXPERIMENT_CONFIGS, start=1):
    print(f"\n[RUN {i}/5] maxDepth={cfg['maxDepth']}  maxIter={cfg['maxIter']}  "
          f"stepSize={cfg['stepSize']}  subsamplingRate={cfg['subsamplingRate']}")

    with mlflow.start_run(run_name=f"GBT_run_{i}"):
        # Log parameters
        mlflow.log_params(cfg)
        mlflow.log_param("train_rows", train_df.count())
        mlflow.log_param("test_rows",  test_df.count())
        mlflow.log_param("n_features", len(feature_cols))

        # Build pipeline
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features",
                                    handleInvalid="skip")
        scaler    = StandardScaler(inputCol="raw_features", outputCol="features",
                                   withStd=True, withMean=False)
        gbt       = GBTClassifier(
            labelCol="isFraud",
            featuresCol="features",
            maxDepth=cfg["maxDepth"],
            maxIter=cfg["maxIter"],
            stepSize=cfg["stepSize"],
            subsamplingRate=cfg["subsamplingRate"],
            seed=42,
        )
        pipeline = Pipeline(stages=[assembler, scaler, gbt])

        # Train
        model = pipeline.fit(train_df)

        # Evaluate
        predictions = model.transform(test_df)
        auc_roc = auc_evaluator.evaluate(predictions)
        auc_pr  = aupr_evaluator.evaluate(predictions)
        f1      = f1_evaluator.evaluate(predictions)

        # Log metrics
        mlflow.log_metric("auc_roc", round(auc_roc, 4))
        mlflow.log_metric("auc_pr",  round(auc_pr,  4))
        mlflow.log_metric("f1_score", round(f1,     4))

        print(f"  AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  F1={f1:.4f}")

        # Save model to MinIO and log path as tag
        model_path = f"{MODEL_SAVE_PATH}/run_{i}"
        model.write().overwrite().save(model_path)
        mlflow.set_tag("model_path", model_path)
        mlflow.set_tag("run_index", str(i))

        run_id = mlflow.active_run().info.run_id
        print(f"  MLflow run_id: {run_id}")

        if auc_roc > best_auc:
            best_auc    = auc_roc
            best_run_id = run_id
            best_model  = model
            best_model_path = model_path

print(f"\n[INFO] Best run: {best_run_id}  AUC-ROC={best_auc:.4f}")

# ── Register Best Model in MLflow ─────────────────────────────────────────────
# Log best model path to a dedicated "best model" run
with mlflow.start_run(run_name="GBT_BEST_MODEL"):
    mlflow.log_metric("best_auc_roc", round(best_auc, 4))
    mlflow.set_tag("best_run_id",    best_run_id)
    mlflow.set_tag("model_path",     best_model_path)
    mlflow.set_tag("status",         "STAGING")
    print(f"[INFO] Best model registered in MLflow under run 'GBT_BEST_MODEL' ✅")

print("\n[DONE] ML training complete ✅")
print(f"       Open http://localhost:5000 to see all 5 runs + best model.")
spark.stop()
