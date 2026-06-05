"""
spark_feature_engineering.py
-----------------------------
Batch job: reads raw Delta Lake table from MinIO,
engineers features, writes cleaned feature table back to MinIO.

Place in notebooks/ folder and run via:
    MSYS_NO_PATHCONV=1 docker exec spark-master spark-submit \
      --master spark://spark-master:7077 \
      --packages io.delta:delta-spark_2.12:3.1.0,org.apache.hadoop:hadoop-aws:3.3.4 \
      /opt/bitnami/spark/work-dir/spark_feature_engineering.py
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, mean, stddev, count, isnan, isnull,
    log1p, abs as spark_abs
)

# ── Config ────────────────────────────────────────────────────────────────────
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "bigdata123")

RAW_PATH     = "s3a://warehouse/raw/transactions"
FEATURE_PATH = "s3a://warehouse/features/transactions"

# ── Spark session ─────────────────────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("FraudDetection-FeatureEngineering")
    .config("spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.hadoop.fs.s3a.endpoint",            MINIO_ENDPOINT)
    .config("spark.hadoop.fs.s3a.access.key",          MINIO_ACCESS_KEY)
    .config("spark.hadoop.fs.s3a.secret.key",          MINIO_SECRET_KEY)
    .config("spark.hadoop.fs.s3a.path.style.access",   "true")
    .config("spark.hadoop.fs.s3a.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print("[INFO] Spark session created")

# ── 1. Read raw Delta table ───────────────────────────────────────────────────
print(f"[INFO] Reading raw Delta table from {RAW_PATH} ...")
df = spark.read.format("delta").load(RAW_PATH)
raw_count = df.count()
print(f"[INFO] Raw row count: {raw_count:,}")

# ── 2. Select core columns ────────────────────────────────────────────────────
# Keep the most predictive columns from IEEE-CIS to avoid OOM
core_cols = [
    "TransactionID", "TransactionDT", "TransactionAmt",
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1",
    "P_emaildomain", "R_emaildomain",
    "isFraud"
]
df = df.select([c for c in core_cols if c in df.columns])

# ── 3. Drop rows where label is null ─────────────────────────────────────────
df = df.filter(col("isFraud").isNotNull())

# ── 4. Feature engineering ───────────────────────────────────────────────────

# 4a. Log-transform transaction amount (reduces skewness)
df = df.withColumn("log_TransactionAmt", log1p(col("TransactionAmt")))

# 4b. Transaction hour of day (proxy — TransactionDT is seconds from reference)
df = df.withColumn("tx_hour", (col("TransactionDT") / 3600 % 24).cast("int"))

# 4c. Is it a high-value transaction? (above $500)
df = df.withColumn("is_high_value",
    when(col("TransactionAmt") > 500, 1).otherwise(0))

# 4d. Email domain match — same sender/receiver domain is lower risk
df = df.withColumn("email_match",
    when(col("P_emaildomain") == col("R_emaildomain"), 1).otherwise(0))

# 4e. Fill numeric nulls with -999 (sentinel for GBT)
numeric_cols = ["card1", "card2", "card3", "card5",
                "addr1", "addr2", "dist1",
                "log_TransactionAmt", "tx_hour"]
for c in numeric_cols:
    if c in df.columns:
        df = df.withColumn(c, when(col(c).isNull(), -999).otherwise(col(c)))

# 4f. Encode categorical columns as integer index (simple label encoding)
cat_map = {
    "ProductCD": {"W": 0, "H": 1, "C": 2, "S": 3, "R": 4},
    "card4":     {"visa": 0, "mastercard": 1, "american express": 2, "discover": 3},
    "card6":     {"debit": 0, "credit": 1, "debit or credit": 2, "charge card": 3},
}
for col_name, mapping in cat_map.items():
    if col_name in df.columns:
        expr = when(col(col_name).isNull(), -1)
        for val, idx in mapping.items():
            expr = expr.when(col(col_name) == val, idx)
        df = df.withColumn(col_name, expr.otherwise(-1))

# Fill remaining nulls in all remaining columns
df = df.fillna(-999)

# ── 5. Final feature columns ──────────────────────────────────────────────────
feature_cols = [
    "TransactionAmt", "log_TransactionAmt", "tx_hour",
    "is_high_value", "email_match",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1",
    "ProductCD", "isFraud"
]
feature_cols = [c for c in feature_cols if c in df.columns]
df_features = df.select(feature_cols)

# ── 6. Print summary ──────────────────────────────────────────────────────────
feature_count = df_features.count()
fraud_count   = df_features.filter(col("isFraud") == 1).count()
legit_count   = feature_count - fraud_count
print(f"[INFO] Feature table rows : {feature_count:,}")
print(f"[INFO] Fraud transactions  : {fraud_count:,} "
      f"({100*fraud_count/feature_count:.2f}%)")
print(f"[INFO] Legit transactions  : {legit_count:,}")

# ── 7. Write feature Delta table ──────────────────────────────────────────────
print(f"[INFO] Writing feature table to {FEATURE_PATH} ...")
(
    df_features.write
    .format("delta")
    .mode("overwrite")
    .save(FEATURE_PATH)
)
print("[DONE] Feature engineering complete ✅")
spark.stop()
