"""
spark_streaming_to_minio.py
----------------------------
Reads the fraud-transactions Kafka topic and writes to MinIO
as a Delta Lake table (the 'raw' lakehouse layer).

Run from your Spark container or spark-submit:

    spark-submit \
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,\
io.delta:delta-spark_2.12:3.1.0,\
org.apache.hadoop:hadoop-aws:3.3.4 \
        spark_streaming_to_minio.py

Environment variables expected (set in .env / docker-compose):
    MINIO_ENDPOINT     e.g. http://minio:9000
    MINIO_ACCESS_KEY   e.g. minioadmin
    MINIO_SECRET_KEY   e.g. minioadmin
    KAFKA_BROKER       e.g. kafka:9092
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_timestamp
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, IntegerType, LongType
)

# ── Config from environment ───────────────────────────────────────────────────
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "bigdata123")
KAFKA_BROKER     = os.getenv("KAFKA_BROKER",     "kafka:9092")
KAFKA_TOPIC      = "fraud-transactions"
DELTA_RAW_PATH   = "s3a://warehouse/raw/transactions"
CHECKPOINT_PATH  = "s3a://warehouse/checkpoints/raw_transactions"

# ── Spark session ─────────────────────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("FraudDetection-RawIngestion")
    .config("spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    # MinIO / S3A config
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

# ── Schema — core IEEE-CIS fields (extend as needed) ─────────────────────────
# We parse the most important columns explicitly;
# remaining 400+ features arrive as strings and are cast downstream.
transaction_schema = StructType([
    StructField("TransactionID",  LongType(),   True),
    StructField("TransactionDT",  LongType(),   True),   # seconds offset
    StructField("TransactionAmt", DoubleType(), True),
    StructField("ProductCD",      StringType(), True),
    StructField("card1",          IntegerType(), True),
    StructField("card2",          DoubleType(), True),
    StructField("card3",          DoubleType(), True),
    StructField("card4",          StringType(), True),
    StructField("card5",          DoubleType(), True),
    StructField("card6",          StringType(), True),
    StructField("addr1",          DoubleType(), True),
    StructField("addr2",          DoubleType(), True),
    StructField("dist1",          DoubleType(), True),
    StructField("dist2",          DoubleType(), True),
    StructField("P_emaildomain",  StringType(), True),
    StructField("R_emaildomain",  StringType(), True),
    StructField("isFraud",        IntegerType(), True),   # label
])

# ── Read from Kafka ───────────────────────────────────────────────────────────
raw_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BROKER)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "earliest")
    .option("failOnDataLoss", "false")
    .load()
)

# Kafka delivers bytes; decode value as JSON string, then parse schema
parsed = (
    raw_stream
    .selectExpr("CAST(value AS STRING) AS json_str",
                "timestamp AS kafka_timestamp")
    .select(
        from_json(col("json_str"), transaction_schema).alias("data"),
        col("kafka_timestamp")
    )
    .select("data.*", "kafka_timestamp")
    # Add ingestion timestamp for lineage
    .withColumn("ingested_at", current_timestamp())
)

# ── Write to MinIO as Delta Lake (raw layer) ──────────────────────────────────
query = (
    parsed.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .option("mergeSchema", "true")
    # Partition by fraud label for fast downstream filtering
    .partitionBy("isFraud")
    .start(DELTA_RAW_PATH)
)

print(f"[INFO] Streaming Kafka → Delta Lake at {DELTA_RAW_PATH}")
print("[INFO] Press Ctrl+C to stop.")
query.awaitTermination()
