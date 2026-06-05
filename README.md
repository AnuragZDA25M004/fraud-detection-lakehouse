# Real-Time Fraud Detection — Lakehouse & MLOps Platform

**Z5008 Big Data Lab · IIT Madras Zanzibar · Even Semester 2026**

| Team Member | Roll Number | Responsibility |
|---|---|---|
| Anurag Roychowdhury | ZDA25M004 | Kafka, Spark jobs, MLlib training |
| Sreejita Roy | ZDA25M008 | MinIO/Delta Lake, Airflow DAGs, MLflow, BentoML |

> **AI tool declaration:** GitHub Copilot and Claude (Anthropic) were used for code assistance. All AI-assisted code is fully understood and explainable by both team members.

---

## Project Overview

An end-to-end, production-grade real-time fraud detection system that:
- Streams 590,540 IEEE-CIS credit card transactions through **Apache Kafka** at 200 tx/sec
- Stores raw and feature-engineered data in a **Delta Lake lakehouse on MinIO**
- Trains a **Gradient Boosted Tree classifier** via Spark MLlib with **MLflow** experiment tracking
- Serves predictions via a **BentoML REST API** in Docker (`POST /predict → {label: FRAUD, confidence: 0.94}`)
- Monitors all layers via **Grafana + Prometheus** dashboards

---

## Architecture

```
IEEE-CIS Dataset (590K transactions)
         ↓
  Kafka Producer (200 tx/sec)
         ↓
  Kafka Topic: fraud-transactions
         ↓
  Spark Structured Streaming          ← Layer 1 + 2
         ↓
  MinIO — raw/transactions/           ← Delta Lake (ACID)
         ↓
  Spark Batch Feature Engineering     ← Layer 3
         ↓
  MinIO — features/transactions/
         ↓
  Airflow DAG (daily schedule)        ← Layer 4
         ↓
  Spark MLlib GBTClassifier           ← Layer 5
         ↓
  MLflow Experiment Tracking          ← Layer 6
         ↓
  BentoML REST API (Docker)           ← Layer 7
         ↓
  Grafana + Prometheus                ← Layer 8
```

---

## Prerequisites

- Docker Desktop (16GB RAM recommended)
- Python 3.10+
- Git

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/AnuragZDA25M004/fraud-detection-lakehouse
cd fraud-detection-lakehouse
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env if you want to change passwords
```

### 3. Start all services

```bash
docker-compose up -d
docker-compose ps   # verify all containers are running
```

### 4. Verify service URLs

| Service | URL | Credentials |
|---|---|---|
| JupyterLab | http://localhost:8888 | token: `bigdata` |
| Spark UI | http://localhost:8080 | — |
| MinIO Console | http://localhost:9001 | admin / bigdata123 |
| MLflow | http://localhost:5000 | — |
| Airflow | http://localhost:8090 | admin / admin |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| Kafka UI | http://localhost:8085 | — |

---

## Running the Pipeline

### Step 1 — Download the dataset

Download `train_transaction.csv` from [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/competitions/ieee-fraud-detection/data) and place it in the `data/` folder.

### Step 2 — Start the Kafka producer (from Jupyter terminal)

```bash
pip install kafka-python pandas
python /home/jovyan/work/kafka_producer.py \
  --csv /home/jovyan/work/data/train_transaction.csv \
  --broker kafka:9092 \
  --rate 200
```

### Step 3 — Start Spark Structured Streaming

```bash
MSYS_NO_PATHCONV=1 docker exec spark-master spark-submit \
  --master spark://spark-master:7077 \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3,\
io.delta:delta-spark_2.12:3.1.0,\
org.apache.hadoop:hadoop-aws:3.3.4 \
  /opt/bitnami/spark/work-dir/spark_jobs/spark_streaming_to_minio.py
```

### Step 4 — Run feature engineering batch job

```bash
MSYS_NO_PATHCONV=1 docker exec spark-master spark-submit \
  --master spark://spark-master:7077 \
  --packages io.delta:delta-spark_2.12:3.1.0,org.apache.hadoop:hadoop-aws:3.3.4 \
  /opt/bitnami/spark/work-dir/spark_jobs/spark_feature_engineering.py
```

### Step 5 — Train ML model (5 MLflow runs)

```bash
MSYS_NO_PATHCONV=1 docker exec spark-master spark-submit \
  --master spark://spark-master:7077 \
  --packages io.delta:delta-spark_2.12:3.1.0,org.apache.hadoop:hadoop-aws:3.3.4 \
  /opt/bitnami/spark/work-dir/ml/spark_ml_training.py
```

### Step 6 — Start BentoML API

```bash
cd api/
docker build -t fraud-api .
docker run -p 3001:3000 --network bigdata-net fraud-api
```

Test the API:
```bash
curl -X POST http://localhost:3001/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 1500.0, "card1": 9500, "card2": 321.0,
       "card3": 150.0, "card4": 0, "card5": 226.0, "card6": 1,
       "addr1": 299.0, "addr2": 87.0, "dist1": 0.0,
       "ProductCD": 0, "tx_hour": 2, "is_high_value": 1, "email_match": 0}'
```

### Step 7 — Airflow DAG

The pipeline is also scheduled via Airflow. Access at http://localhost:8090, enable the `fraud_detection_pipeline` DAG and trigger a run.

---

## Repository Structure

```
fraud-detection-lakehouse/
│
├── docker-compose.yml              # All 8 services
├── .env.example                    # Environment variable template
├── README.md
│
├── tests/
│   ├── test_pipeline.py            # Unit tests
│
├── spark_jobs/                     # Production Spark Python files
│   ├── spark_streaming_to_minio.py # Layer 1+2: Kafka → Delta Lake
│   └── spark_feature_engineering.py# Layer 3: Batch feature engineering
│
├── ml/                             # ML training scripts
│   └── spark_ml_training.py        # Layer 5+6: GBT + MLflow tracking
│
├── notebooks/                      # Jupyter notebooks
│   ├── kafka_producer.py           # Dataset replay producer
│   └── PR1_verification.ipynb      # PR1 end-to-end verification
│
├── dags/                           # Airflow pipeline definitions
│   └── fraud_detection_dag.py      # Layer 4: Daily orchestration DAG
│
├── api/                            # BentoML serving layer
│   ├── service.py                  # Layer 7: REST API definition
│   └── Dockerfile                  # API container
│
├── dashboards/                     # Grafana dashboard exports
│   └── fraud_detection_dashboard.json
│
├── data/                           # Sample data and generators
│   └── generate_sample.py          # Synthetic data generator (no Kaggle needed)
│
└── config/                         # Service config files
    ├── prometheus.yml
    ├── grafana-datasources.yml
    ├── spark-defaults.conf
    └── init-db.sql
```

---

## Unit Tests

```bash
pip install pytest pyspark delta-spark
pytest tests/ -v
```

---

## Stopping the Stack

```bash
docker-compose down        # stops containers, keeps data
docker-compose down -v     # stops containers AND deletes all data
```
