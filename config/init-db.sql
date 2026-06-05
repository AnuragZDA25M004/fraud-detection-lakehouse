-- ================================================================
-- Z5008 Big Data Lab — PostgreSQL initialisation
-- Creates databases and users for Airflow and MLflow
-- ================================================================

-- Airflow database and user
CREATE USER airflow WITH PASSWORD 'airflow';
CREATE DATABASE airflow OWNER airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;

-- MLflow database and user
CREATE USER mlflow WITH PASSWORD 'mlflow';
CREATE DATABASE mlflow OWNER mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;
