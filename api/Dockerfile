# api/Dockerfile
# Builds the BentoML fraud detection REST API
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    bentoml==1.2.2 \
    mlflow==2.11.0 \
    pandas==2.1.0 \
    numpy==1.26.0 \
    pydantic==2.6.0 \
    boto3==1.34.0 \
    pyspark==3.5.0

# Copy service definition
COPY service.py .

# Environment variables (override at runtime)
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000
ENV AWS_ACCESS_KEY_ID=admin
ENV AWS_SECRET_ACCESS_KEY=bigdata123

EXPOSE 3000

CMD ["bentoml", "serve", "service:svc", "--host", "0.0.0.0", "--port", "3000"]
