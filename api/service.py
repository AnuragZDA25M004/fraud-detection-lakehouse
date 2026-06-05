"""
api/service.py
--------------
BentoML REST API — loads the best registered MLflow model
and serves fraud predictions.

Endpoint: POST /predict
Input:    JSON transaction features
Output:   {"label": "FRAUD" | "LEGIT", "confidence": 0.94,
           "transaction_amt": 1500.0}

Build and run:
    docker build -t fraud-api .
    docker run -p 3001:3000 --network bigdata-net fraud-api
"""

import bentoml
import mlflow
import mlflow.spark
import numpy as np
import pandas as pd
from bentoml.io import JSON
from pydantic import BaseModel
from typing import Optional
import os

# ── Input schema ──────────────────────────────────────────────────────────────
class Transaction(BaseModel):
    TransactionAmt:     float
    log_TransactionAmt: Optional[float] = None   # auto-computed if not given
    tx_hour:            Optional[int]   = 12
    is_high_value:      Optional[int]   = None   # auto-computed if not given
    email_match:        Optional[int]   = 0
    card1:              Optional[float] = -999
    card2:              Optional[float] = -999
    card3:              Optional[float] = -999
    card4:              Optional[int]   = -1
    card5:              Optional[float] = -999
    card6:              Optional[int]   = -1
    addr1:              Optional[float] = -999
    addr2:              Optional[float] = -999
    dist1:              Optional[float] = -999
    ProductCD:          Optional[int]   = -1

# ── Output schema ─────────────────────────────────────────────────────────────
class Prediction(BaseModel):
    label:           str
    confidence:      float
    transaction_amt: float

# ── Load model from MLflow registry ──────────────────────────────────────────
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME  = "FraudDetection-GBT"
MODEL_STAGE = "Production"

mlflow.set_tracking_uri(MLFLOW_URI)

# Load the latest production model
model_uri = f"models:/{MODEL_NAME}/latest"
print(f"[INFO] Loading model from {model_uri} ...")
loaded_model = mlflow.pyfunc.load_model(model_uri)
print("[INFO] Model loaded ✅")

# ── BentoML runner ────────────────────────────────────────────────────────────
fraud_runner = bentoml.picklable_model.get("fraud_gbt:latest").to_runner()

svc = bentoml.Service("fraud_detection", runners=[fraud_runner])

# ── Prediction endpoint ───────────────────────────────────────────────────────
@svc.api(input=JSON(pydantic_model=Transaction), output=JSON(pydantic_model=Prediction))
def predict(transaction: Transaction) -> Prediction:
    import math

    # Auto-compute derived features
    log_amt    = math.log1p(transaction.TransactionAmt) \
                 if transaction.log_TransactionAmt is None \
                 else transaction.log_TransactionAmt
    high_value = 1 if transaction.TransactionAmt > 500 else 0 \
                 if transaction.is_high_value is None \
                 else transaction.is_high_value

    features = pd.DataFrame([{
        "TransactionAmt":     transaction.TransactionAmt,
        "log_TransactionAmt": log_amt,
        "tx_hour":            transaction.tx_hour,
        "is_high_value":      high_value,
        "email_match":        transaction.email_match,
        "card1":              transaction.card1,
        "card2":              transaction.card2,
        "card3":              transaction.card3,
        "card4":              transaction.card4,
        "card5":              transaction.card5,
        "card6":              transaction.card6,
        "addr1":              transaction.addr1,
        "addr2":              transaction.addr2,
        "dist1":              transaction.dist1,
        "ProductCD":          transaction.ProductCD,
    }])

    result      = loaded_model.predict(features)
    probability = float(result[0]) if hasattr(result, '__iter__') else float(result)
    label       = "FRAUD" if probability >= 0.5 else "LEGIT"

    return Prediction(
        label=label,
        confidence=round(probability if label == "FRAUD" else 1 - probability, 4),
        transaction_amt=transaction.TransactionAmt,
    )
