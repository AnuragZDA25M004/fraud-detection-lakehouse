#!/usr/bin/env python3
"""
promote_best_model.py — Promote Best MLflow Run to Production
Z5008 Big Data Lab · IIT Madras Zanzibar 2026

Connects to the MLflow tracking server, finds the experiment run with the
highest AUC-PR, registers the model (if not already registered), and
transitions it to the "Production" stage — demonstrating MLOps maturity.

Usage:
    pip install mlflow
    python ml/promote_best_model.py

    # Or point to a remote MLflow server:
    MLFLOW_TRACKING_URI=http://localhost:5000 python ml/promote_best_model.py
"""

import os
import sys

import mlflow
from mlflow.tracking import MlflowClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI  = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME      = "fraud_detection_gbt"
REGISTERED_MODEL_NAME = "FraudDetectionGBT"
METRIC_TO_OPTIMISE   = "auc_pr"          # higher is better
STAGE_TARGET         = "Production"
STAGE_ARCHIVE        = "Archived"        # demote previous Production runs here


def promote_best_model() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print("=" * 60)
    print("  MLflow Model Promotion — Fraud Detection GBT")
    print(f"  Tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"  Experiment   : {EXPERIMENT_NAME}")
    print(f"  Optimising   : {METRIC_TO_OPTIMISE}  (higher = better)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Locate the experiment
    # ------------------------------------------------------------------
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"\n[ERROR] Experiment '{EXPERIMENT_NAME}' not found.")
        print("  → Run ml/spark_ml_training.py first to generate MLflow runs.")
        sys.exit(1)

    experiment_id = experiment.experiment_id
    print(f"\n[✓] Experiment found (id={experiment_id})")

    # ------------------------------------------------------------------
    # 2. Fetch all completed runs and find the best one
    # ------------------------------------------------------------------
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=[f"metrics.{METRIC_TO_OPTIMISE} DESC"],
        max_results=10,
    )

    if not runs:
        print("\n[ERROR] No completed runs found in this experiment.")
        sys.exit(1)

    print(f"\n[*] Completed runs found: {len(runs)}")
    print(f"\n{'Run ID':>36}  {'AUC-PR':>8}  {'F1':>8}  {'AUC-ROC':>8}")
    print("-" * 65)
    for run in runs:
        m = run.data.metrics
        print(
            f"  {run.info.run_id}  "
            f"{m.get(METRIC_TO_OPTIMISE, 0.0):>8.4f}  "
            f"{m.get('f1', 0.0):>8.4f}  "
            f"{m.get('auc_roc', 0.0):>8.4f}"
        )

    best_run = runs[0]
    best_run_id  = best_run.info.run_id
    best_auc_pr  = best_run.data.metrics.get(METRIC_TO_OPTIMISE, 0.0)
    best_f1      = best_run.data.metrics.get("f1", 0.0)
    best_auc_roc = best_run.data.metrics.get("auc_roc", 0.0)

    print(f"\n[✓] Best run selected: {best_run_id}")
    print(f"      AUC-PR  = {best_auc_pr:.4f}")
    print(f"      F1      = {best_f1:.4f}")
    print(f"      AUC-ROC = {best_auc_roc:.4f}")

    # ------------------------------------------------------------------
    # 3. Register the model (creates a new version if already registered)
    # ------------------------------------------------------------------
    model_uri = f"runs:/{best_run_id}/model"
    print(f"\n[*] Registering model from: {model_uri}")

    try:
        model_version_info = mlflow.register_model(
            model_uri=model_uri,
            name=REGISTERED_MODEL_NAME,
        )
        version = model_version_info.version
        print(f"[✓] Registered as '{REGISTERED_MODEL_NAME}' version {version}")
    except Exception as exc:
        print(f"[ERROR] Registration failed: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Archive any existing Production versions
    # ------------------------------------------------------------------
    try:
        existing_prod = client.get_latest_versions(
            REGISTERED_MODEL_NAME, stages=[STAGE_TARGET]
        )
        for prod_version in existing_prod:
            if prod_version.version != str(version):
                client.transition_model_version_stage(
                    name=REGISTERED_MODEL_NAME,
                    version=prod_version.version,
                    stage=STAGE_ARCHIVE,
                    archive_existing_versions=False,
                )
                print(
                    f"[~] Archived previous Production version {prod_version.version}"
                )
    except Exception as exc:
        print(f"[WARN] Could not archive old Production versions: {exc}")

    # ------------------------------------------------------------------
    # 5. Promote new version to Production
    # ------------------------------------------------------------------
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage=STAGE_TARGET,
        archive_existing_versions=True,
    )

    print(f"\n[✓] Model version {version} promoted to '{STAGE_TARGET}'")

    # ------------------------------------------------------------------
    # 6. Tag the version with key metrics for traceability
    # ------------------------------------------------------------------
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="auc_pr",
        value=str(round(best_auc_pr, 4)),
    )
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="f1",
        value=str(round(best_f1, 4)),
    )
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="promoted_by",
        value="promote_best_model.py",
    )

    print(f"[✓] Tags written: auc_pr={best_auc_pr:.4f}, f1={best_f1:.4f}")

    # ------------------------------------------------------------------
    # 7. Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PROMOTION COMPLETE")
    print("=" * 60)
    print(f"  Model    : {REGISTERED_MODEL_NAME}")
    print(f"  Version  : {version}")
    print(f"  Stage    : {STAGE_TARGET}")
    print(f"  Run ID   : {best_run_id}")
    print(f"  AUC-PR   : {best_auc_pr:.4f}")
    print()
    print("  Load in BentoML / Spark with:")
    print(f"    mlflow.spark.load_model('models:/{REGISTERED_MODEL_NAME}/Production')")
    print()
    print(f"  View in MLflow UI → {MLFLOW_TRACKING_URI}/#/models/{REGISTERED_MODEL_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    promote_best_model()
