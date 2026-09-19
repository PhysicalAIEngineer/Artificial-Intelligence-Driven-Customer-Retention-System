# Production ML / MLOps Layer

This branch adds a deployable ML system around the existing churn notebook.

## Architecture

Data → preprocessing/model pipeline → recall-aware threshold optimization → MLflow → versioned artifact → FastAPI → Prometheus → Docker/CI.

## Train

Start MLflow:

\`docker compose up -d mlflow\`

Then:

\`python -m training.train --data "train data.zip" --out artifacts --min-recall 0.80\`

The training job logs parameters and default/optimized metrics to MLflow and saves:
- \`artifacts/churn_model.joblib\`
- \`artifacts/threshold.json\`
- \`artifacts/metadata.json\`

The model artifact is a complete sklearn Pipeline, keeping preprocessing and inference coupled.

## Serve

\`uvicorn api.main:app --host 0.0.0.0 --port 8000\`

Endpoints:
- GET \`/health\`
- GET \`/ready\`
- GET \`/model\`
- POST \`/predict\`
- GET \`/metrics\`

Example request:

\`\`\`json
{
  "customer_id": "customer-001",
  "features": {
    "arpu_8": 120.0,
    "arpu_7": 145.0,
    "mou_8": 310.0
  }
}
\`\`\`

## MLOps

- MLflow experiment tracking.
- Optional MLflow registered model via \`MLFLOW_REGISTERED_MODEL_NAME\`.
- Recall-constrained threshold selection.
- FastAPI health/readiness.
- Prometheus latency/request/model metrics.
- Dockerized inference.
- GitHub Actions lint/test/compile/image-build gates.
- Unit tests for evaluation and API health.

## Production hardening

Before real customer deployment, add managed MLflow artifact storage/model registry, authentication/TLS, secret management, encrypted PII storage, PostgreSQL prediction/event logging, Redis caching, feature/data validation, drift alerts, scheduled retraining, model approval gates, canary rollout, and rollback automation.
