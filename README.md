# Artificial Intelligence Driven Customer Retention System

> End-to-end telecom customer-churn prediction platform combining behavioral feature engineering, recall-oriented machine learning, experiment tracking, API serving, caching, persistence, observability, and a Streamlit experiment workspace.

## Overview

Customer churn is a business problem as much as a classification problem. A production retention system needs to answer three questions:

1. Which customers are at risk of churn?
2. Which behavioral signals are associated with that risk?
3. How can the prediction be operationalized for retention workflows?

This repository contains both the original analytical notebook and a production-oriented ML/MLOps layer.

## Complete Architecture
<img width="1215" height="1295" alt="ChatGPT Image Sep 19, 2026, 09_18_41 PM" src="https://github.com/user-attachments/assets/e9eb5a39-63e2-4f24-8b4a-46c363b79fd9" />



## Repository Structure

```text
.
├── Artificial_intelligence_Driven_Customer_Retention_System.ipynb
├── app.py
├── api/
│   └── main.py
├── src/
│   ├── data.py
│   ├── evaluation.py
│   └── model.py
├── training/
│   └── train.py
├── mlflow/
│   └── Dockerfile
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       ├── dashboards/
│       └── provisioning/
├── tests/
├── Dockerfile
├── Dockerfile.web
├── docker-compose.yml
├── requirements.txt
├── requirements-api.txt
├── requirements-train.txt
├── requirements-web.txt
├── train data.zip
├── test.csv
├── sample.csv
├── data_dictionary.csv
└── README.md
```

## Machine Learning Pipeline

The production training path is deliberately separated from the exploratory notebook.

```text
Raw Dataset
    │
    ▼
Dataset Loader
    │
    ▼
Column Normalization
    │
    ▼
Target Detection / Encoding
    │
    ▼
Train / Test Split
    │
    ▼
Preprocessing Pipeline
    ├── numeric imputation
    ├── standard scaling
    └── categorical one-hot encoding
    │
    ▼
Balanced Logistic Regression
    │
    ▼
Churn Probabilities
    │
    ▼
Threshold Search
    │
    ├── minimum recall constraint
    ├── precision
    ├── recall
    ├── F1
    ├── ROC-AUC
    └── PR-AUC
    │
    ▼
MLflow + Local Artifacts
```

The saved model is a single sklearn pipeline so preprocessing used during training remains attached to the model at inference time.

## Business-Oriented Evaluation

Because the system is intended for customer-retention prioritization, accuracy alone is insufficient.

The production evaluator records:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Fraction of predicted churners that are actual churners |
| Recall | Fraction of actual churners detected |
| F1 | Precision/recall trade-off |
| ROC-AUC | Ranking quality across thresholds |
| PR-AUC | Performance under class imbalance |

The training command can enforce a minimum churn recall:

```bash
python -m training.train \
  --data "train data.zip" \
  --out artifacts \
  --min-recall 0.80
```

The decision threshold is therefore treated as a business configuration rather than assuming 0.50 is always appropriate.

## Feature Engineering

The original project focuses on telecom behavioral signals such as:

- ARPU and revenue changes
- call and usage behavior
- recharge behavior and gaps
- recent-vs-historical behavioral changes
- network tenure / age on network
- service and usage indicators

The repository also contains `data_dictionary.csv` describing the telecom feature abbreviations.

## Model Tracking with MLflow

Training logs the experiment into MLflow, including:

- model type
- class weighting
- target column
- training/test row counts
- minimum-recall target
- default-threshold metrics
- optimized-threshold metrics
- optimized decision threshold
- serialized model

Start MLflow with the Compose stack and inspect runs at:

```text
http://localhost:5000
```

An optional registered-model name can be supplied with:

```text
MLFLOW_REGISTERED_MODEL_NAME
```

## Streamlit Experiment Workspace

The Streamlit application provides a simple experiment environment:

```text
┌──────────────────────────────────────────────────┐
│          CUSTOMER RETENTION ML LAB               │
├──────────────────────────────────────────────────┤
│ Dataset: train data.zip                          │
│ Minimum Recall: 80%                              │
│                                                  │
│              [ RUN FULL EXPERIMENT ]              │
├──────────────────────────────────────────────────┤
│ Overview | Data | Output | Artifacts | API       │
├──────────────────────────────────────────────────┤
│ Recall │ Precision │ F1 │ PR-AUC                 │
└──────────────────────────────────────────────────┘
```

The UI calls the same `training.train` module used by the production pipeline instead of maintaining a second implementation of the model.

## Measured Experiment Artifacts

After a successful experiment, the shared `artifacts/` directory contains:

```text
artifacts/
├── churn_model.joblib
├── threshold.json
├── metrics.json
├── metadata.json
└── experiment_history.jsonl
```

These files are generated from the actual experiment run. The dashboard should display measured values from these artifacts rather than hard-coded demonstration metrics.

## FastAPI Inference Service

The production API exposes:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness |
| `GET /ready` | Readiness/model availability |
| `GET /model` | Model metadata |
| `POST /predict` | Churn prediction |
| `POST /reload` | Reload updated model artifact |
| `GET /metrics` | Prometheus metrics |

Swagger/OpenAPI is available at:

```text
http://localhost:8000/docs
```

Example request:

```json
{
  "customer_id": "customer-001",
  "features": {
    "arpu_8": 120.0,
    "arpu_7": 145.0,
    "mou_8": 310.0
  }
}
```

## Redis

Redis is used as a low-latency prediction cache.

```text
Prediction Request
       │
       ▼
   Hash Payload
       │
       ▼
    Redis GET
     /     \
   hit     miss
    │        │
    │        ▼
    │      Model
    │        │
    │        ▼
    │      Result
    │        │
    └────────┴──► Redis SETEX
```

## PostgreSQL

Prediction events can be persisted for operational analysis:

```text
prediction_events
├── id
├── customer_id
├── probability
├── prediction
├── model_version
└── created_at
```

The prediction path is designed so temporary analytics-storage problems do not turn into model-serving outages.

## Prometheus and Grafana

The API exports operational metrics such as:

- prediction request count
- prediction count by class
- cache hits
- prediction latency
- model-loaded status
- Redis status
- PostgreSQL status

Grafana is provisioned against Prometheus with a starter dashboard containing request volume, model status, request rate, and p95 latency.

## Docker Compose

The full local platform can be started with one command:

```bash
docker compose up --build
```

Services:

| Service | Port | Purpose |
|---|---:|---|
| Streamlit | 8501 | Experiment interface |
| FastAPI | 8000 | Model serving |
| MLflow | 5000 | Experiment tracking |
| PostgreSQL | 5432 | Persistent metadata/events |
| Redis | 6379 | Prediction cache |
| Prometheus | 9090 | Metrics |
| Grafana | 3000 | Monitoring UI |

## Local Workflow

### 1. Start platform

```bash
docker compose up --build
```

### 2. Open Streamlit

```text
http://localhost:8501
```

### 3. Run experiment

Use the **Run full experiment** button. The application uses:

```text
train data.zip
        ↓
training.train
        ↓
model + threshold + metrics
```

### 4. Inspect MLflow

```text
http://localhost:5000
```

### 5. Test API

```text
http://localhost:8000/docs
```

### 6. Monitor

```text
Prometheus → http://localhost:9090
Grafana    → http://localhost:3000
```

## Environment Variables

Useful runtime configuration includes:

```text
MODEL_VERSION
MODEL_RELOAD_TOKEN
ARTIFACT_DIR
DATABASE_URL
REDIS_URL
CACHE_TTL_SECONDS
CHURN_THRESHOLD
MLFLOW_TRACKING_URI
MLFLOW_EXPERIMENT
MLFLOW_REGISTERED_MODEL_NAME
MIN_RECALL_TARGET
```

For real deployments, secrets should be injected by the runtime/orchestrator rather than committed to Git.

## Testing and CI

GitHub Actions validates the production source tree with:

```text
Checkout
   ↓
Python 3.12
   ↓
Dependency installation
   ↓
Ruff
   ↓
Python compile check
   ↓
Pytest
```

The intention is to catch lint, import, syntax, and unit-test failures before deployment.

## Production Considerations

The included Compose stack is a strong local integration environment, but a real production deployment should additionally use:

- managed PostgreSQL
- durable object storage for model artifacts
- managed MLflow or equivalent model registry
- authentication and authorization
- TLS
- secret management
- network isolation
- encrypted storage
- PII minimization
- data validation
- model/data drift alerting
- scheduled retraining
- approval gates before promotion
- canary/blue-green rollout
- automated rollback
- backup and disaster recovery

## Important Data Note

`sample.csv` is a prediction-output sample, not the primary training dataset. The production experiment UI therefore defaults to:

```text
train data.zip
```

The repository's `data_dictionary.csv` documents the telecom feature abbreviations used by the source data.

## Original Notebook

The original exploratory work remains available in:

```text
Artificial_intelligence_Driven_Customer_Retention_System.ipynb
```

Use the notebook for EDA, experimentation, feature analysis, and model comparison. Use the `src/`, `training/`, `api/`, and Docker/MLOps components for the reproducible serving path.

## License

See [LICENSE](LICENSE).
