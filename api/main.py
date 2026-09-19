from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "artifacts" / "churn_model.joblib"))
THRESHOLD_PATH = Path(os.getenv("THRESHOLD_PATH", ROOT / "artifacts" / "threshold.json"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "churn-v1")

REQUESTS = Counter("churn_prediction_requests_total", "Prediction requests")
LATENCY = Histogram("churn_prediction_latency_seconds", "Prediction latency")
MODEL_LOADED = Gauge("churn_model_loaded", "Whether model is loaded")

app = FastAPI(title="Customer Retention ML API", version="1.0.0")
model = None
threshold = float(os.getenv("CHURN_THRESHOLD", "0.50"))


class PredictionRequest(BaseModel):
    customer_id: str | None = Field(default=None, max_length=128)
    features: dict[str, float | int | str | None]


@app.on_event("startup")
def startup():
    global model, threshold
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        MODEL_LOADED.set(1)
    if THRESHOLD_PATH.exists():
        threshold = float(
            json.loads(THRESHOLD_PATH.read_text()).get("threshold", threshold)
        )


@app.get("/health")
def health():
    return {"status": "ok", "service": "customer-retention-api"}


@app.get("/ready")
def ready():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact is not loaded",
        )
    return {
        "status": "ready",
        "model_version": MODEL_VERSION,
        "threshold": threshold,
    }


@app.get("/model")
def model_info():
    return {
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION,
        "threshold": threshold,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    REQUESTS.inc()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact is not loaded",
        )

    started = time.perf_counter()
    try:
        features = dict(request.features)
        frame = pd.DataFrame([features])
        probability = float(model.predict_proba(frame)[0, 1])
        prediction = int(probability >= threshold)
        return {
            "customer_id": request.customer_id,
            "churn_probability": probability,
            "churn_prediction": prediction,
            "at_risk": bool(prediction),
            "threshold": threshold,
            "model_version": MODEL_VERSION,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed: {exc}",
        ) from exc
    finally:
        LATENCY.observe(time.perf_counter() - started)


app.mount("/metrics", make_asgi_app())
