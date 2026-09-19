from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd
import redis
from fastapi import FastAPI, Header, HTTPException
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text\nfrom sqlalchemy.exc import SQLAlchemyError

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(
    os.getenv("ARTIFACT_DIR", ROOT / "artifacts")
)
MODEL_PATH = ARTIFACT_DIR / "churn_model.joblib"
THRESHOLD_PATH = ARTIFACT_DIR / "threshold.json"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

MODEL_VERSION = os.getenv("MODEL_VERSION", "churn-v1")
RELOAD_TOKEN = os.getenv("MODEL_RELOAD_TOKEN", "")

DATABASE_URL = os.getenv("DATABASE_URL", "")
REDIS_URL = os.getenv("REDIS_URL", "")

db_engine = (
    create_engine(DATABASE_URL, pool_pre_ping=True)
    if DATABASE_URL
    else None
)

redis_client = (
    redis.Redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=0.2,
        socket_timeout=0.2,
    )
    if REDIS_URL
    else None
)

REQUESTS = Counter(
    "churn_prediction_requests_total",
    "Prediction requests",
)
PREDICTIONS = Counter(
    "churn_predictions_total",
    "Predictions by class",
    ["prediction"],
)
CACHE_HITS = Counter(
    "churn_prediction_cache_hits_total",
    "Prediction cache hits",
)
LATENCY = Histogram(
    "churn_prediction_latency_seconds",
    "Prediction latency",
)
MODEL_LOADED = Gauge(
    "churn_model_loaded",
    "Whether model is loaded",
)
DB_READY = Gauge(
    "churn_database_ready",
    "Whether prediction database is ready",
)
REDIS_READY = Gauge(
    "churn_redis_ready",
    "Whether Redis is ready",
)

app = FastAPI(
    title="Customer Retention ML API",
    version="1.1.0",
)

model = None
threshold = float(os.getenv("CHURN_THRESHOLD", "0.50"))
model_mtime = None


class PredictionRequest(BaseModel):
    customer_id: str | None = Field(
        default=None,
        max_length=128,
    )
    features: dict[str, float | int | str | None]


def _cache_key(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode()
    return "churn:" + hashlib.sha256(encoded).hexdigest()


def _init_database():
    if not db_engine:
        DB_READY.set(1)
        return
    try:
        with db_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS prediction_events (
                        id BIGSERIAL PRIMARY KEY,
                        customer_id VARCHAR(128),
                        probability DOUBLE PRECISION NOT NULL,
                        prediction INTEGER NOT NULL,
                        model_version VARCHAR(128) NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
            )
        DB_READY.set(1)
    except (redis.RedisError, OSError):
        DB_READY.set(0)


def _check_redis():
    if not redis_client:
        REDIS_READY.set(1)
        return
    try:
        redis_client.ping()
        REDIS_READY.set(1)
    except (redis.RedisError, OSError):
        REDIS_READY.set(0)


def _load_model():
    global model, threshold, model_mtime

    if not MODEL_PATH.exists():
        model = None
        MODEL_LOADED.set(0)
        return False

    model = joblib.load(MODEL_PATH)

    if THRESHOLD_PATH.exists():
        threshold = float(
            json.loads(THRESHOLD_PATH.read_text()).get(
                "threshold",
                threshold,
            )
        )

    model_mtime = MODEL_PATH.stat().st_mtime_ns
    MODEL_LOADED.set(1)
    return True


def _ensure_latest_model():
    if not MODEL_PATH.exists():
        return
    current_mtime = MODEL_PATH.stat().st_mtime_ns
    if model is None or current_mtime != model_mtime:
        _load_model()


def _write_prediction_event(
    request: PredictionRequest,
    probability: float,
    prediction: int,
):
    if not db_engine:
        return
    try:
        with db_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO prediction_events
                    (customer_id, probability, prediction, model_version)
                    VALUES (:customer_id, :probability, :prediction, :model_version)
                    """
                ),
                {
                    "customer_id": request.customer_id,
                    "probability": probability,
                    "prediction": prediction,
                    "model_version": MODEL_VERSION,
                },
            )
    except (redis.RedisError, OSError):
        DB_READY.set(0)


@app.on_event("startup")
def startup():
    _init_database()
    _check_redis()
    _load_model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "customer-retention-api",
    }


@app.get("/ready")
def ready():
    _ensure_latest_model()
    _check_redis()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact is not loaded",
        )

    return {
        "status": "ready",
        "model_version": MODEL_VERSION,
        "threshold": threshold,
        "database_ready": bool(DB_READY._value.get()),
        "redis_ready": bool(REDIS_READY._value.get()),
    }


@app.get("/model")
def model_info():
    _ensure_latest_model()

    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(
            METADATA_PATH.read_text()
        )

    return {
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION,
        "threshold": threshold,
        "metadata": metadata,
    }


@app.post("/reload")
def reload_model(
    x_model_reload_token: str | None = Header(default=None),
):
    if RELOAD_TOKEN and x_model_reload_token != RELOAD_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="Invalid model reload token",
        )

    if not _load_model():
        raise HTTPException(
            status_code=503,
            detail="Model artifact does not exist",
        )

    return {
        "status": "reloaded",
        "model_version": MODEL_VERSION,
        "threshold": threshold,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    REQUESTS.inc()
    _ensure_latest_model()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact is not loaded. Run an experiment first.",
        )

    payload = request.model_dump()
    key = _cache_key(payload)

    if redis_client:
        try:
            cached = redis_client.get(key)
            if cached:
                CACHE_HITS.inc()
                result = json.loads(cached)
                result["cached"] = True
                return result
        except (redis.RedisError, OSError):
            REDIS_READY.set(0)

    started = time.perf_counter()

    try:
        frame = pd.DataFrame([request.features])
        probability = float(
            model.predict_proba(frame)[0, 1]
        )
        prediction = int(probability >= threshold)

        result = {
            "customer_id": request.customer_id,
            "churn_probability": probability,
            "churn_prediction": prediction,
            "at_risk": bool(prediction),
            "threshold": threshold,
            "model_version": MODEL_VERSION,
            "cached": False,
        }

        PREDICTIONS.labels(str(prediction)).inc()

        if redis_client:
            try:
                redis_client.setex(
                    key,
                    int(os.getenv("CACHE_TTL_SECONDS", "300")),
                    json.dumps(result),
                )
            except (redis.RedisError, OSError):
                REDIS_READY.set(0)

        _write_prediction_event(
            request,
            probability,
            prediction,
        )

        return result
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed: {exc}",
        ) from exc
    finally:
        LATENCY.observe(
            time.perf_counter() - started
        )


app.mount("/metrics", make_asgi_app())
