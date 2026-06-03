from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT_DIR / "model" / "artifacts" / "fixed_route_eta_model.joblib"
QUANTILE_LABELS = ("p50", "p85", "p90")


class EtaPredictRequest(BaseModel):
    departure_time: datetime | None = Field(
        default=None,
        description="Local departure datetime. Only the hour is used by the current fixed-route model.",
    )
    hour: int | None = Field(
        default=None,
        ge=0,
        le=23,
        description="Optional direct hour override for quick tests.",
    )

    @model_validator(mode="after")
    def require_time_or_hour(self) -> "EtaPredictRequest":
        if self.departure_time is None and self.hour is None:
            raise ValueError("Provide departure_time or hour.")
        return self


class EtaPrediction(BaseModel):
    seconds: float
    minutes: float


app = FastAPI(
    title="Fixed Route ETA API",
    version="1.0.0",
    description="Serves the packed ETA model for one fixed bus route.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def load_model_package() -> dict[str, Any]:
    if not ARTIFACT_PATH.exists():
        raise RuntimeError(f"Model artifact not found: {ARTIFACT_PATH}")
    package = joblib.load(ARTIFACT_PATH)
    if not isinstance(package, dict) or "model" not in package:
        raise RuntimeError("Model artifact has an unexpected format.")
    return package


MODEL_PACKAGE = load_model_package()


def make_features(hour: int) -> pd.DataFrame:
    feature_row = {
        "hour": hour,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "is_morning_peak": int(6 <= hour <= 8),
        "is_evening_peak": int(16 <= hour <= 18),
        "is_late": int(19 <= hour <= 23),
    }
    feature_columns = MODEL_PACKAGE.get("feature_columns") or list(feature_row)
    return pd.DataFrame([feature_row], columns=feature_columns)


def format_prediction(seconds: float) -> EtaPrediction:
    clipped_seconds = max(float(seconds), 0.0)
    return EtaPrediction(seconds=clipped_seconds, minutes=clipped_seconds / 60.0)


def predict_for_hour(hour: int) -> dict[str, Any]:
    features = make_features(hour)
    model = MODEL_PACKAGE["model"]
    point_seconds = float(np.asarray(model.predict(features)).reshape(-1)[0])

    quantile_models = MODEL_PACKAGE.get("quantile_models") or {}
    quantile_predictions: list[tuple[str, float]] = []
    quantiles: dict[str, EtaPrediction] = {}
    for label in QUANTILE_LABELS:
        fitted = quantile_models.get(label)
        if fitted is None:
            continue
        raw_seconds = float(np.asarray(fitted.predict(features)).reshape(-1)[0])
        quantile_predictions.append((label, max(raw_seconds, 0.0)))

    if quantile_predictions:
        labels = [label for label, _ in quantile_predictions]
        raw_seconds = [seconds for _, seconds in quantile_predictions]
        monotonic_seconds = np.maximum.accumulate(np.asarray(raw_seconds, dtype=float))
        for label, seconds in zip(labels, monotonic_seconds):
            quantiles[label] = format_prediction(float(seconds))

    return {
        "hour": hour,
        "point": format_prediction(point_seconds),
        "quantiles": quantiles,
    }


def route_payload() -> dict[str, Any]:
    route = MODEL_PACKAGE.get("route") or {}
    return {
        "stationId": route.get("stationId"),
        "destination_stationId": route.get("destination_stationId"),
        "origin": {
            "lat": route.get("lat"),
            "lng": route.get("lng"),
            "label": f"Station {route.get('stationId', '-')}",
        },
        "destination": {
            "lat": route.get("destination_lat"),
            "lng": route.get("destination_lng"),
            "label": f"Station {route.get('destination_stationId', '-')}",
        },
        "distance_meters": route.get("haversine_distance_meters"),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/eta/model-info")
def model_info() -> dict[str, Any]:
    return {
        "artifact": str(ARTIFACT_PATH.relative_to(ROOT_DIR)),
        "model_name": MODEL_PACKAGE.get("model_name"),
        "selection_policy": MODEL_PACKAGE.get("selection_policy"),
        "cv_best_model_name": MODEL_PACKAGE.get("cv_best_model_name"),
        "selected_quantile_model_names": MODEL_PACKAGE.get("selected_quantile_model_names", {}),
        "feature_columns": MODEL_PACKAGE.get("feature_columns", []),
        "target_column": MODEL_PACKAGE.get("target_column"),
        "route": route_payload(),
        "holdout_metrics": MODEL_PACKAGE.get("holdout_metrics", {}),
        "quantile_holdout_metrics": MODEL_PACKAGE.get("quantile_holdout_metrics", []),
    }


@app.post("/api/eta/predict")
def predict_eta(payload: EtaPredictRequest) -> dict[str, Any]:
    hour = payload.hour if payload.hour is not None else payload.departure_time.hour
    if hour is None or hour < 0 or hour > 23:
        raise HTTPException(status_code=422, detail="Hour must be between 0 and 23.")

    prediction = predict_for_hour(hour)
    return {
        "route": route_payload(),
        "model_name": MODEL_PACKAGE.get("model_name"),
        "selection_policy": MODEL_PACKAGE.get("selection_policy"),
        "cv_best_model_name": MODEL_PACKAGE.get("cv_best_model_name"),
        "selected_quantile_model_names": MODEL_PACKAGE.get("selected_quantile_model_names", {}),
        "input": {
            "departure_time": payload.departure_time.isoformat() if payload.departure_time else None,
            "hour": hour,
        },
        "prediction": prediction,
    }
