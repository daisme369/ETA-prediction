from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = (ROOT_DIR / "mlruns").as_uri()
DEFAULT_EXPERIMENT_NAME = "eta_xgboost_baseline"
DEFAULT_MODEL_NAME = "xgb_eta_residual_pipeline"
DEFAULT_JOBLIB_NAME = "xgb_eta_residual.joblib"
DEFAULT_JOBLIB_ARTIFACT_PATH = f"model/{DEFAULT_JOBLIB_NAME}"
DEFAULT_LOCAL_JOBLIB_PATH = ROOT_DIR / "model" / "artifacts" / DEFAULT_JOBLIB_NAME

REQUIRED_FEATURE_COLUMNS = [
    "origin_h3",
    "destination_h3",
    "origin_lng",
    "origin_lat",
    "destination_lng",
    "destination_lat",
    "hour_of_day",
    "is_rush_hour",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "haversine_distance_meters",
    "baseline_distance_meters",
    "traffic_level",
    "is_raining",
    "rain_level",
    "weather_condition",
    "baseline_eta_secs",
]

DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

MODEL = None
MODEL_INFO: dict[str, Any] = {}

app = FastAPI(title="ETA Prediction API", version="0.1.0")

cors_origins = os.getenv("CORS_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS))
allowed_origins = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def find_latest_joblib(root_dir: Path) -> Path | None:
    if not root_dir.exists():
        return None

    candidates: list[Path] = []
    for path in root_dir.rglob(DEFAULT_JOBLIB_NAME):
        if ".trash" in path.parts:
            continue
        if "artifacts" not in path.parts:
            continue
        candidates.append(path)

    if not candidates:
        return None

    return max(candidates, key=lambda item: item.stat().st_mtime)


def ensure_pickle_compatibility() -> None:
    # Older joblib artifacts were saved when model_baseline.py ran as __main__.
    try:
        from model import model_baseline
    except Exception:
        return

    main_module = sys.modules.get("__main__")
    if main_module is None:
        return

    for name in ("ETAFeatureEngineer", "QuantileClipper"):
        if not hasattr(main_module, name) and hasattr(model_baseline, name):
            setattr(main_module, name, getattr(model_baseline, name))


def load_joblib_from_uri(artifact_uri: str) -> tuple[Any, Path]:
    ensure_pickle_compatibility()
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
    model_path = Path(local_path)
    return joblib.load(model_path), model_path


def load_model() -> tuple[Any, dict[str, Any]]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    model_uri = os.getenv("MLFLOW_MODEL_URI", "").strip()
    joblib_uri = os.getenv("MLFLOW_JOBLIB_URI", "").strip()
    joblib_artifact_path = (
        os.getenv("MLFLOW_JOBLIB_ARTIFACT_PATH", DEFAULT_JOBLIB_ARTIFACT_PATH).strip()
        or DEFAULT_JOBLIB_ARTIFACT_PATH
    )
    model_name = os.getenv("MLFLOW_MODEL_NAME", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
    run_id = os.getenv("MLFLOW_RUN_ID", "").strip()

    mlflow.set_tracking_uri(tracking_uri)

    if model_uri:
        model = mlflow.sklearn.load_model(model_uri)
        return model, {
            "tracking_uri": tracking_uri,
            "model_uri": model_uri,
            "model_name": model_name,
            "model_source": "mlflow_model",
        }

    if joblib_uri:
        ensure_pickle_compatibility()
        model, model_path = load_joblib_from_uri(joblib_uri)
        return model, {
            "tracking_uri": tracking_uri,
            "model_uri": joblib_uri,
            "model_name": model_name,
            "model_source": "mlflow_joblib_uri",
            "model_path": str(model_path),
        }

    if run_id:
        joblib_uri = f"runs:/{run_id}/{joblib_artifact_path}"
        try:
            ensure_pickle_compatibility()
            model, model_path = load_joblib_from_uri(joblib_uri)
            return model, {
                "tracking_uri": tracking_uri,
                "model_uri": joblib_uri,
                "model_name": model_name,
                "model_source": "mlflow_joblib_run",
                "model_path": str(model_path),
            }
        except Exception:
            pass

    latest_joblib = find_latest_joblib(ROOT_DIR / "mlruns")
    if latest_joblib:
        ensure_pickle_compatibility()
        return joblib.load(latest_joblib), {
            "tracking_uri": tracking_uri,
            "model_uri": str(latest_joblib),
            "model_name": model_name,
            "model_source": "mlruns_joblib",
        }

    if DEFAULT_LOCAL_JOBLIB_PATH.exists():
        ensure_pickle_compatibility()
        return joblib.load(DEFAULT_LOCAL_JOBLIB_PATH), {
            "tracking_uri": tracking_uri,
            "model_uri": str(DEFAULT_LOCAL_JOBLIB_PATH),
            "model_name": model_name,
            "model_source": "local_joblib",
        }

    raise RuntimeError(
        "Could not locate a usable model. Set MLFLOW_MODEL_URI, MLFLOW_JOBLIB_URI, "
        "or MLFLOW_RUN_ID (with MLFLOW_JOBLIB_ARTIFACT_PATH)."
    )


def normalize_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "records" in payload:
        records = payload["records"]
        if not isinstance(records, list):
            raise HTTPException(status_code=422, detail="records must be a list.")
        return records
    return [payload]


def validate_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cleaned: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append({"index": index, "error": "record must be an object"})
            continue
        missing = [
            column
            for column in REQUIRED_FEATURE_COLUMNS
            if column not in record or record[column] is None
        ]
        if missing:
            errors.append({"index": index, "missing": missing})
            continue
        cleaned.append({column: record[column] for column in REQUIRED_FEATURE_COLUMNS})
    return cleaned, errors


@app.on_event("startup")
def startup() -> None:
    global MODEL, MODEL_INFO
    MODEL, MODEL_INFO = load_model()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_uri": MODEL_INFO.get("model_uri"),
        "model_name": MODEL_INFO.get("model_name"),
        "model_source": MODEL_INFO.get("model_source"),
        "required_features": REQUIRED_FEATURE_COLUMNS,
    }


@app.get("/ready")
def ready() -> dict[str, Any]:
    return {
        "status": "ready" if MODEL is not None else "loading",
        "model_loaded": MODEL is not None,
    }


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    records = normalize_records(payload)
    cleaned, errors = validate_records(records)
    if errors:
        raise HTTPException(status_code=422, detail={"message": "Invalid input", "errors": errors})

    frame = pd.DataFrame(cleaned)
    raw_predictions = MODEL.predict(frame)
    baseline_eta = frame["baseline_eta_secs"].to_numpy(dtype=float)
    eta_predictions = np.maximum(raw_predictions + baseline_eta, 1.0)

    response = {
        "predictions": [
            {
                "eta_seconds": float(value),
                "eta_minutes": float(value) / 60.0,
            }
            for value in eta_predictions
        ],
        "model_uri": MODEL_INFO.get("model_uri"),
        "model_name": MODEL_INFO.get("model_name"),
    }

    if payload.get("return_features"):
        response["used_features"] = REQUIRED_FEATURE_COLUMNS

    return response
