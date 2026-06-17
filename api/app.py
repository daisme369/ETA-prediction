from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator


ROOT_DIR = Path(__file__).resolve().parents[1]
ETA_MODELING_DIR = ROOT_DIR / "eta_modeling"
LEGACY_ARTIFACT_PATH = ROOT_DIR / "model" / "artifacts" / "fixed_route_eta_model.joblib"
EXPERIMENT_MODELS_DIR = ETA_MODELING_DIR / "artifacts" / "models"
EXPERIMENT_METRICS_DIR = ETA_MODELING_DIR / "artifacts" / "metrics"
RESIDUAL_MODELING_ARTIFACTS_DIR = ROOT_DIR / "residual_modeling" / "artifacts"
RAW_TIME_BIN_RESIDUALS_PATH = RESIDUAL_MODELING_ARTIFACTS_DIR / "enhanced_method_1_time_bin_residuals.csv"
RAW_TIME_BIN_MODEL_CARD_PATH = RESIDUAL_MODELING_ARTIFACTS_DIR / "enhanced_method_1_model_card.json"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed_data.csv"
QUANTILE_LABELS = ("p50", "p85", "p90")
DEFAULT_MODEL_ID = "mlp_residual_eta"
LEGACY_MODEL_ID = "legacy_fixed_route"

if ETA_MODELING_DIR.exists() and str(ETA_MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(ETA_MODELING_DIR))

try:
    import torch
    from src.features import add_engineered_features
    from src.models.deepr_eta_like import DeepRETALikeModel, predict_deepr_eta
    from src.models.mlp_residual import ResidualMLP, predict_residual
except Exception as import_error:  # pragma: no cover - surfaced in /api/eta/models
    torch = None
    add_engineered_features = None
    DeepRETALikeModel = None
    predict_deepr_eta = None
    ResidualMLP = None
    predict_residual = None
    ETA_MODELING_IMPORT_ERROR = import_error
else:
    ETA_MODELING_IMPORT_ERROR = None


class EtaPredictRequest(BaseModel):
    departure_time: datetime | None = Field(
        default=None,
        description="Local departure datetime. Experiment models use date and hour features.",
    )
    hour: int | None = Field(
        default=None,
        ge=0,
        le=23,
        description="Optional direct hour override for quick tests.",
    )
    model_id: str | None = Field(
        default=None,
        description="Prediction model id from /api/eta/models.",
    )
    baseline_eta_secs: float | None = Field(
        default=None,
        gt=0,
        description="Vietmap baseline ETA in seconds. Required by experiment models.",
    )
    rain: float | None = 0.0
    origin_rain: float | None = 0.0
    destination_rain: float | None = 0.0

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
    version="2.0.0",
    description="Serves the fixed-route ETA model plus experiment models from eta_modeling artifacts.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Thay thế bằng domain Vercel cụ thể nếu bạn muốn bảo mật hơn
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def load_legacy_model_package() -> dict[str, Any]:
    if not LEGACY_ARTIFACT_PATH.exists():
        raise RuntimeError(f"Model artifact not found: {LEGACY_ARTIFACT_PATH}")
    package = joblib.load(LEGACY_ARTIFACT_PATH)
    if not isinstance(package, dict) or "model" not in package:
        raise RuntimeError("Legacy model artifact has an unexpected format.")
    return package


def load_route_defaults() -> dict[str, Any]:
    if not PROCESSED_DATA_PATH.exists():
        return {}
    try:
        row = pd.read_csv(PROCESSED_DATA_PATH, nrows=1).iloc[0].to_dict()
    except Exception:
        return {}
    return row


LEGACY_MODEL_PACKAGE = load_legacy_model_package()
ROUTE_DEFAULTS = load_route_defaults()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def as_str(value: Any, default: str = "") -> str:
    return default if value is None else str(value)


def json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return value
    return value


def format_prediction(seconds: float) -> EtaPrediction:
    clipped_seconds = max(float(seconds), 0.0)
    return EtaPrediction(seconds=clipped_seconds, minutes=clipped_seconds / 60.0)


def route_payload() -> dict[str, Any]:
    route = LEGACY_MODEL_PACKAGE.get("route") or {}
    station_id = json_scalar(route.get("stationId", ROUTE_DEFAULTS.get("stationId")))
    destination_station_id = json_scalar(route.get("destination_stationId", ROUTE_DEFAULTS.get("destination_stationId")))
    lat = json_scalar(route.get("lat", ROUTE_DEFAULTS.get("lat")))
    lng = json_scalar(route.get("lng", ROUTE_DEFAULTS.get("lng")))
    destination_lat = json_scalar(route.get("destination_lat", ROUTE_DEFAULTS.get("destination_lat")))
    destination_lng = json_scalar(route.get("destination_lng", ROUTE_DEFAULTS.get("destination_lng")))
    return {
        "stationId": station_id,
        "destination_stationId": destination_station_id,
        "origin": {
            "lat": lat,
            "lng": lng,
            "label": f"Station {station_id or '-'}",
        },
        "destination": {
            "lat": destination_lat,
            "lng": destination_lng,
            "label": f"Station {destination_station_id or '-'}",
        },
        "distance_meters": json_scalar(
            route.get("haversine_distance_meters", ROUTE_DEFAULTS.get("haversine_distance_meters")),
        ),
    }


def make_legacy_features(hour: int) -> pd.DataFrame:
    feature_row = {
        "hour": hour,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "is_morning_peak": int(6 <= hour <= 8),
        "is_evening_peak": int(16 <= hour <= 18),
        "is_late": int(19 <= hour <= 23),
    }
    feature_columns = LEGACY_MODEL_PACKAGE.get("feature_columns") or list(feature_row)
    return pd.DataFrame([feature_row], columns=feature_columns)


def predict_legacy_for_hour(hour: int) -> dict[str, Any]:
    features = make_legacy_features(hour)
    model = LEGACY_MODEL_PACKAGE["model"]
    point_seconds = float(np.asarray(model.predict(features)).reshape(-1)[0])

    quantile_models = LEGACY_MODEL_PACKAGE.get("quantile_models") or {}
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


def read_metrics(model_name: str) -> dict[str, Any]:
    if model_name == "api_raw_time_bin_median_residual":
        return read_residual_modeling_metrics(
            RAW_TIME_BIN_MODEL_CARD_PATH,
            "api_plus_time_bin_median_residual",
        )

    metrics_path = EXPERIMENT_METRICS_DIR / f"{model_name}_metrics.json"
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    comparison_path = EXPERIMENT_METRICS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        try:
            comparison = pd.read_csv(comparison_path)
            match = comparison[comparison["model_name"] == model_name]
            if not match.empty:
                return match.iloc[0].dropna().to_dict()
        except Exception:
            return {}
    return {}


def read_residual_modeling_metrics(model_card_path: Path, method_name: str) -> dict[str, Any]:
    if not model_card_path.exists():
        return {}
    try:
        model_card = json.loads(model_card_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    metrics = model_card.get("metrics", [])
    if not isinstance(metrics, list):
        return {}
    for row in metrics:
        if isinstance(row, dict) and row.get("split") == "test" and row.get("method") == method_name:
            return row
    return {}


def read_deepr_eta_metadata(model_id: str = "deepr_eta_like") -> dict[str, Any]:
    """Read lightweight metadata from the latest DeeprETA-like artifacts."""
    if torch is None:
        return {}
    model_path = EXPERIMENT_MODELS_DIR / f"{model_id}.pt"
    bucketizer_path = EXPERIMENT_MODELS_DIR / f"{model_id}_bucketizers.joblib"
    if not model_path.exists():
        return {}
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
    except Exception:
        return {}
    metadata = {
        "feature_groups": checkpoint.get("feature_groups", {}),
        "cardinalities": checkpoint.get("cardinalities", {}),
        "bucketizer_artifact": (
            str(bucketizer_path.relative_to(ROOT_DIR))
            if bucketizer_path.exists()
            else None
        ),
    }
    feature_groups = metadata["feature_groups"]
    if isinstance(feature_groups, dict):
        metadata["categorical_embedding_features"] = feature_groups.get("categorical_embedding_features", [])
        metadata["continuous_bucket_embedding_features"] = feature_groups.get("continuous_bucket_embedding_features", [])
        metadata["time_features"] = feature_groups.get("time_features", [])
        metadata["weather_features"] = feature_groups.get("weather_features", [])
        metadata["bucket_features"] = feature_groups.get("bucket_features", {})
    return metadata


def model_spec(
    model_id: str,
    label: str,
    model_type: str,
    target_type: str,
    *,
    requires_baseline: bool,
    artifact_paths: list[Path] | None = None,
    description: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_paths = artifact_paths or []
    missing = [str(path.relative_to(ROOT_DIR)) for path in artifact_paths if not path.exists()]
    available = not missing
    error = None
    if ("mlp" in model_type or "deepr" in model_type) and ETA_MODELING_IMPORT_ERROR is not None:
        available = False
        error = f"Cannot import eta_modeling model code: {ETA_MODELING_IMPORT_ERROR}"
    return {
        "id": model_id,
        "label": label,
        "model_name": model_id,
        "model_type": model_type,
        "target_type": target_type,
        "requires_baseline": requires_baseline,
        "available": available,
        "missing_artifacts": missing,
        "error": error,
        "description": description,
        "metrics": read_metrics(model_id),
        "metadata": metadata or {},
    }


def available_model_specs() -> list[dict[str, Any]]:
    return [
        model_spec(
            LEGACY_MODEL_ID,
            "Legacy fixed-route model",
            "legacy_sklearn",
            "direct_eta_secs",
            requires_baseline=False,
            artifact_paths=[LEGACY_ARTIFACT_PATH],
            description="Original packed model with point ETA and quantile outputs.",
        ),
        model_spec(
            "vietmap_baseline",
            "Vietmap baseline",
            "vietmap_baseline",
            "baseline_eta_secs",
            requires_baseline=True,
            description="Returns Vietmap Route API duration as ETA.",
        ),
        model_spec(
            "xgb_direct_eta",
            "XGBoost direct ETA",
            "xgboost_direct",
            "actual_eta_secs",
            requires_baseline=True,
            artifact_paths=[EXPERIMENT_MODELS_DIR / "xgb_direct_eta.joblib"],
            description="XGBoost model trained to predict final ETA directly.",
        ),
        model_spec(
            "xgb_residual_eta",
            "XGBoost residual ETA",
            "xgboost_residual",
            "residual_secs",
            requires_baseline=True,
            artifact_paths=[EXPERIMENT_MODELS_DIR / "xgb_residual_eta.joblib"],
            description="XGBoost model predicts correction over Vietmap ETA.",
        ),
        model_spec(
            "mlp_residual_eta",
            "MLP residual ETA",
            "mlp_residual",
            "residual_secs",
            requires_baseline=True,
            artifact_paths=[
                EXPERIMENT_MODELS_DIR / "mlp_residual.pt",
                EXPERIMENT_MODELS_DIR / "mlp_residual_preprocessor.joblib",
            ],
            description="PyTorch MLP residual model; currently best MAE in the comparison table.",
        ),
        model_spec(
            "deepr_eta_like",
            "DeeprETA-like residual",
            "deepr_eta_like",
            "residual_secs_final_eta_loss",
            requires_baseline=True,
            artifact_paths=[
                EXPERIMENT_MODELS_DIR / "deepr_eta_like.pt",
                EXPERIMENT_MODELS_DIR / "deepr_eta_like_encoder.joblib",
                EXPERIMENT_MODELS_DIR / "deepr_eta_like_bucketizers.joblib",
            ],
            description="Embedding-based residual model inspired by DeeprETA.",
            metadata=read_deepr_eta_metadata(),
        ),
        model_spec(
            "hour_bin_median_eta",
            "Hour-bin median residual",
            "hour_bin_median",
            "median_residual_secs_by_hour_bin",
            requires_baseline=True,
            artifact_paths=[EXPERIMENT_MODELS_DIR / "hour_bin_median_eta.joblib"],
            description="Uses train-set median residual per service-period hour bin.",
            metadata={
                "hour_bin_feature": "hour_bin",
                "hour_bins": {
                    "early_morning": "5-6",
                    "morning_peak": "7-9",
                    "off_peak_midday": "10-14",
                    "afternoon_evening_peak": "15-18",
                    "late_evening_low_service": "19-21",
                    "other": "outside configured service hours",
                },
            },
        ),
        model_spec(
            "api_raw_time_bin_median_residual",
            "API + raw time-bin median residual",
            "api_raw_time_bin_median_residual",
            "raw_time_bin_median_residual_secs",
            requires_baseline=True,
            artifact_paths=[RAW_TIME_BIN_RESIDUALS_PATH],
            description="Adds the residual_modeling raw time-bin median correction to the routing API ETA.",
            metadata={
                "source_notebook": "residual_modeling/enhanced_method_1.ipynb",
                "residual_artifact": str(RAW_TIME_BIN_RESIDUALS_PATH.relative_to(ROOT_DIR)),
                "hour_bin_feature": "raw_time_bin",
                "fallback": "global median residual from the 'other' row, or artifact median when unavailable",
                "hour_bins": {
                    "early_morning": "4-6",
                    "morning_peak": "7-9",
                    "off_peak_day": "10-14",
                    "evening_peak": "15-18",
                    "late_evening": "19-21",
                    "other": "outside configured bins",
                },
            },
        ),
        model_spec(
            "hour_bin_xgb_residual_eta",
            "Hour-bin XGBoost residual",
            "hour_bin_xgboost_residual",
            "residual_secs",
            requires_baseline=True,
            artifact_paths=[EXPERIMENT_MODELS_DIR / "hour_bin_xgb_residual_eta.joblib"],
            description="XGBoost residual model using categorical hour_bin instead of sparse raw hour.",
        ),
        model_spec(
            "hour_bin_mlp_residual_eta",
            "Hour-bin MLP residual",
            "hour_bin_mlp_residual",
            "residual_secs_final_eta_loss",
            requires_baseline=True,
            artifact_paths=[
                EXPERIMENT_MODELS_DIR / "hour_bin_mlp_residual_eta.pt",
                EXPERIMENT_MODELS_DIR / "hour_bin_mlp_residual_eta_preprocessor.joblib",
            ],
            description="PyTorch MLP residual model using categorical hour_bin.",
        ),
        model_spec(
            "hour_bin_deepr_eta_like",
            "Hour-bin DeeprETA-like residual",
            "hour_bin_deepr_eta_like",
            "residual_secs_final_eta_loss",
            requires_baseline=True,
            artifact_paths=[
                EXPERIMENT_MODELS_DIR / "hour_bin_deepr_eta_like.pt",
                EXPERIMENT_MODELS_DIR / "hour_bin_deepr_eta_like_encoder.joblib",
                EXPERIMENT_MODELS_DIR / "hour_bin_deepr_eta_like_bucketizers.joblib",
            ],
            description="DeeprETA-like embedding residual model using hour_bin as the time token.",
            metadata=read_deepr_eta_metadata("hour_bin_deepr_eta_like"),
        ),
    ]


def get_model_spec(model_id: str) -> dict[str, Any]:
    specs = {spec["id"]: spec for spec in available_model_specs()}
    if model_id not in specs:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")
    return specs[model_id]


def resolve_model_id(model_id: str | None) -> str:
    if model_id:
        return model_id
    spec = get_model_spec(DEFAULT_MODEL_ID)
    return DEFAULT_MODEL_ID if spec["available"] else LEGACY_MODEL_ID


def request_hour(payload: EtaPredictRequest) -> int:
    hour = payload.hour if payload.hour is not None else payload.departure_time.hour
    if hour is None or hour < 0 or hour > 23:
        raise HTTPException(status_code=422, detail="Hour must be between 0 and 23.")
    return int(hour)


def experiment_feature_frame(payload: EtaPredictRequest, hour: int) -> pd.DataFrame:
    if add_engineered_features is None:
        raise HTTPException(status_code=503, detail=f"eta_modeling code is not importable: {ETA_MODELING_IMPORT_ERROR}")
    if payload.baseline_eta_secs is None or payload.baseline_eta_secs <= 0:
        raise HTTPException(status_code=422, detail="baseline_eta_secs from Vietmap is required for this model.")

    route = route_payload()
    origin = route["origin"]
    destination = route["destination"]
    departure = payload.departure_time or datetime.now()
    weekday = int(departure.weekday())
    row = {
        "stationId": as_str(route.get("stationId"), as_str(ROUTE_DEFAULTS.get("stationId"), "unknown")),
        "destination_stationId": as_str(
            route.get("destination_stationId"),
            as_str(ROUTE_DEFAULTS.get("destination_stationId"), "unknown"),
        ),
        "hour": hour,
        "lat": as_float(origin.get("lat"), as_float(ROUTE_DEFAULTS.get("lat"))),
        "lng": as_float(origin.get("lng"), as_float(ROUTE_DEFAULTS.get("lng"))),
        "destination_lat": as_float(destination.get("lat"), as_float(ROUTE_DEFAULTS.get("destination_lat"))),
        "destination_lng": as_float(destination.get("lng"), as_float(ROUTE_DEFAULTS.get("destination_lng"))),
        "timestamp": departure,
        "weekday": weekday,
        "month": int(departure.month),
        "is_weekend": int(weekday >= 5),
        "is_rush_hour": int(hour in {7, 8, 9, 17, 18, 19, 20}),
        "date": departure.date().isoformat(),
        "time": departure.time().isoformat(timespec="seconds"),
        "origin_rain": as_float(payload.origin_rain),
        "destination_rain": as_float(payload.destination_rain),
        "rain": as_float(payload.rain),
        "baseline_eta_secs": float(payload.baseline_eta_secs),
    }
    return add_engineered_features(pd.DataFrame([row]), use_cyclic=True, use_geohash=False)


@lru_cache(maxsize=None)
def load_joblib_experiment_model(filename: str) -> dict[str, Any]:
    path = EXPERIMENT_MODELS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Model artifact is missing: {path.relative_to(ROOT_DIR)}")
    package = joblib.load(path)
    if not isinstance(package, dict) or "model" not in package or "preprocessor" not in package:
        raise HTTPException(status_code=503, detail=f"Model artifact has an unexpected format: {filename}")
    return package


@lru_cache(maxsize=None)
def load_mlp_bundle(model_id: str = "mlp_residual_eta") -> tuple[Any, Any, list[str]]:
    if torch is None or ResidualMLP is None:
        raise HTTPException(status_code=503, detail=f"PyTorch model code is not available: {ETA_MODELING_IMPORT_ERROR}")
    if model_id == "mlp_residual_eta":
        model_path = EXPERIMENT_MODELS_DIR / "mlp_residual.pt"
        preprocessor_path = EXPERIMENT_MODELS_DIR / "mlp_residual_preprocessor.joblib"
    elif model_id == "hour_bin_mlp_residual_eta":
        model_path = EXPERIMENT_MODELS_DIR / "hour_bin_mlp_residual_eta.pt"
        preprocessor_path = EXPERIMENT_MODELS_DIR / "hour_bin_mlp_residual_eta_preprocessor.joblib"
    else:
        raise HTTPException(status_code=404, detail=f"Unsupported MLP model_id: {model_id}")
    if not model_path.exists() or not preprocessor_path.exists():
        raise HTTPException(status_code=503, detail=f"MLP residual artifacts are missing for {model_id}.")

    checkpoint = torch.load(model_path, map_location="cpu")
    cfg = checkpoint.get("config", {})
    model = ResidualMLP(
        int(checkpoint["input_dim"]),
        hidden_sizes=[int(x) for x in cfg.get("hidden_sizes", [128, 64, 32])],
        dropout=float(cfg.get("dropout", 0.15)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor, checkpoint.get("features") or []


def predict_xgb(model_id: str, frame: pd.DataFrame) -> float:
    package = load_joblib_experiment_model(f"{model_id}.joblib")
    features = package.get("features") or []
    x = package["preprocessor"].transform(frame[features])
    raw = float(np.asarray(package["model"].predict(x)).reshape(-1)[0])
    if model_id in {"xgb_residual_eta", "hour_bin_xgb_residual_eta"}:
        raw = float(frame["baseline_eta_secs"].iloc[0]) + raw
    return max(raw, 0.0)


def predict_mlp(frame: pd.DataFrame, model_id: str = "mlp_residual_eta") -> float:
    if predict_residual is None:
        raise HTTPException(status_code=503, detail=f"PyTorch model code is not available: {ETA_MODELING_IMPORT_ERROR}")
    model, preprocessor, features = load_mlp_bundle(model_id)
    x = np.asarray(preprocessor.transform(frame[features]), dtype=np.float32)
    residual = float(np.asarray(predict_residual(model, x)).reshape(-1)[0])
    return max(float(frame["baseline_eta_secs"].iloc[0]) + residual, 0.0)


@lru_cache(maxsize=None)
def load_deep_bundle(model_id: str = "deepr_eta_like") -> tuple[Any, Any]:
    if torch is None or DeepRETALikeModel is None:
        raise HTTPException(status_code=503, detail=f"DeepETA-like model code is not available: {ETA_MODELING_IMPORT_ERROR}")
    if model_id not in {"deepr_eta_like", "hour_bin_deepr_eta_like"}:
        raise HTTPException(status_code=404, detail=f"Unsupported DeeprETA-like model_id: {model_id}")
    model_path = EXPERIMENT_MODELS_DIR / f"{model_id}.pt"
    encoder_path = EXPERIMENT_MODELS_DIR / f"{model_id}_encoder.joblib"
    if not model_path.exists() or not encoder_path.exists():
        raise HTTPException(status_code=503, detail=f"DeeprETA-like artifacts are missing for {model_id}.")

    checkpoint = torch.load(model_path, map_location="cpu")
    cfg = checkpoint.get("config", {})
    encoder = joblib.load(encoder_path)
    model = DeepRETALikeModel(
        checkpoint["cardinalities"],
        embedding_dim=int(cfg.get("embedding_dim", 16)),
        hidden_sizes=[int(x) for x in cfg.get("hidden_sizes", [128, 64])],
        dropout=float(cfg.get("dropout", 0.15)),
        use_attention=bool(cfg.get("use_attention", False)),
        calibration_feature=cfg.get("calibration_feature") if cfg.get("use_calibration", True) else None,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, encoder


def predict_deep(frame: pd.DataFrame, model_id: str = "deepr_eta_like") -> float:
    if predict_deepr_eta is None:
        raise HTTPException(status_code=503, detail=f"DeepETA-like model code is not available: {ETA_MODELING_IMPORT_ERROR}")
    model, encoder = load_deep_bundle(model_id)
    return float(np.asarray(predict_deepr_eta(model, encoder, frame)).reshape(-1)[0])


def predict_hour_bin_median(frame: pd.DataFrame) -> float:
    path = EXPERIMENT_MODELS_DIR / "hour_bin_median_eta.joblib"
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Model artifact is missing: {path.relative_to(ROOT_DIR)}")
    package = joblib.load(path)
    medians = package.get("hour_bin_median_residual_secs") or {}
    global_median = as_float(package.get("global_median_residual_secs"), 0.0)
    hour_bin = as_str(frame["hour_bin"].iloc[0], "other")
    residual = as_float(medians.get(hour_bin), global_median)
    return max(float(frame["baseline_eta_secs"].iloc[0]) + residual, 0.0)


def raw_time_bin_for_hour(hour: int) -> str:
    if 4 <= hour <= 6:
        return "early_morning"
    if 7 <= hour <= 9:
        return "morning_peak"
    if 10 <= hour <= 14:
        return "off_peak_day"
    if 15 <= hour <= 18:
        return "evening_peak"
    if 19 <= hour <= 21:
        return "late_evening"
    return "other"


@lru_cache(maxsize=None)
def load_raw_time_bin_residuals() -> dict[str, Any]:
    if not RAW_TIME_BIN_RESIDUALS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Raw time-bin residual artifact is missing: {RAW_TIME_BIN_RESIDUALS_PATH.relative_to(ROOT_DIR)}",
        )
    try:
        residuals = pd.read_csv(RAW_TIME_BIN_RESIDUALS_PATH)
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Cannot read raw time-bin residual artifact: {error}") from error

    required_columns = {"time_bin", "median_residual_secs"}
    missing_columns = required_columns - set(residuals.columns)
    if missing_columns:
        raise HTTPException(
            status_code=503,
            detail=f"Raw time-bin residual artifact is missing columns: {sorted(missing_columns)}",
        )

    residual_map = {
        as_str(row["time_bin"], "other"): as_float(row["median_residual_secs"], 0.0)
        for _, row in residuals.iterrows()
    }
    global_median = as_float(residual_map.get("other"), 0.0)
    if "other" not in residual_map:
        global_median = as_float(residuals["median_residual_secs"].median(), 0.0)
        residual_map["other"] = global_median

    return {
        "residual_map": residual_map,
        "global_median_residual_secs": global_median,
    }


def predict_raw_time_bin_median(payload: EtaPredictRequest, hour: int) -> tuple[float, dict[str, Any]]:
    if payload.baseline_eta_secs is None or payload.baseline_eta_secs <= 0:
        raise HTTPException(status_code=422, detail="baseline_eta_secs from Vietmap is required for this model.")

    package = load_raw_time_bin_residuals()
    hour_bin = raw_time_bin_for_hour(hour)
    residual_map = package["residual_map"]
    global_median = as_float(package["global_median_residual_secs"], 0.0)
    residual = as_float(residual_map.get(hour_bin), global_median)
    pred_eta = max(float(payload.baseline_eta_secs) + residual, 0.0)

    return pred_eta, {
        "hour_bin": hour_bin,
        "residual_secs": residual,
        "residual_source": "time_bin" if hour_bin in residual_map else "global_fallback",
        "residual_artifact": str(RAW_TIME_BIN_RESIDUALS_PATH.relative_to(ROOT_DIR)),
    }


def predict_experiment_model(model_id: str, payload: EtaPredictRequest, hour: int) -> dict[str, Any]:
    frame = None
    if model_id == "api_raw_time_bin_median_residual":
        pred_eta, extra_metadata = predict_raw_time_bin_median(payload, hour)
        baseline = float(payload.baseline_eta_secs)
    else:
        frame = experiment_feature_frame(payload, hour)
        baseline = float(frame["baseline_eta_secs"].iloc[0])
        extra_metadata = {}

    if model_id == "vietmap_baseline":
        pred_eta = baseline
    elif model_id in {"xgb_direct_eta", "xgb_residual_eta", "hour_bin_xgb_residual_eta"}:
        if frame is None:
            raise HTTPException(status_code=500, detail="Experiment feature frame was not initialized.")
        pred_eta = predict_xgb(model_id, frame)
    elif model_id == "hour_bin_median_eta":
        if frame is None:
            raise HTTPException(status_code=500, detail="Experiment feature frame was not initialized.")
        pred_eta = predict_hour_bin_median(frame)
    elif model_id == "api_raw_time_bin_median_residual":
        pass
    elif model_id in {"mlp_residual_eta", "hour_bin_mlp_residual_eta"}:
        if frame is None:
            raise HTTPException(status_code=500, detail="Experiment feature frame was not initialized.")
        pred_eta = predict_mlp(frame, model_id)
    elif model_id in {"deepr_eta_like", "hour_bin_deepr_eta_like"}:
        if frame is None:
            raise HTTPException(status_code=500, detail="Experiment feature frame was not initialized.")
        pred_eta = predict_deep(frame, model_id)
    else:
        raise HTTPException(status_code=404, detail=f"Unknown experiment model_id: {model_id}")

    return {
        "hour": hour,
        "point": format_prediction(pred_eta),
        "quantiles": {},
        "baseline": format_prediction(baseline),
        "predicted_residual_secs": float(pred_eta - baseline),
        "metadata": extra_metadata,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/eta/models")
def eta_models() -> dict[str, Any]:
    return {
        "default_model_id": resolve_model_id(None),
        "route": route_payload(),
        "models": available_model_specs(),
    }


@app.get("/api/eta/model-info")
def model_info(model_id: str | None = Query(default=None)) -> dict[str, Any]:
    selected_model_id = resolve_model_id(model_id)
    selected = get_model_spec(selected_model_id)
    legacy_info = {
        "artifact": str(LEGACY_ARTIFACT_PATH.relative_to(ROOT_DIR)),
        "model_name": LEGACY_MODEL_PACKAGE.get("model_name"),
        "selection_policy": LEGACY_MODEL_PACKAGE.get("selection_policy"),
        "cv_best_model_name": LEGACY_MODEL_PACKAGE.get("cv_best_model_name"),
        "selected_quantile_model_names": LEGACY_MODEL_PACKAGE.get("selected_quantile_model_names", {}),
        "feature_columns": LEGACY_MODEL_PACKAGE.get("feature_columns", []),
        "target_column": LEGACY_MODEL_PACKAGE.get("target_column"),
        "holdout_metrics": LEGACY_MODEL_PACKAGE.get("holdout_metrics", {}),
        "quantile_holdout_metrics": LEGACY_MODEL_PACKAGE.get("quantile_holdout_metrics", []),
    }
    return {
        **legacy_info,
        "selected_model_id": selected_model_id,
        "selected_model": selected,
        "available_models": available_model_specs(),
        "route": route_payload(),
    }


@app.post("/api/eta/predict")
def predict_eta(payload: EtaPredictRequest) -> dict[str, Any]:
    model_id = resolve_model_id(payload.model_id)
    spec = get_model_spec(model_id)
    if not spec["available"]:
        detail = spec.get("error") or f"Model is unavailable. Missing artifacts: {spec.get('missing_artifacts')}"
        raise HTTPException(status_code=503, detail=detail)

    hour = request_hour(payload)
    if model_id == LEGACY_MODEL_ID:
        prediction = predict_legacy_for_hour(hour)
    else:
        prediction = predict_experiment_model(model_id, payload, hour)

    return {
        "route": route_payload(),
        "model_id": model_id,
        "model_name": spec["label"],
        "model_type": spec["model_type"],
        "target_type": spec["target_type"],
        "requires_baseline": spec["requires_baseline"],
        "selection_policy": LEGACY_MODEL_PACKAGE.get("selection_policy") if model_id == LEGACY_MODEL_ID else "frontend_selected",
        "cv_best_model_name": LEGACY_MODEL_PACKAGE.get("cv_best_model_name") if model_id == LEGACY_MODEL_ID else None,
        "selected_quantile_model_names": LEGACY_MODEL_PACKAGE.get("selected_quantile_model_names", {}) if model_id == LEGACY_MODEL_ID else {},
        "input": {
            "departure_time": payload.departure_time.isoformat() if payload.departure_time else None,
            "hour": hour,
            "baseline_eta_secs": payload.baseline_eta_secs,
            "origin_rain": payload.origin_rain,
            "destination_rain": payload.destination_rain,
            "rain": payload.rain,
        },
        "prediction": prediction,
    }
