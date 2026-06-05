from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow

from .data_loading import resolve_path

LOGGER = logging.getLogger(__name__)


def ensure_artifact_dirs(config: dict[str, Any]) -> dict[str, Path]:
    """Create configured artifact directories and return resolved paths."""
    paths = config.get("paths", {})
    resolved = {}
    for key in ["artifacts_dir", "plots_dir", "metrics_dir", "models_dir", "predictions_dir"]:
        raw = paths.get(key, key)
        path = resolve_path(config, raw)
        path.mkdir(parents=True, exist_ok=True)
        resolved[key] = path
    return resolved


def configure_mlflow(config: dict[str, Any], experiment_name: str) -> None:
    tracking_uri = config.get("mlflow", {}).get("tracking_uri", "file:./mlruns")
    if tracking_uri.startswith("file:"):
        raw_file_path = tracking_uri.removeprefix("file:")
        file_path = Path(raw_file_path)
        if not file_path.is_absolute():
            tracking_uri = "file:" + str(resolve_path(config, raw_file_path))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    LOGGER.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())


def flatten_params(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_params(value, full_key))
        elif isinstance(value, (list, tuple)):
            flat[full_key] = json.dumps(value)
        elif value is None or isinstance(value, (str, int, float, bool)):
            flat[full_key] = value
        else:
            flat[full_key] = str(value)
    return flat


def log_params_safe(params: dict[str, Any]) -> None:
    for key, value in flatten_params(params).items():
        safe_key = key[:250]
        try:
            mlflow.log_param(safe_key, value)
        except Exception as exc:
            LOGGER.warning("Could not log MLflow param %s=%s: %s", key, value, exc)


def log_metrics_safe(metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        try:
            if value == value:
                mlflow.log_metric(key[:250], float(value))
        except Exception as exc:
            LOGGER.warning("Could not log MLflow metric %s=%s: %s", key, value, exc)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)
