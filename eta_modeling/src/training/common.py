from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from ..data_loading import load_config, load_experiment_dataframe
from ..features import add_engineered_features, configured_feature_columns, get_feature_frame
from ..metrics import eta_metrics_with_baseline
from ..mlflow_utils import ensure_artifact_dirs, log_metrics_safe, log_params_safe, write_json
from ..preprocessing import split_date_ranges, time_based_split, validate_experiment_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def prepare_data(config: dict[str, Any]) -> dict[str, Any]:
    paths = ensure_artifact_dirs(config)
    raw_df, od_metadata = load_experiment_dataframe(config)
    warnings = validate_experiment_dataframe(raw_df, config.get("fixed_trip", {}).get("min_samples", 30))
    feature_cfg = config.get("features", {})
    df = add_engineered_features(
        raw_df,
        use_cyclic=feature_cfg.get("use_cyclic_time_features", True),
        use_geohash=feature_cfg.get("use_geohash", False),
    )
    train_df, val_df, test_df = time_based_split(
        df,
        train_ratio=config.get("split", {}).get("train_ratio", 0.7),
        val_ratio=config.get("split", {}).get("val_ratio", 0.15),
    )
    numeric, categorical = configured_feature_columns(train_df, config)
    date_ranges = split_date_ranges(train_df, val_df, test_df)
    return {
        "paths": paths,
        "df": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "feature_list": numeric + categorical,
        "od_metadata": od_metadata,
        "date_ranges": date_ranges,
        "validation_warnings": warnings,
    }


def split_xy(frame: pd.DataFrame, features: list[str], target: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    x = get_feature_frame(frame, [f for f in features if f in frame.columns and frame[f].dtype.kind in "biufc"], [])
    # Preserve original configured order and mixed dtypes.
    x = frame[features].copy()
    y = frame[target].to_numpy(dtype=float)
    actual = frame["actual_eta_secs"].to_numpy(dtype=float)
    baseline = frame["baseline_eta_secs"].to_numpy(dtype=float)
    return x, y, actual, baseline


def evaluate_all_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_train: np.ndarray,
    pred_val: np.ndarray,
    pred_test: np.ndarray,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, frame, pred in [
        ("train", train_df, pred_train),
        ("val", val_df, pred_val),
        ("test", test_df, pred_test),
    ]:
        metrics.update(
            eta_metrics_with_baseline(
                frame["actual_eta_secs"].to_numpy(dtype=float),
                pred,
                frame["baseline_eta_secs"].to_numpy(dtype=float),
                name,
            )
        )
    metrics["baseline_test_mae"] = metrics["test_baseline_mae"]
    metrics["baseline_test_mape"] = metrics["test_baseline_mape"]
    metrics["baseline_test_p95"] = metrics["test_baseline_p95"]
    metrics["test_mae_improvement_pct"] = metrics["test_mae_improvement_pct"]
    metrics["test_p95_improvement_pct"] = metrics["test_p95_improvement_pct"]
    return metrics


def prediction_frame(frame: pd.DataFrame, pred_eta: np.ndarray, model_name: str) -> pd.DataFrame:
    out = frame[
        [
            "timestamp",
            "stationId",
            "destination_stationId",
            "hour",
            "baseline_eta_secs",
            "actual_eta_secs",
            "residual_secs",
        ]
    ].copy()
    out["model_name"] = model_name
    out["pred_eta_secs"] = np.asarray(pred_eta, dtype=float)
    out["pred_residual_secs"] = out["pred_eta_secs"] - out["baseline_eta_secs"]
    out["absolute_error_secs"] = (out["actual_eta_secs"] - out["pred_eta_secs"]).abs()
    return out


def save_predictions(paths: dict[str, Path], model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, pred_train: np.ndarray, pred_val: np.ndarray, pred_test: np.ndarray) -> Path:
    pred = pd.concat(
        [
            prediction_frame(train_df, pred_train, model_name).assign(split="train"),
            prediction_frame(val_df, pred_val, model_name).assign(split="val"),
            prediction_frame(test_df, pred_test, model_name).assign(split="test"),
        ],
        ignore_index=True,
    )
    path = paths["predictions_dir"] / f"{model_name}_predictions.csv"
    pred.to_csv(path, index=False)
    mlflow.log_artifact(str(path))
    return path


def save_metrics(paths: dict[str, Path], model_name: str, metrics: dict[str, float]) -> Path:
    path = paths["metrics_dir"] / f"{model_name}_metrics.json"
    write_json(path, metrics)
    mlflow.log_artifact(str(path))
    return path


def plot_diagnostics(paths: dict[str, Path], model_name: str, test_df: pd.DataFrame, pred_eta: np.ndarray) -> list[Path]:
    y_true = test_df["actual_eta_secs"].to_numpy(dtype=float)
    baseline = test_df["baseline_eta_secs"].to_numpy(dtype=float)
    pred_eta = np.asarray(pred_eta, dtype=float)
    abs_error = np.abs(y_true - pred_eta)
    residual = y_true - baseline
    output_paths: list[Path] = []

    specs = [
        ("actual_vs_predicted", "Predicted ETA (secs)", pred_eta),
        ("actual_vs_baseline", "Vietmap baseline ETA (secs)", baseline),
    ]
    for suffix, xlabel, x_values in specs:
        plt.figure(figsize=(7, 5))
        plt.scatter(x_values, y_true, alpha=0.75)
        lo = float(min(np.min(x_values), np.min(y_true)))
        hi = float(max(np.max(x_values), np.max(y_true)))
        plt.plot([lo, hi], [lo, hi], linestyle="--", color="black")
        plt.xlabel(xlabel)
        plt.ylabel("Actual ETA (secs)")
        plt.title(f"{model_name}: {suffix}")
        path = paths["plots_dir"] / f"{model_name}_{suffix}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        output_paths.append(path)

    plt.figure(figsize=(7, 5))
    plt.hist(residual, bins=30)
    plt.xlabel("Actual - baseline (secs)")
    plt.ylabel("Count")
    plt.title(f"{model_name}: residual distribution")
    path = paths["plots_dir"] / f"{model_name}_residual_distribution.png"
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    output_paths.append(path)

    plt.figure(figsize=(7, 5))
    plt.hist(abs_error, bins=30)
    plt.xlabel("Absolute error (secs)")
    plt.ylabel("Count")
    plt.title(f"{model_name}: absolute error distribution")
    path = paths["plots_dir"] / f"{model_name}_absolute_error_distribution.png"
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    output_paths.append(path)

    plt.figure(figsize=(9, 5))
    plt.plot(test_df["timestamp"], abs_error, marker="o", linewidth=1)
    plt.xlabel("Timestamp")
    plt.ylabel("Absolute error (secs)")
    plt.title(f"{model_name}: error over time")
    plt.xticks(rotation=30, ha="right")
    path = paths["plots_dir"] / f"{model_name}_error_over_time.png"
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    output_paths.append(path)

    for artifact in output_paths:
        mlflow.log_artifact(str(artifact))
    return output_paths


def log_run_context(config: dict[str, Any], prepared: dict[str, Any], model_type: str, target_type: str, extra_params: dict[str, Any] | None = None) -> None:
    train_df = prepared["train_df"]
    val_df = prepared["val_df"]
    test_df = prepared["test_df"]
    params = {
        "model_type": model_type,
        "target_type": target_type,
        "feature_list": prepared["feature_list"],
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "fixed_stationId": prepared["od_metadata"]["stationId"],
        "fixed_destination_stationId": prepared["od_metadata"]["destination_stationId"],
        "split_strategy": "time_based",
        "date_ranges": prepared["date_ranges"],
        "validation_warnings": prepared["validation_warnings"],
    }
    if extra_params:
        params.update(extra_params)
    log_params_safe(params)
    mlflow.log_artifact(str(Path(config["_config_path"])))


def save_joblib_model(paths: dict[str, Path], filename: str, payload: Any) -> Path:
    path = paths["models_dir"] / filename
    joblib.dump(payload, path)
    mlflow.log_artifact(str(path))
    return path


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_args: object) -> None:
        self.elapsed = time.perf_counter() - self.start
