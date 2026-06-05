from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute ETA regression metrics in seconds."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_error = np.abs(y_true - y_pred)
    denom = np.maximum(np.abs(y_true), 1e-9)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mape": float(np.mean(abs_error / denom) * 100.0),
        "median_absolute_error": float(np.median(abs_error)),
        "p50": float(np.percentile(abs_error, 50)),
        "p90": float(np.percentile(abs_error, 90)),
        "p95": float(np.percentile(abs_error, 95)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
    }


def prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def eta_metrics_with_baseline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """Compute model ETA metrics and baseline-relative improvements."""
    model_metrics = regression_metrics(y_true, y_pred)
    baseline_metrics = regression_metrics(y_true, baseline_pred)
    output = prefixed_metrics(prefix, model_metrics)
    output[f"{prefix}_baseline_mae"] = baseline_metrics["mae"]
    output[f"{prefix}_baseline_mape"] = baseline_metrics["mape"]
    output[f"{prefix}_baseline_p50"] = baseline_metrics["p50"]
    output[f"{prefix}_baseline_p95"] = baseline_metrics["p95"]
    output[f"{prefix}_mae_improvement_pct"] = (
        (baseline_metrics["mae"] - model_metrics["mae"]) / baseline_metrics["mae"] * 100.0
        if baseline_metrics["mae"] > 0
        else float("nan")
    )
    output[f"{prefix}_p95_improvement_pct"] = (
        (baseline_metrics["p95"] - model_metrics["p95"]) / baseline_metrics["p95"] * 100.0
        if baseline_metrics["p95"] > 0
        else float("nan")
    )
    return output
