from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "output_log.csv"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "residual_modeling" / "artifacts"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
PRIMARY_SELECTION_METRIC = "mae"
K_CANDIDATES = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200]
RATIO_CLIP_QUANTILES = (0.01, 0.99)
MIN_AFFINE_BIN_ROWS = 10
HUBER_EPSILON = 1.35
HUBER_ALPHA = 0.0001
HUBER_MAX_ITER = 1000

TIME_BIN_ORDER = [
    "early_morning",
    "morning_peak",
    "off_peak_day",
    "evening_peak",
    "late_evening",
    "other",
]
TIME_BIN_DEFINITIONS = {
    "early_morning": "4-6",
    "morning_peak": "7-9",
    "off_peak_day": "10-14",
    "evening_peak": "15-18",
    "late_evening": "19-21",
    "other": "outside configured bins",
}


def assign_time_bin(hour: float | int) -> str:
    if pd.isna(hour):
        return "other"
    hour = int(hour)
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


def chronological_split(
    frame: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1.")
    n_rows = len(frame)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))
    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:val_end].copy(),
        frame.iloc[val_end:].copy(),
    )


def load_eta_data(data_path: Path) -> pd.DataFrame:
    required_columns = {
        "stationId",
        "destination_stationId",
        "hour",
        "lat",
        "lng",
        "destination_lat",
        "destination_lng",
        "delta_time",
        "estimate_time",
        "timestamp",
    }

    raw_df = pd.read_csv(data_path)
    missing_columns = sorted(required_columns - set(raw_df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    numeric_columns = [
        "delta_time",
        "estimate_time",
        "hour",
        "lat",
        "lng",
        "destination_lat",
        "destination_lng",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_mask = (
        df["timestamp"].isna()
        | df["delta_time"].isna()
        | df["estimate_time"].isna()
        | (df["delta_time"] <= 0)
        | (df["estimate_time"] <= 0)
    )
    if invalid_mask.any():
        print(f"Dropping {int(invalid_mask.sum())} invalid rows.")
        df = df.loc[~invalid_mask].copy()

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["api_eta_secs"] = df["estimate_time"].astype(float)
    df["actual_eta_secs"] = df["delta_time"].astype(float)
    df["residual_secs"] = df["actual_eta_secs"] - df["api_eta_secs"]
    df["ratio"] = df["actual_eta_secs"] / df["api_eta_secs"]
    df["log_ratio"] = np.log(df["ratio"])
    df["time_bin"] = df["hour"].map(assign_time_bin)
    df["od_pair"] = (
        df["stationId"].astype(str) + "->" + df["destination_stationId"].astype(str)
    )
    return df


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_error = np.abs(y_true - y_pred)
    denom = np.maximum(np.abs(y_true), 1e-9)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mape": float(np.mean(abs_error / denom) * 100.0),
        "median_absolute_error": float(np.median(abs_error)),
        "p50": float(np.percentile(abs_error, 50)),
        "p90": float(np.percentile(abs_error, 90)),
        "p95": float(np.percentile(abs_error, 95)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def evaluate_method(
    split_name: str,
    frame: pd.DataFrame,
    method_name: str,
    prediction_column: str,
) -> dict[str, float | str | int]:
    api_metrics = regression_metrics(frame["actual_eta_secs"], frame["api_baseline_eta_secs"])
    method_metrics = regression_metrics(frame["actual_eta_secs"], frame[prediction_column])
    return {
        "split": split_name,
        "method": method_name,
        "rows": len(frame),
        "mae": method_metrics["mae"],
        "rmse": method_metrics["rmse"],
        "mape": method_metrics["mape"],
        "p50": method_metrics["p50"],
        "p90": method_metrics["p90"],
        "p95": method_metrics["p95"],
        "r2": method_metrics["r2"],
        "mae_improvement_vs_api_pct": (
            (api_metrics["mae"] - method_metrics["mae"]) / api_metrics["mae"] * 100.0
        ),
        "p95_improvement_vs_api_pct": (
            (api_metrics["p95"] - method_metrics["p95"]) / api_metrics["p95"] * 100.0
        ),
    }


def aggregate_bin_target(
    train_df: pd.DataFrame,
    target_column: str,
    output_prefix: str,
    global_value: float,
) -> pd.DataFrame:
    table = (
        train_df.groupby("time_bin", observed=True)
        .agg(
            train_rows=(target_column, "size"),
            median_value=(target_column, "median"),
            mean_value=(target_column, "mean"),
            p25_value=(target_column, lambda values: float(np.percentile(values, 25))),
            p75_value=(target_column, lambda values: float(np.percentile(values, 75))),
        )
        .reindex(TIME_BIN_ORDER)
    )
    table["fallback_used_when_missing"] = table["median_value"].isna()
    table["median_value"] = table["median_value"].fillna(global_value)
    table["train_rows"] = table["train_rows"].fillna(0).astype(int)
    return table.rename(
        columns={
            "median_value": f"median_{output_prefix}",
            "mean_value": f"mean_{output_prefix}",
            "p25_value": f"p25_{output_prefix}",
            "p75_value": f"p75_{output_prefix}",
        }
    )


def smooth_scalar_table(
    raw_stats: pd.DataFrame,
    value_column: str,
    global_value: float,
    k: float,
    output_column: str,
) -> pd.DataFrame:
    table = raw_stats.copy()
    n = table["train_rows"].astype(float)
    if k < 0:
        raise ValueError("k must be non-negative.")
    if k == 0:
        weight = pd.Series(np.where(n > 0, 1.0, 0.0), index=table.index)
    else:
        weight = n / (n + float(k))
    table["k"] = float(k)
    table["shrinkage_weight"] = weight.astype(float)
    table["global_value"] = float(global_value)
    table[output_column] = (
        table["shrinkage_weight"] * table[value_column]
        + (1.0 - table["shrinkage_weight"]) * float(global_value)
    )
    table.loc[table["train_rows"].eq(0), output_column] = float(global_value)
    return table


def select_smoothed_scalar(
    train_stats: pd.DataFrame,
    value_column: str,
    global_value: float,
    val_df: pd.DataFrame,
    prediction_fn,
    output_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    rows = []
    for k in K_CANDIDATES:
        table = smooth_scalar_table(train_stats, value_column, global_value, k, output_column)
        value_map = table[output_column].to_dict()
        val_pred = prediction_fn(val_df, value_map, global_value)
        val_metrics = regression_metrics(val_df["actual_eta_secs"], val_pred)
        rows.append({"k": k, **{f"val_{key}": value for key, value in val_metrics.items()}})

    search_df = pd.DataFrame(rows)
    best_row = search_df.sort_values(
        [f"val_{PRIMARY_SELECTION_METRIC}", "val_p95", "k"]
    ).iloc[0]
    best_k = float(best_row["k"])
    selected_table = smooth_scalar_table(
        train_stats, value_column, global_value, best_k, output_column
    )
    return selected_table, search_df, best_k


def predict_additive(
    frame: pd.DataFrame,
    value_map: dict[str, float],
    fallback_value: float,
) -> pd.Series:
    residual = frame["time_bin"].map(value_map).fillna(fallback_value).astype(float)
    return frame["api_eta_secs"] + residual


def predict_ratio(
    frame: pd.DataFrame,
    value_map: dict[str, float],
    fallback_value: float,
) -> pd.Series:
    ratio = frame["time_bin"].map(value_map).fillna(fallback_value).astype(float)
    return frame["api_eta_secs"] * ratio


def predict_log_ratio(
    frame: pd.DataFrame,
    value_map: dict[str, float],
    fallback_value: float,
) -> pd.Series:
    log_ratio = frame["time_bin"].map(value_map).fillna(fallback_value).astype(float)
    return frame["api_eta_secs"] * np.exp(log_ratio)


def fit_global_affine(train_df: pd.DataFrame) -> tuple[float, float]:
    model = HuberRegressor(
        epsilon=HUBER_EPSILON,
        alpha=HUBER_ALPHA,
        max_iter=HUBER_MAX_ITER,
    )
    model.fit(train_df[["api_eta_secs"]], train_df["actual_eta_secs"])
    return float(model.coef_[0]), float(model.intercept_)


def fit_time_bin_affine(
    train_df: pd.DataFrame,
    global_a: float,
    global_b: float,
    min_rows: int = MIN_AFFINE_BIN_ROWS,
) -> pd.DataFrame:
    rows = []
    for time_bin in TIME_BIN_ORDER:
        bin_df = train_df.loc[train_df["time_bin"].eq(time_bin)]
        fit_used = len(bin_df) >= min_rows
        if fit_used:
            a, b = fit_global_affine(bin_df)
        else:
            a, b = global_a, global_b
        rows.append(
            {
                "time_bin": time_bin,
                "train_rows": len(bin_df),
                "a": float(a),
                "b": float(b),
                "fit_used": bool(fit_used),
                "fallback_used_when_missing": not fit_used,
            }
        )
    return pd.DataFrame(rows).set_index("time_bin")


def smooth_affine_table(
    raw_table: pd.DataFrame,
    global_a: float,
    global_b: float,
    k: float,
) -> pd.DataFrame:
    table = raw_table.copy()
    n = table["train_rows"].astype(float)
    if k < 0:
        raise ValueError("k must be non-negative.")
    if k == 0:
        weight = pd.Series(np.where(table["fit_used"], 1.0, 0.0), index=table.index)
    else:
        weight = n / (n + float(k))
        weight = weight.where(table["fit_used"], 0.0)
    table["k"] = float(k)
    table["shrinkage_weight"] = weight.astype(float)
    table["global_a"] = float(global_a)
    table["global_b"] = float(global_b)
    table["smoothed_a"] = (
        table["shrinkage_weight"] * table["a"]
        + (1.0 - table["shrinkage_weight"]) * float(global_a)
    )
    table["smoothed_b"] = (
        table["shrinkage_weight"] * table["b"]
        + (1.0 - table["shrinkage_weight"]) * float(global_b)
    )
    return table


def predict_affine(frame: pd.DataFrame, a_map: dict[str, float], b_map: dict[str, float]) -> pd.Series:
    a = frame["time_bin"].map(a_map).astype(float)
    b = frame["time_bin"].map(b_map).astype(float)
    return a * frame["api_eta_secs"] + b


def select_smoothed_affine(
    raw_table: pd.DataFrame,
    global_a: float,
    global_b: float,
    val_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    rows = []
    for k in K_CANDIDATES:
        table = smooth_affine_table(raw_table, global_a, global_b, k)
        val_pred = predict_affine(
            val_df,
            table["smoothed_a"].to_dict(),
            table["smoothed_b"].to_dict(),
        )
        val_metrics = regression_metrics(val_df["actual_eta_secs"], val_pred)
        rows.append({"k": k, **{f"val_{key}": value for key, value in val_metrics.items()}})

    search_df = pd.DataFrame(rows)
    best_row = search_df.sort_values(
        [f"val_{PRIMARY_SELECTION_METRIC}", "val_p95", "k"]
    ).iloc[0]
    best_k = float(best_row["k"])
    selected_table = smooth_affine_table(raw_table, global_a, global_b, best_k)
    return selected_table, search_df, best_k


def attach_prediction(frame: pd.DataFrame, column: str, prediction: pd.Series | np.ndarray) -> None:
    frame[column] = np.asarray(prediction, dtype=float)
    frame[f"{column}_abs_error_secs"] = (
        frame["actual_eta_secs"] - frame[column]
    ).abs()


def make_split_summary(split_frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    return [
        {
            "split": split_name,
            "rows": len(frame),
            "timestamp_min": frame["timestamp"].min(),
            "timestamp_max": frame["timestamp"].max(),
        }
        for split_name, frame in split_frames.items()
    ]


def to_json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.reset_index().astype(object).where(pd.notna(frame.reset_index()), None).to_dict(
        orient="records"
    )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def build_comparison(data_path: Path, artifact_dir: Path) -> dict[str, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    df = load_eta_data(data_path)
    train_df, val_df, test_df = chronological_split(df)
    split_frames = {"train": train_df, "val": val_df, "test": test_df}

    ratio_low, ratio_high = (
        float(train_df["ratio"].quantile(RATIO_CLIP_QUANTILES[0])),
        float(train_df["ratio"].quantile(RATIO_CLIP_QUANTILES[1])),
    )
    for split_df in split_frames.values():
        split_df["ratio_clipped"] = split_df["ratio"].clip(ratio_low, ratio_high)
        split_df["log_ratio_clipped"] = np.log(split_df["ratio_clipped"])
        split_df["api_baseline_eta_secs"] = split_df["api_eta_secs"]
        split_df["api_baseline_eta_secs_abs_error_secs"] = (
            split_df["actual_eta_secs"] - split_df["api_baseline_eta_secs"]
        ).abs()

    train_df = split_frames["train"]
    val_df = split_frames["val"]

    global_additive = float(train_df["residual_secs"].median())
    additive_stats = aggregate_bin_target(
        train_df, "residual_secs", "residual_secs", global_additive
    )
    additive_smoothed, additive_search, additive_best_k = select_smoothed_scalar(
        additive_stats,
        "median_residual_secs",
        global_additive,
        val_df,
        predict_additive,
        "smoothed_residual_secs",
    )

    global_ratio = float(train_df["ratio_clipped"].median())
    ratio_stats = aggregate_bin_target(train_df, "ratio_clipped", "ratio", global_ratio)
    ratio_smoothed, ratio_search, ratio_best_k = select_smoothed_scalar(
        ratio_stats,
        "median_ratio",
        global_ratio,
        val_df,
        predict_ratio,
        "smoothed_ratio",
    )

    global_log_ratio = float(train_df["log_ratio_clipped"].median())
    log_ratio_stats = aggregate_bin_target(
        train_df, "log_ratio_clipped", "log_ratio", global_log_ratio
    )
    log_ratio_smoothed, log_ratio_search, log_ratio_best_k = select_smoothed_scalar(
        log_ratio_stats,
        "median_log_ratio",
        global_log_ratio,
        val_df,
        predict_log_ratio,
        "smoothed_log_ratio",
    )

    global_a, global_b = fit_global_affine(train_df)
    affine_raw = fit_time_bin_affine(train_df, global_a, global_b)
    affine_smoothed, affine_search, affine_best_k = select_smoothed_affine(
        affine_raw,
        global_a,
        global_b,
        val_df,
    )

    additive_raw_map = additive_stats["median_residual_secs"].to_dict()
    additive_smoothed_map = additive_smoothed["smoothed_residual_secs"].to_dict()
    ratio_raw_map = ratio_stats["median_ratio"].to_dict()
    ratio_smoothed_map = ratio_smoothed["smoothed_ratio"].to_dict()
    log_ratio_raw_map = log_ratio_stats["median_log_ratio"].to_dict()
    log_ratio_smoothed_map = log_ratio_smoothed["smoothed_log_ratio"].to_dict()

    raw_affine_a_map = affine_raw["a"].to_dict()
    raw_affine_b_map = affine_raw["b"].to_dict()
    smoothed_affine_a_map = affine_smoothed["smoothed_a"].to_dict()
    smoothed_affine_b_map = affine_smoothed["smoothed_b"].to_dict()

    prediction_columns = {
        "api_eta": "api_baseline_eta_secs",
        "additive_global": "additive_global_eta_secs",
        "additive_time_bin": "additive_time_bin_eta_secs",
        "additive_smoothed_time_bin": "additive_smoothed_time_bin_eta_secs",
        "ratio_global": "ratio_global_eta_secs",
        "ratio_time_bin": "ratio_time_bin_eta_secs",
        "ratio_smoothed_time_bin": "ratio_smoothed_time_bin_eta_secs",
        "affine_global": "affine_global_eta_secs",
        "affine_time_bin": "affine_time_bin_eta_secs",
        "affine_smoothed_time_bin": "affine_smoothed_time_bin_eta_secs",
        "log_ratio_global": "log_ratio_global_eta_secs",
        "log_ratio_time_bin": "log_ratio_time_bin_eta_secs",
        "log_ratio_smoothed_time_bin": "log_ratio_smoothed_time_bin_eta_secs",
    }

    for split_df in split_frames.values():
        attach_prediction(split_df, "additive_global_eta_secs", split_df["api_eta_secs"] + global_additive)
        attach_prediction(
            split_df,
            "additive_time_bin_eta_secs",
            predict_additive(split_df, additive_raw_map, global_additive),
        )
        attach_prediction(
            split_df,
            "additive_smoothed_time_bin_eta_secs",
            predict_additive(split_df, additive_smoothed_map, global_additive),
        )
        attach_prediction(split_df, "ratio_global_eta_secs", split_df["api_eta_secs"] * global_ratio)
        attach_prediction(
            split_df,
            "ratio_time_bin_eta_secs",
            predict_ratio(split_df, ratio_raw_map, global_ratio),
        )
        attach_prediction(
            split_df,
            "ratio_smoothed_time_bin_eta_secs",
            predict_ratio(split_df, ratio_smoothed_map, global_ratio),
        )
        attach_prediction(
            split_df,
            "affine_global_eta_secs",
            global_a * split_df["api_eta_secs"] + global_b,
        )
        attach_prediction(
            split_df,
            "affine_time_bin_eta_secs",
            predict_affine(split_df, raw_affine_a_map, raw_affine_b_map),
        )
        attach_prediction(
            split_df,
            "affine_smoothed_time_bin_eta_secs",
            predict_affine(split_df, smoothed_affine_a_map, smoothed_affine_b_map),
        )
        attach_prediction(
            split_df,
            "log_ratio_global_eta_secs",
            split_df["api_eta_secs"] * np.exp(global_log_ratio),
        )
        attach_prediction(
            split_df,
            "log_ratio_time_bin_eta_secs",
            predict_log_ratio(split_df, log_ratio_raw_map, global_log_ratio),
        )
        attach_prediction(
            split_df,
            "log_ratio_smoothed_time_bin_eta_secs",
            predict_log_ratio(split_df, log_ratio_smoothed_map, global_log_ratio),
        )

    evaluation_rows = []
    for split_name, split_df in split_frames.items():
        for method_name, prediction_column in prediction_columns.items():
            evaluation_rows.append(
                evaluate_method(split_name, split_df, method_name, prediction_column)
            )
    metrics_df = pd.DataFrame(evaluation_rows)

    base_columns = [
        "split",
        "timestamp",
        "stationId",
        "destination_stationId",
        "hour",
        "time_bin",
        "api_eta_secs",
        "actual_eta_secs",
        "residual_secs",
        "ratio",
        "log_ratio",
    ]
    prediction_output_columns = [
        column
        for column in split_frames["train"].columns
        if column.endswith("_eta_secs") or column.endswith("_eta_secs_abs_error_secs")
    ]
    prediction_output_columns = [
        column for column in prediction_output_columns if column not in base_columns
    ]
    predictions_df = pd.concat(
        [
            frame.assign(split=split_name)[base_columns + prediction_output_columns]
            for split_name, frame in split_frames.items()
        ],
        ignore_index=True,
    )

    predictions_path = artifact_dir / "correction_method_comparison_predictions.csv"
    metrics_path = artifact_dir / "correction_method_comparison_metrics.csv"
    model_card_path = artifact_dir / "correction_method_comparison_model_card.json"

    predictions_df.to_csv(predictions_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    model_card = {
        "script": "residual_modeling/method_comparison.py",
        "data_path": str(data_path.relative_to(PROJECT_ROOT)),
        "target": "actual_eta_secs",
        "api_eta_column": "estimate_time",
        "actual_column": "delta_time",
        "split_strategy": "chronological_70_15_15",
        "selection_metric": PRIMARY_SELECTION_METRIC,
        "k_candidates": K_CANDIDATES,
        "time_bins": TIME_BIN_DEFINITIONS,
        "ratio_clip_quantiles": {
            "low_quantile": RATIO_CLIP_QUANTILES[0],
            "high_quantile": RATIO_CLIP_QUANTILES[1],
            "low_value": ratio_low,
            "high_value": ratio_high,
        },
        "methods": {
            "additive": {
                "global_formula": "corrected_eta = estimate_time + median(train.actual_eta - train.estimate_time)",
                "time_bin_formula": "corrected_eta = estimate_time + median(train.residual | train.time_bin)",
                "smoothed_formula": "corrected_eta = estimate_time + (w * time_bin_residual + (1 - w) * global_residual), w=n/(n+k)",
                "global_residual_secs": global_additive,
                "selected_k": additive_best_k,
                "time_bin_table": to_json_records(additive_smoothed),
                "smoothing_search": to_json_records(additive_search),
            },
            "ratio": {
                "global_formula": "corrected_eta = estimate_time * median(train.actual_eta / train.estimate_time)",
                "time_bin_formula": "corrected_eta = estimate_time * median(train.actual_eta / train.estimate_time | train.time_bin)",
                "smoothed_formula": "corrected_eta = estimate_time * (w * time_bin_ratio + (1 - w) * global_ratio), w=n/(n+k)",
                "global_ratio": global_ratio,
                "selected_k": ratio_best_k,
                "time_bin_table": to_json_records(ratio_smoothed),
                "smoothing_search": to_json_records(ratio_search),
            },
            "affine": {
                "global_formula": "corrected_eta = a * estimate_time + b",
                "time_bin_formula": "corrected_eta = a_time_bin * estimate_time + b_time_bin",
                "smoothed_formula": "corrected_eta = (w * a_time_bin + (1 - w) * a_global) * estimate_time + (w * b_time_bin + (1 - w) * b_global), w=n/(n+k)",
                "estimator": "sklearn.linear_model.HuberRegressor",
                "huber_epsilon": HUBER_EPSILON,
                "huber_alpha": HUBER_ALPHA,
                "huber_max_iter": HUBER_MAX_ITER,
                "min_affine_bin_rows": MIN_AFFINE_BIN_ROWS,
                "global_a": global_a,
                "global_b": global_b,
                "selected_k": affine_best_k,
                "time_bin_table": to_json_records(affine_smoothed),
                "smoothing_search": to_json_records(affine_search),
            },
            "log_ratio": {
                "global_formula": "corrected_eta = estimate_time * exp(median(log(train.actual_eta / train.estimate_time)))",
                "time_bin_formula": "corrected_eta = estimate_time * exp(median(log(train.actual_eta / train.estimate_time) | train.time_bin))",
                "smoothed_formula": "corrected_eta = estimate_time * exp(w * time_bin_log_ratio + (1 - w) * global_log_ratio), w=n/(n+k)",
                "global_log_ratio": global_log_ratio,
                "global_exp_log_ratio": float(np.exp(global_log_ratio)),
                "selected_k": log_ratio_best_k,
                "time_bin_table": to_json_records(log_ratio_smoothed),
                "smoothing_search": to_json_records(log_ratio_search),
            },
        },
        "split_summary": make_split_summary(split_frames),
        "metrics": metrics_df.to_dict(orient="records"),
        "artifacts": [
            str(predictions_path.relative_to(PROJECT_ROOT)),
            str(metrics_path.relative_to(PROJECT_ROOT)),
            str(model_card_path.relative_to(PROJECT_ROOT)),
        ],
    }

    model_card_path.write_text(
        json.dumps(to_jsonable(model_card), indent=2),
        encoding="utf-8",
    )

    return {
        "predictions": predictions_path,
        "metrics": metrics_path,
        "model_card": model_card_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare additive, ratio, affine, and log-ratio ETA corrections."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = build_comparison(args.data_path.resolve(), args.artifact_dir.resolve())
    print("Wrote artifacts:")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
