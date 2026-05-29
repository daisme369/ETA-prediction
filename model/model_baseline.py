"""
XGBoost baseline for ETA prediction with MLflow tracking and tracing.

Run from the repository root with the project virtual environment:
    .\\.venv\\Scripts\\python.exe model\\model_baseline.py

Open the MLflow UI:
    .\\.venv\\Scripts\\mlflow.exe ui --backend-store-uri .\\mlruns
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import (
    KFold,
    ParameterSampler,
    TimeSeriesSplit,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "mock_eta_trips.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "model" / "artifacts"
DEFAULT_TRACKING_URI = (ROOT_DIR / "mlruns").as_uri()
TARGET_COLUMN = "actual_eta_secs"
REQUEST_TIMESTAMP_COLUMN = "request_timestamp"
DEFAULT_RUN_NAME = "xgboost_eta_baseline"
MODEL_MODES = ("residual",)

BASE_FEATURE_COLUMNS = [
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

NUMERIC_FEATURES = [
    "origin_lng",
    "origin_lat",
    "destination_lng",
    "destination_lat",
    "hour_of_day",
    "is_rush_hour",
    "is_weekend",
    "is_holiday",
    "haversine_distance_meters",
    "baseline_distance_meters",
    "is_raining",
    "baseline_eta_secs",
    "day_of_week_num",
    "traffic_level_ord",
    "rain_level_ord",
    "weather_condition_ord",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "route_to_haversine_ratio",
    "baseline_speed_mps",
    "same_h3_cell",
]

CATEGORICAL_FEATURES = [
    "origin_h3",
    "destination_h3",
    "h3_pair",
    "day_of_week",
    "traffic_level",
    "rain_level",
    "weather_condition",
]

INTEGER_HYPERPARAMS = {"n_estimators", "max_depth", "min_child_weight"}
CV_STRATEGIES = {"auto", "kfold", "time"}


class ETAFeatureEngineer(BaseEstimator, TransformerMixin):
    """Normalize raw trip columns and add deterministic ETA features."""

    day_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    traffic_map = {"low": 0, "medium": 1, "high": 2, "severe": 3}
    rain_map = {"none": 0, "light": 1, "moderate": 2, "heavy": 3, "very_heavy": 4}
    weather_map = {"clear": 0, "cloudy": 1, "rain": 2, "storm": 3, "fog": 4}
    boolean_columns = ["is_rush_hour", "is_weekend", "is_holiday", "is_raining"]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ETAFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        for column in self.boolean_columns:
            df[column] = self._to_binary(df[column])

        for column in CATEGORICAL_FEATURES:
            if column in df.columns:
                df[column] = df[column].astype("string").fillna("unknown")

        df["day_of_week_num"] = self._map_string_column(df["day_of_week"], self.day_map)
        df["traffic_level_ord"] = self._map_string_column(
            df["traffic_level"], self.traffic_map
        )
        df["rain_level_ord"] = self._map_string_column(df["rain_level"], self.rain_map)
        df["weather_condition_ord"] = self._map_string_column(
            df["weather_condition"], self.weather_map
        )

        hour = pd.to_numeric(df["hour_of_day"], errors="coerce").fillna(0).clip(0, 23)
        day = df["day_of_week_num"].fillna(0).clip(0, 6)
        df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        df["day_sin"] = np.sin(2.0 * np.pi * day / 7.0)
        df["day_cos"] = np.cos(2.0 * np.pi * day / 7.0)

        haversine = pd.to_numeric(
            df["haversine_distance_meters"], errors="coerce"
        ).clip(lower=1.0)
        route_distance = pd.to_numeric(
            df["baseline_distance_meters"], errors="coerce"
        ).clip(lower=1.0)
        baseline_eta = pd.to_numeric(df["baseline_eta_secs"], errors="coerce").clip(
            lower=1.0
        )
        df["route_to_haversine_ratio"] = (route_distance / haversine).replace(
            [np.inf, -np.inf], np.nan
        )
        df["baseline_speed_mps"] = (route_distance / baseline_eta).replace(
            [np.inf, -np.inf], np.nan
        )

        origin_h3 = df["origin_h3"].astype("string").fillna("unknown")
        destination_h3 = df["destination_h3"].astype("string").fillna("unknown")
        df["same_h3_cell"] = (origin_h3 == destination_h3).astype(int)
        df["h3_pair"] = origin_h3 + "->" + destination_h3

        for column in NUMERIC_FEATURES:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    @staticmethod
    def _to_binary(series: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(series):
            return series.astype(int)
        normalized = series.astype("string").str.lower().str.strip()
        return normalized.map(
            {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0}
        ).fillna(pd.to_numeric(series, errors="coerce")).astype(float)

    @staticmethod
    def _map_string_column(series: pd.Series, mapping: dict[str, int]) -> pd.Series:
        return series.astype("string").str.strip().str.title().map(
            {key.title(): value for key, value in mapping.items()}
        )


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric outliers using quantiles learned only from the fit fold."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: np.ndarray, y: pd.Series | None = None) -> "QuantileClipper":
        array = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(array, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(array, self.upper_quantile, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        return np.clip(array, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        return np.asarray(input_features, dtype=object)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and tune an XGBoost ETA baseline with MLflow."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tracking-uri", type=str, default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment-name", type=str, default="eta_xgboost_baseline")
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument(
        "--split-strategy",
        choices=["random", "time"],
        default="time",
        help="Use random train_test_split or chronological split by request_timestamp.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=16)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument(
        "--cv-strategy",
        choices=sorted(CV_STRATEGIES),
        default="time",
        help="Use kfold, time-series CV, or infer from --split-strategy.",
    )
    parser.add_argument("--cv-jobs", type=int, default=1)
    parser.add_argument("--xgb-jobs", type=int, default=1)
    parser.add_argument(
        "--h3-min-count",
        type=int,
        default=5,
        help="Minimum test samples for a H3 pair to appear in error analysis.",
    )
    args = parser.parse_args()
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be greater than 0 and less than 1.")
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be a positive integer.")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")
    if args.h3_min_count <= 0:
        raise ValueError("--h3-min-count must be a positive integer.")
    return args


def load_dataset(path: Path) -> pd.DataFrame:
    with mlflow.start_span("load_dataset", attributes={"path": str(path)}):
        df = pd.read_csv(path)
        missing_columns = (set(BASE_FEATURE_COLUMNS) | {TARGET_COLUMN}) - set(df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Dataset is missing required columns: {missing}")
        if df[TARGET_COLUMN].isna().any():
            raise ValueError(f"Target column {TARGET_COLUMN!r} contains missing values.")
        return df


def make_train_test_split(
    df: pd.DataFrame, test_size: float, seed: int, split_strategy: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    with mlflow.start_span(
        "train_test_split",
        attributes={
            "test_size": test_size,
            "seed": seed,
            "split_strategy": split_strategy,
        },
    ):
        feature_columns = list(BASE_FEATURE_COLUMNS)

        if split_strategy == "time":
            if REQUEST_TIMESTAMP_COLUMN not in df.columns:
                raise ValueError(
                    f"--split-strategy time requires {REQUEST_TIMESTAMP_COLUMN!r} column."
                )

            timestamp = pd.to_datetime(df[REQUEST_TIMESTAMP_COLUMN], errors="coerce")
            if timestamp.isna().any():
                raise ValueError(
                    f"Column {REQUEST_TIMESTAMP_COLUMN!r} contains invalid timestamps."
                )

            sorted_df = (
                df.assign(_request_timestamp=timestamp)
                .sort_values("_request_timestamp", kind="mergesort")
                .drop(columns="_request_timestamp")
            )
            split_index = int((1.0 - test_size) * len(sorted_df))
            if split_index <= 0 or split_index >= len(sorted_df):
                raise ValueError(
                    "Time split produced an empty train or test set; adjust --test-size."
                )

            features = sorted_df[feature_columns].copy()
            target = sorted_df[TARGET_COLUMN].astype(float)
            return (
                features.iloc[:split_index],
                features.iloc[split_index:],
                target.iloc[:split_index],
                target.iloc[split_index:],
            )

        features = df[feature_columns].copy()
        target = df[TARGET_COLUMN].astype(float)
        return train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )


def build_pipeline(params: dict[str, Any], seed: int, xgb_jobs: int) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", QuantileClipper()),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model = XGBRegressor(
        objective="reg:absoluteerror",
        eval_metric="mae",
        tree_method="hist",
        random_state=seed,
        n_jobs=xgb_jobs,
        **params,
    )
    return Pipeline(
        steps=[
            ("features", ETAFeatureEngineer()),
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def sample_hyperparameters(n_trials: int, seed: int) -> list[dict[str, Any]]:
    distributions = {
        "n_estimators": [150, 250, 400, 600],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": loguniform(0.015, 0.20),
        "subsample": [0.70, 0.85, 1.00],
        "colsample_bytree": [0.70, 0.85, 1.00],
        "min_child_weight": [1, 3, 5, 8],
        "gamma": [0.0, 0.05, 0.10, 0.20],
        "reg_alpha": [0.0, 0.05, 0.10, 0.50],
        "reg_lambda": [0.75, 1.0, 1.5, 2.0, 3.0],
    }
    return list(ParameterSampler(distributions, n_iter=n_trials, random_state=seed))


def coerce_hyperparameter_types(params: dict[str, Any]) -> dict[str, Any]:
    typed_params: dict[str, Any] = {}
    for key, value in params.items():
        if key in INTEGER_HYPERPARAMS:
            typed_params[key] = int(round(float(value)))
        else:
            typed_params[key] = float(value)
    return typed_params


def resolve_cv_strategy(split_strategy: str, cv_strategy: str) -> str:
    if cv_strategy != "auto":
        return cv_strategy
    return "time" if split_strategy == "time" else "kfold"


def make_cv(cv_strategy: str, cv_folds: int, seed: int) -> KFold | TimeSeriesSplit:
    if cv_strategy == "time":
        return TimeSeriesSplit(n_splits=cv_folds)
    return KFold(n_splits=cv_folds, shuffle=True, random_state=seed)


def flatten_for_mlflow(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in payload.items()}


def make_residual_target(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    return y.astype(float) - X["baseline_eta_secs"].astype(float)


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int,
    cv_folds: int,
    cv_strategy: str,
    cv_jobs: int,
    xgb_jobs: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    with mlflow.start_span(
        "hyperparameter_tuning",
        attributes={
            "n_trials": n_trials,
            "cv_folds": cv_folds,
            "cv_strategy": cv_strategy,
            "cv_jobs": cv_jobs,
            "xgb_jobs": xgb_jobs,
        },
    ):
        cv = make_cv(cv_strategy=cv_strategy, cv_folds=cv_folds, seed=seed)
        scoring = {
            "mae": "neg_mean_absolute_error",
            "mape": "neg_mean_absolute_percentage_error",
        }
        records: list[dict[str, Any]] = []

        for trial_index, params in enumerate(sample_hyperparameters(n_trials, seed), start=1):
            with mlflow.start_run(run_name=f"trial_{trial_index:02d}", nested=True):
                mlflow.log_params(
                    {
                        **params,
                        "trial": trial_index,
                        "cv_strategy": cv_strategy,
                    }
                )
                span_attributes = {
                    "trial": trial_index,
                    "cv_strategy": cv_strategy,
                    **flatten_for_mlflow("param", params),
                }
                with mlflow.start_span("tune_trial", attributes=span_attributes):
                    pipeline = build_pipeline(params, seed=seed, xgb_jobs=xgb_jobs)
                    scores = cross_validate(
                        pipeline,
                        X_train,
                        y_train,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=cv_jobs,
                        error_score="raise",
                    )
                    metrics = {
                        "cv_mae_mean": float(-scores["test_mae"].mean()),
                        "cv_mae_std": float(scores["test_mae"].std()),
                        "cv_mape_mean": float(-scores["test_mape"].mean()),
                        "cv_mape_std": float(scores["test_mape"].std()),
                        "fit_time_mean": float(scores["fit_time"].mean()),
                    }
                    mlflow.log_metrics(metrics)
                    records.append({"trial": trial_index, **params, **metrics})
                    print(
                        f"Trial {trial_index:02d}/{n_trials} | "
                        f"CV MAE={metrics['cv_mae_mean']:.2f}s | "
                        f"CV MAPE={metrics['cv_mape_mean']:.4f}"
                    )

        results = pd.DataFrame(records).sort_values(
            ["cv_mae_mean", "cv_mape_mean"], ascending=[True, True]
        )
        best_params = {
            key: value
            for key, value in results.iloc[0].to_dict().items()
            if key
            not in {
                "trial",
                "cv_mae_mean",
                "cv_mae_std",
                "cv_mape_mean",
                "cv_mape_std",
                "fit_time_mean",
            }
        }
        return coerce_hyperparameter_types(best_params), results


def predict_eta(model: Pipeline, X: pd.DataFrame, model_mode: str) -> np.ndarray:
    raw_predictions = model.predict(X)
    if model_mode == "residual":
        raw_predictions = X["baseline_eta_secs"].to_numpy(dtype=float) + raw_predictions
    return np.maximum(raw_predictions, 1.0)


def regression_metrics(
    actual: np.ndarray, predictions: np.ndarray, prefix: str
) -> dict[str, float]:
    errors = predictions - actual
    abs_errors = np.abs(errors)
    return {
        f"{prefix}_mae": float(abs_errors.mean()),
        f"{prefix}_mape": float(mean_absolute_percentage_error(actual, predictions)),
        f"{prefix}_rmse": float(math.sqrt(np.mean(np.square(errors)))),
        f"{prefix}_bias": float(errors.mean()),
        f"{prefix}_p50_abs_error": float(np.quantile(abs_errors, 0.50)),
        f"{prefix}_p90_abs_error": float(np.quantile(abs_errors, 0.90)),
        f"{prefix}_underprediction_rate": float(np.mean(errors < 0.0)),
    }


def evaluate_predictions(
    split_name: str,
    model_name: str,
    actual: np.ndarray,
    predictions: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, float]:
    metrics = regression_metrics(actual, predictions, f"{split_name}_{model_name}")
    baseline_mae = float(mean_absolute_error(actual, baseline))
    model_mae = metrics[f"{split_name}_{model_name}_mae"]
    metrics[f"{split_name}_{model_name}_mae_improvement_vs_baseline"] = (
        baseline_mae - model_mae
    )
    metrics[f"{split_name}_{model_name}_mae_improvement_pct_vs_baseline"] = (
        (baseline_mae - model_mae) / baseline_mae if baseline_mae > 0.0 else 0.0
    )
    return metrics


def evaluate_model(
    split_name: str, model_name: str, model: Pipeline, X: pd.DataFrame, y: pd.Series
) -> tuple[dict[str, float], np.ndarray]:
    with mlflow.start_span(
        "evaluate_model",
        attributes={"split": split_name, "model_name": model_name},
    ):
        predictions = predict_eta(model, X, model_mode=model_name)
        actual = y.to_numpy(dtype=float)
        baseline = X["baseline_eta_secs"].to_numpy(dtype=float)
        metrics = evaluate_predictions(
            split_name=split_name,
            model_name=model_name,
            actual=actual,
            predictions=predictions,
            baseline=baseline,
        )
        return metrics, predictions


def extract_feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    booster = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = booster.feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def feature_importance_summary(
    feature_importance: pd.DataFrame, model_name: str
) -> dict[str, float]:
    total_importance = float(feature_importance["importance"].sum())
    if total_importance <= 0.0:
        total_importance = 1.0

    features = feature_importance["feature"].astype(str)
    baseline_mask = features.str.contains("baseline_eta_secs", regex=False)
    baseline_related_mask = features.str.contains("baseline_", regex=False)
    top_feature = str(feature_importance.iloc[0]["feature"])

    return {
        f"{model_name}_baseline_eta_importance": float(
            feature_importance.loc[baseline_mask, "importance"].sum()
        ),
        f"{model_name}_baseline_eta_importance_share": float(
            feature_importance.loc[baseline_mask, "importance"].sum() / total_importance
        ),
        f"{model_name}_baseline_related_importance_share": float(
            feature_importance.loc[baseline_related_mask, "importance"].sum()
            / total_importance
        ),
        f"{model_name}_top_feature_is_baseline_eta": float(
            top_feature == "num__baseline_eta_secs"
        ),
    }


def bucketize_duration(actual_eta_secs: pd.Series) -> pd.Series:
    return pd.cut(
        actual_eta_secs,
        bins=[0, 300, 600, 900, 1_200, 1_800, 2_400, 3_600, np.inf],
        labels=[
            "000-005m",
            "005-010m",
            "010-015m",
            "015-020m",
            "020-030m",
            "030-040m",
            "040-060m",
            "060m+",
        ],
        include_lowest=True,
    ).astype("string")


def bucketize_distance(distance_meters: pd.Series) -> pd.Series:
    return pd.cut(
        distance_meters,
        bins=[0, 1_000, 2_000, 5_000, 10_000, 15_000, 25_000, np.inf],
        labels=[
            "000-001km",
            "001-002km",
            "002-005km",
            "005-010km",
            "010-015km",
            "015-025km",
            "025km+",
        ],
        include_lowest=True,
    ).astype("string")


def bucketize_time_of_day(hour_of_day: pd.Series) -> pd.Series:
    hour = pd.to_numeric(hour_of_day, errors="coerce").fillna(-1).astype(int)
    labels = np.select(
        [
            hour.between(0, 5),
            hour.between(6, 9),
            hour.between(10, 15),
            hour.between(16, 19),
            hour.between(20, 23),
        ],
        ["night", "morning_peak", "midday", "evening_peak", "late_evening"],
        default="unknown",
    )
    return pd.Series(labels, index=hour_of_day.index, dtype="string")


def make_prediction_frame(
    X: pd.DataFrame,
    y: pd.Series,
    predictions_by_model: dict[str, np.ndarray],
) -> pd.DataFrame:
    frame = X.reset_index(drop=True).copy()
    frame["actual_eta_secs"] = y.reset_index(drop=True).astype(float)
    frame["baseline_prediction_secs"] = frame["baseline_eta_secs"].astype(float)
    frame["h3_pair"] = (
        frame["origin_h3"].astype("string").fillna("unknown")
        + "->"
        + frame["destination_h3"].astype("string").fillna("unknown")
    )

    for model_name, predictions in predictions_by_model.items():
        frame[f"{model_name}_prediction_secs"] = predictions
        frame[f"{model_name}_error_secs"] = predictions - frame["actual_eta_secs"]
        frame[f"{model_name}_abs_error_secs"] = frame[f"{model_name}_error_secs"].abs()
        frame[f"{model_name}_ape"] = (
            frame[f"{model_name}_abs_error_secs"] / frame["actual_eta_secs"].clip(lower=1.0)
        )

    frame["baseline_error_secs"] = (
        frame["baseline_prediction_secs"] - frame["actual_eta_secs"]
    )
    frame["baseline_abs_error_secs"] = frame["baseline_error_secs"].abs()
    frame["baseline_ape"] = (
        frame["baseline_abs_error_secs"] / frame["actual_eta_secs"].clip(lower=1.0)
    )
    frame["duration_bucket"] = bucketize_duration(frame["actual_eta_secs"])
    frame["distance_bucket"] = bucketize_distance(frame["baseline_distance_meters"])
    frame["time_bucket"] = bucketize_time_of_day(frame["hour_of_day"])
    frame["rush_hour_bucket"] = np.where(
        frame["is_rush_hour"].astype("string").str.lower().isin(["true", "1"]),
        "rush_hour",
        "non_rush_hour",
    )
    frame["weekend_bucket"] = np.where(
        frame["is_weekend"].astype("string").str.lower().isin(["true", "1"]),
        "weekend",
        "weekday",
    )
    return frame


def summarize_bucket_metrics(
    prediction_frame: pd.DataFrame, model_names: tuple[str, ...] = MODEL_MODES
) -> pd.DataFrame:
    dimensions = {
        "duration": "duration_bucket",
        "distance": "distance_bucket",
        "time_of_day": "time_bucket",
        "rush_hour": "rush_hour_bucket",
        "weekend": "weekend_bucket",
        "origin_h3": "origin_h3",
        "destination_h3": "destination_h3",
    }
    records: list[dict[str, Any]] = []

    for dimension, column in dimensions.items():
        for bucket_value, group in prediction_frame.groupby(column, dropna=False):
            if len(group) == 0:
                continue
            baseline_mae = float(group["baseline_abs_error_secs"].mean())
            row_base = {
                "dimension": dimension,
                "bucket": str(bucket_value),
                "count": int(len(group)),
                "actual_eta_mean": float(group["actual_eta_secs"].mean()),
                "baseline_mae": baseline_mae,
                "baseline_mape": float(group["baseline_ape"].mean()),
            }
            for model_name in model_names:
                model_mae = float(group[f"{model_name}_abs_error_secs"].mean())
                records.append(
                    {
                        **row_base,
                        "model": model_name,
                        "mae": model_mae,
                        "mape": float(group[f"{model_name}_ape"].mean()),
                        "bias": float(group[f"{model_name}_error_secs"].mean()),
                        "p50_abs_error": float(
                            group[f"{model_name}_abs_error_secs"].quantile(0.50)
                        ),
                        "p90_abs_error": float(
                            group[f"{model_name}_abs_error_secs"].quantile(0.90)
                        ),
                        "mae_improvement_vs_baseline": baseline_mae - model_mae,
                    }
                )

    return pd.DataFrame(records).sort_values(
        ["dimension", "model", "count"], ascending=[True, True, False]
    )


def summarize_h3_pair_errors(
    prediction_frame: pd.DataFrame,
    min_count: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for h3_pair, group in prediction_frame.groupby("h3_pair"):
        if len(group) < min_count:
            continue

        baseline_mae = float(group["baseline_abs_error_secs"].mean())
        residual_mae = float(group["residual_abs_error_secs"].mean())
        records.append(
            {
                "h3_pair": h3_pair,
                "count": int(len(group)),
                "origin_h3": str(group["origin_h3"].iloc[0]),
                "destination_h3": str(group["destination_h3"].iloc[0]),
                "actual_eta_mean": float(group["actual_eta_secs"].mean()),
                "baseline_mae": baseline_mae,
                "residual_mae": residual_mae,
                "residual_improvement_vs_baseline": baseline_mae - residual_mae,
                "residual_p90_abs_error": float(
                    group["residual_abs_error_secs"].quantile(0.90)
                ),
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "h3_pair",
                "count",
                "origin_h3",
                "destination_h3",
                "actual_eta_mean",
                "baseline_mae",
                "residual_mae",
                "residual_improvement_vs_baseline",
                "residual_p90_abs_error",
            ]
        )

    return pd.DataFrame(records).sort_values(
        ["residual_mae", "count"], ascending=[False, False]
    )


def make_residual_evaluation(metrics: dict[str, float]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for split_name in ["train", "test"]:
        prefix = f"{split_name}_residual"
        records.append(
            {
                "split": split_name,
                "model": "residual",
                "mae": metrics[f"{prefix}_mae"],
                "mape": metrics[f"{prefix}_mape"],
                "rmse": metrics[f"{prefix}_rmse"],
                "bias": metrics[f"{prefix}_bias"],
                "p50_abs_error": metrics[f"{prefix}_p50_abs_error"],
                "p90_abs_error": metrics[f"{prefix}_p90_abs_error"],
                "underprediction_rate": metrics[f"{prefix}_underprediction_rate"],
                "mae_improvement_vs_baseline": metrics[
                    f"{prefix}_mae_improvement_vs_baseline"
                ],
                "mae_improvement_pct_vs_baseline": metrics[
                    f"{prefix}_mae_improvement_pct_vs_baseline"
                ],
            }
        )
    return pd.DataFrame(records)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv_strategy = resolve_cv_strategy(
        split_strategy=args.split_strategy,
        cv_strategy=args.cv_strategy,
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tags(
            {
                "model_family": "xgboost",
                "pipeline": "eta_baseline",
                "split_strategy": args.split_strategy,
                "cv_strategy": cv_strategy,
            }
        )
        mlflow.log_params(
            {
                "data_path": str(args.data_path),
                "test_size": args.test_size,
                "split_strategy": args.split_strategy,
                "cv_strategy": cv_strategy,
                "seed": args.seed,
                "n_trials": args.n_trials,
                "cv_folds": args.cv_folds,
                "cv_jobs": args.cv_jobs,
                "xgb_jobs": args.xgb_jobs,
                "target": TARGET_COLUMN,
            }
        )

        df = load_dataset(args.data_path)
        mlflow.log_metrics(
            {
                "dataset_rows": int(len(df)),
                "dataset_columns": int(len(df.columns)),
                "target_mean_seconds": float(df[TARGET_COLUMN].mean()),
                "target_median_seconds": float(df[TARGET_COLUMN].median()),
            }
        )

        X_train, X_test, y_train, y_test = make_train_test_split(
            df,
            test_size=args.test_size,
            seed=args.seed,
            split_strategy=args.split_strategy,
        )
        best_params, tuning_results = tune_hyperparameters(
            X_train,
            y_train,
            n_trials=args.n_trials,
            cv_folds=args.cv_folds,
            cv_strategy=cv_strategy,
            cv_jobs=args.cv_jobs,
            xgb_jobs=args.xgb_jobs,
            seed=args.seed,
        )

        tuning_path = args.output_dir / "xgb_tuning_results.csv"
        tuning_results.to_csv(tuning_path, index=False)
        mlflow.log_artifact(str(tuning_path), artifact_path="tuning")
        mlflow.log_params({f"best_{key}": value for key, value in best_params.items()})

        with mlflow.start_span(
            "fit_residual_model",
            attributes={"best_params": best_params, "target": "actual_minus_baseline"},
        ):
            residual_model = build_pipeline(
                best_params,
                seed=args.seed,
                xgb_jobs=args.xgb_jobs,
            )
            residual_model.fit(X_train, make_residual_target(X_train, y_train))

        train_actual = y_train.to_numpy(dtype=float)
        test_actual = y_test.to_numpy(dtype=float)
        train_baseline = X_train["baseline_eta_secs"].to_numpy(dtype=float)
        test_baseline = X_test["baseline_eta_secs"].to_numpy(dtype=float)

        train_residual_metrics, train_residual_predictions = evaluate_model(
            "train", "residual", residual_model, X_train, y_train
        )
        test_residual_metrics, test_residual_predictions = evaluate_model(
            "test", "residual", residual_model, X_test, y_test
        )

        metrics = {
            **regression_metrics(train_actual, train_baseline, "train_baseline"),
            **regression_metrics(test_actual, test_baseline, "test_baseline"),
            **train_residual_metrics,
            **test_residual_metrics,
        }
        # Backward-compatible aliases now point at the residual ETA model.
        metrics.update(
            {
                "train_mae": metrics["train_residual_mae"],
                "train_mape": metrics["train_residual_mape"],
                "test_mae": metrics["test_residual_mae"],
                "test_mape": metrics["test_residual_mape"],
            }
        )
        mlflow.log_metrics(metrics)

        metrics_path = args.output_dir / "xgb_eta_baseline_metrics.json"
        save_json(metrics_path, metrics)
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        residual_evaluation = make_residual_evaluation(metrics)
        residual_evaluation_path = args.output_dir / "xgb_residual_evaluation.csv"
        residual_evaluation.to_csv(residual_evaluation_path, index=False)
        mlflow.log_artifact(str(residual_evaluation_path), artifact_path="metrics")

        test_prediction_frame = make_prediction_frame(
            X_test,
            y_test,
            {
                "residual": test_residual_predictions,
            },
        )
        prediction_path = args.output_dir / "xgb_test_predictions.csv"
        test_prediction_frame.to_csv(prediction_path, index=False)
        mlflow.log_artifact(str(prediction_path), artifact_path="model_diagnostics")

        bucket_metrics = summarize_bucket_metrics(test_prediction_frame)
        bucket_metrics_path = args.output_dir / "xgb_bucket_evaluation.csv"
        bucket_metrics.to_csv(bucket_metrics_path, index=False)
        mlflow.log_artifact(str(bucket_metrics_path), artifact_path="model_diagnostics")

        h3_pair_errors = summarize_h3_pair_errors(
            test_prediction_frame,
            min_count=args.h3_min_count,
        )
        h3_pair_errors_path = args.output_dir / "xgb_h3_pair_error_analysis.csv"
        h3_pair_errors.to_csv(h3_pair_errors_path, index=False)
        mlflow.log_artifact(str(h3_pair_errors_path), artifact_path="model_diagnostics")

        feature_importance = extract_feature_importance(residual_model)
        importance_path = args.output_dir / "xgb_feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path), artifact_path="model_diagnostics")

        residual_importance_path = args.output_dir / "xgb_residual_feature_importance.csv"
        feature_importance.to_csv(residual_importance_path, index=False)
        mlflow.log_artifact(
            str(residual_importance_path), artifact_path="model_diagnostics"
        )

        importance_summary = {
            **feature_importance_summary(feature_importance, "residual"),
        }
        importance_summary_path = args.output_dir / "xgb_feature_importance_summary.json"
        save_json(importance_summary_path, importance_summary)
        mlflow.log_metrics(importance_summary)
        mlflow.log_artifact(
            str(importance_summary_path), artifact_path="model_diagnostics"
        )

        model_path = args.output_dir / "xgb_eta_residual.joblib"
        joblib.dump(residual_model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.sklearn.log_model(
            residual_model,
            name="xgb_eta_residual_pipeline",
            input_example=X_train.head(5),
        )

        print("\nBest hyperparameters")
        print(json.dumps(best_params, indent=2, sort_keys=True))
        print("\nEvaluation")
        for key, value in metrics.items():
            is_ratio_metric = key.endswith(("mape", "_rate")) or "_pct_" in key
            suffix = "%" if is_ratio_metric else " seconds"
            display_value = value * 100.0 if is_ratio_metric else value
            print(f"{key}: {display_value:.2f}{suffix}")
        print("\nFeature importance summary")
        print(json.dumps(importance_summary, indent=2, sort_keys=True))
        print(f"\nSaved residual model artifact: {model_path}")
        print(f"MLflow tracking URI: {args.tracking_uri}")


if __name__ == "__main__":
    main()
