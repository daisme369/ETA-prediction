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
from sklearn.model_selection import KFold, ParameterSampler, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "mock_eta_trips.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "model" / "artifacts"
DEFAULT_TRACKING_URI = (ROOT_DIR / "mlruns").as_uri()
TARGET_COLUMN = "actual_eta_secs"
LEAKAGE_COLUMNS = {"actual_eta_secs", "residual_secs"}

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
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=16)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--cv-jobs", type=int, default=1)
    return parser.parse_args()


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
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    with mlflow.start_span(
        "train_test_split", attributes={"test_size": test_size, "seed": seed}
    ):
        features = df.drop(columns=[col for col in LEAKAGE_COLUMNS if col in df.columns])
        target = df[TARGET_COLUMN].astype(float)
        return train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )


def build_pipeline(params: dict[str, Any], seed: int) -> Pipeline:
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
        n_jobs=1,
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


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int,
    cv_folds: int,
    cv_jobs: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    with mlflow.start_span(
        "hyperparameter_tuning",
        attributes={"n_trials": n_trials, "cv_folds": cv_folds},
    ):
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        scoring = {
            "mae": "neg_mean_absolute_error",
            "mape": "neg_mean_absolute_percentage_error",
        }
        records: list[dict[str, Any]] = []

        for trial_index, params in enumerate(sample_hyperparameters(n_trials, seed), start=1):
            with mlflow.start_run(run_name=f"trial_{trial_index:02d}", nested=True):
                mlflow.log_params(params)
                pipeline = build_pipeline(params, seed=seed)
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


def evaluate_model(
    name: str, model: Pipeline, X: pd.DataFrame, y: pd.Series
) -> dict[str, float]:
    with mlflow.start_span("evaluate_model", attributes={"split": name}):
        predictions = np.maximum(model.predict(X), 1.0)
        actual = y.to_numpy(dtype=float)
        baseline = X["baseline_eta_secs"].to_numpy(dtype=float)
        return {
            f"{name}_mae": float(mean_absolute_error(actual, predictions)),
            f"{name}_mape": float(mean_absolute_percentage_error(actual, predictions)),
            f"{name}_baseline_mae": float(mean_absolute_error(actual, baseline)),
            f"{name}_baseline_mape": float(
                mean_absolute_percentage_error(actual, baseline)
            ),
        }


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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name="xgboost_eta_baseline"):
        mlflow.log_params(
            {
                "data_path": str(args.data_path),
                "test_size": args.test_size,
                "seed": args.seed,
                "n_trials": args.n_trials,
                "cv_folds": args.cv_folds,
                "cv_jobs": args.cv_jobs,
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
            df, test_size=args.test_size, seed=args.seed
        )
        best_params, tuning_results = tune_hyperparameters(
            X_train,
            y_train,
            n_trials=args.n_trials,
            cv_folds=args.cv_folds,
            cv_jobs=args.cv_jobs,
            seed=args.seed,
        )

        tuning_path = args.output_dir / "xgb_tuning_results.csv"
        tuning_results.to_csv(tuning_path, index=False)
        mlflow.log_artifact(str(tuning_path), artifact_path="tuning")
        mlflow.log_params({f"best_{key}": value for key, value in best_params.items()})

        with mlflow.start_span("fit_final_model", attributes={"best_params": best_params}):
            final_model = build_pipeline(best_params, seed=args.seed)
            final_model.fit(X_train, y_train)

        metrics = {
            **evaluate_model("train", final_model, X_train, y_train),
            **evaluate_model("test", final_model, X_test, y_test),
        }
        mlflow.log_metrics(metrics)

        metrics_path = args.output_dir / "xgb_eta_baseline_metrics.json"
        save_json(metrics_path, metrics)
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        feature_importance = extract_feature_importance(final_model)
        importance_path = args.output_dir / "xgb_feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path), artifact_path="model_diagnostics")

        model_path = args.output_dir / "xgb_eta_baseline.joblib"
        joblib.dump(final_model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.sklearn.log_model(
            final_model,
            name="xgb_eta_pipeline",
            input_example=X_train.head(5),
        )

        print("\nBest hyperparameters")
        print(json.dumps(best_params, indent=2, sort_keys=True))
        print("\nEvaluation")
        for key, value in metrics.items():
            suffix = "%" if key.endswith("mape") else " seconds"
            display_value = value * 100.0 if key.endswith("mape") else value
            print(f"{key}: {display_value:.2f}{suffix}")
        print(f"\nSaved model artifact: {model_path}")
        print(f"MLflow tracking URI: {args.tracking_uri}")


if __name__ == "__main__":
    main()
