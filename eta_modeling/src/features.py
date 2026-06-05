from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def haversine_distance_meters(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    """Vectorized haversine distance in meters."""
    radius_m = 6_371_000.0
    phi1 = np.radians(lat1.astype(float))
    phi2 = np.radians(lat2.astype(float))
    d_phi = np.radians(lat2.astype(float) - lat1.astype(float))
    d_lambda = np.radians(lon2.astype(float) - lon1.astype(float))
    a = np.sin(d_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2.0) ** 2
    return pd.Series(radius_m * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)), index=lat1.index)


def add_engineered_features(df: pd.DataFrame, use_cyclic: bool = True, use_geohash: bool = False) -> pd.DataFrame:
    """Create reusable ETA features without target leakage."""
    out = df.copy()
    out["hour_bin"] = create_hour_bins(out["hour"])
    out["stationId"] = out["stationId"].astype(str)
    out["destination_stationId"] = out["destination_stationId"].astype(str)
    out["od_pair"] = out["stationId"] + "_" + out["destination_stationId"]
    out["haversine_distance_meters"] = haversine_distance_meters(
        out["lat"],
        out["lng"],
        out["destination_lat"],
        out["destination_lng"],
    )

    bool_cols = ["is_weekend", "is_rush_hour"]
    for col in bool_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    if use_cyclic:
        out["hour_sin"] = np.sin(2 * math.pi * out["hour"].astype(float) / 24.0)
        out["hour_cos"] = np.cos(2 * math.pi * out["hour"].astype(float) / 24.0)
        out["weekday_sin"] = np.sin(2 * math.pi * out["weekday"].astype(float) / 7.0)
        out["weekday_cos"] = np.cos(2 * math.pi * out["weekday"].astype(float) / 7.0)
        out["month_sin"] = np.sin(2 * math.pi * out["month"].astype(float) / 12.0)
        out["month_cos"] = np.cos(2 * math.pi * out["month"].astype(float) / 12.0)

    if use_geohash:
        try:
            import geohash  # type: ignore

            out["origin_geohash"] = [
                geohash.encode(float(lat), float(lng), precision=6)
                for lat, lng in zip(out["lat"], out["lng"])
            ]
            out["destination_geohash"] = [
                geohash.encode(float(lat), float(lng), precision=6)
                for lat, lng in zip(out["destination_lat"], out["destination_lng"])
            ]
            out["od_geohash_pair"] = out["origin_geohash"] + "_" + out["destination_geohash"]
        except Exception:
            use_geohash = False

    return out


def create_hour_bins(hour: pd.Series) -> pd.Series:
    """Map sparse hourly ETA samples into operational service-period bins."""
    hour_num = pd.to_numeric(hour, errors="coerce")
    conditions = [
        hour_num.between(5, 6, inclusive="both"),
        hour_num.between(7, 9, inclusive="both"),
        hour_num.between(10, 14, inclusive="both"),
        hour_num.between(15, 18, inclusive="both"),
        hour_num.between(19, 21, inclusive="both"),
    ]
    labels = [
        "early_morning",
        "morning_peak",
        "off_peak_midday",
        "afternoon_evening_peak",
        "late_evening_low_service",
    ]
    return pd.Series(np.select(conditions, labels, default="other"), index=hour.index, dtype="object")


def configured_feature_columns(df: pd.DataFrame, config: dict) -> tuple[list[str], list[str]]:
    feature_cfg = config.get("features", {})
    numeric = [col for col in feature_cfg.get("numeric", []) if col in df.columns]
    categorical = [col for col in feature_cfg.get("categorical", []) if col in df.columns]
    if not numeric and not categorical:
        raise ValueError("No configured features are present in dataframe.")
    return numeric, categorical


def build_tabular_preprocessor(numeric_features: Iterable[str], categorical_features: Iterable[str]) -> ColumnTransformer:
    """Build a leakage-safe sklearn preprocessor fit on train data only."""
    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_frame(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    cols = numeric + categorical
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[cols].copy()
