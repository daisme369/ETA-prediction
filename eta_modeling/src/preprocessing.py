from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def validate_experiment_dataframe(df: pd.DataFrame, min_samples: int = 30) -> list[str]:
    """Validate target, baseline, timestamps, and fixed-route sample size."""
    issues: list[str] = []
    required = ["actual_eta_secs", "baseline_eta_secs", "timestamp", "residual_secs"]
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns after loading: {missing_cols}")

    if df["actual_eta_secs"].isna().any():
        raise ValueError("actual_eta_secs has missing target values.")
    if (df["actual_eta_secs"] <= 0).any():
        raise ValueError("actual_eta_secs contains non-positive values.")
    if df["baseline_eta_secs"].isna().any():
        raise ValueError("baseline_eta_secs has missing values.")
    if (df["baseline_eta_secs"] < 0).any():
        raise ValueError("baseline_eta_secs contains negative values.")
    if df["timestamp"].isna().any():
        raise ValueError("timestamp contains unparseable values.")
    if len(df) < min_samples:
        raise ValueError(f"Fixed OD pair has too few samples: {len(df)} < {min_samples}.")

    if len(df) < 500:
        issues.append(f"dataset is small for ML training: {len(df)} rows")
    if len(df) < max(100, min_samples * 2):
        issues.append(f"OD pair has low sample count: {len(df)} rows")
    if df["timestamp"].duplicated().any():
        issues.append("timestamp has duplicates")

    corr = df[["actual_eta_secs", "baseline_eta_secs"]].corr().iloc[0, 1]
    if np.isfinite(corr) and abs(corr) >= 0.95:
        issues.append(f"baseline_eta_secs is highly correlated with actual_eta_secs: {corr:.3f}")

    for issue in issues:
        warnings.warn(issue, RuntimeWarning, stacklevel=2)
        LOGGER.warning(issue)
    return issues


def time_based_split(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe chronologically into train/validation/test partitions."""
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must satisfy 0 < train_ratio, val_ratio and sum < 1.")

    ordered = df.sort_values(timestamp_col).reset_index(drop=True)
    n_rows = len(ordered)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))
    if train_end == 0 or val_end <= train_end or val_end >= n_rows:
        raise ValueError(f"Not enough rows for requested split: {n_rows}")

    train_df = ordered.iloc[:train_end].copy()
    val_df = ordered.iloc[train_end:val_end].copy()
    test_df = ordered.iloc[val_end:].copy()
    return train_df, val_df, test_df


def split_date_ranges(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> dict[str, str]:
    ranges: dict[str, str] = {}
    for name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        ranges[f"{name}_start"] = str(frame[timestamp_col].min())
        ranges[f"{name}_end"] = str(frame[timestamp_col].max())
    return ranges
