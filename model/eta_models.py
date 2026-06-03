from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


def _extract_hour(X) -> pd.Series:
    X_df = pd.DataFrame(X).copy()
    if "hour" in X_df.columns:
        return X_df["hour"].astype(int).reset_index(drop=True)
    return X_df.iloc[:, 0].astype(int).reset_index(drop=True)


class HourMedianRegressor(BaseEstimator, RegressorMixin):
    """Predict ETA by the median observed for each hour, with a global fallback."""

    def fit(self, X, y):
        y_series = pd.Series(y).reset_index(drop=True)
        hour = _extract_hour(X)
        self.global_median_ = float(y_series.median())
        self.hour_median_ = (
            pd.DataFrame({"hour": hour, "target": y_series})
            .groupby("hour")["target"]
            .median()
            .to_dict()
        )
        return self

    def predict(self, X):
        hour = _extract_hour(X)
        return hour.map(self.hour_median_).fillna(self.global_median_).to_numpy(dtype=float)


class HourQuantileRegressor(BaseEstimator, RegressorMixin):
    """Predict ETA quantiles by hour, with a global quantile fallback."""

    def __init__(self, quantile: float = 0.5):
        self.quantile = quantile

    def fit(self, X, y):
        y_series = pd.Series(y).reset_index(drop=True)
        hour = _extract_hour(X)
        self.global_quantile_ = float(y_series.quantile(self.quantile))
        self.hour_quantile_ = (
            pd.DataFrame({"hour": hour, "target": y_series})
            .groupby("hour")["target"]
            .quantile(self.quantile)
            .to_dict()
        )
        return self

    def predict(self, X):
        hour = _extract_hour(X)
        return hour.map(self.hour_quantile_).fillna(self.global_quantile_).to_numpy(dtype=float)


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    error = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))
