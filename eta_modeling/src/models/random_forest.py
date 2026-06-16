from __future__ import annotations

from typing import Any
from sklearn.ensemble import RandomForestRegressor

def build_rf_regressor(config: dict[str, Any]) -> RandomForestRegressor:
    """Create a Random Forest regressor from config."""
    params = dict(config.get("random_forest", {}))
    # Default parameters if none provided
    if not params:
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1
        }
    model = RandomForestRegressor(**params)
    return model
