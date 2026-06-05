from __future__ import annotations

from typing import Any

from xgboost import XGBRegressor


def build_xgb_regressor(config: dict[str, Any]) -> XGBRegressor:
    """Create an XGBoost regressor from config."""
    params = dict(config.get("xgboost", {}))
    early_stopping_rounds = params.pop("early_stopping_rounds", None)
    model = XGBRegressor(**params)
    if early_stopping_rounds is not None:
        model.set_params(early_stopping_rounds=early_stopping_rounds)
    return model
