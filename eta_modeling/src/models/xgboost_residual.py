from __future__ import annotations

from typing import Any

from .xgboost_direct import build_xgb_regressor


def build_xgb_residual_regressor(config: dict[str, Any]):
    """Create an XGBoost regressor for residual learning."""
    return build_xgb_regressor(config)
