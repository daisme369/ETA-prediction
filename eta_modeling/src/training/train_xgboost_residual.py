from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd

from ..data_loading import load_config
from ..features import build_tabular_preprocessor
from ..mlflow_utils import configure_mlflow, log_metrics_safe
from ..models.xgboost_residual import build_xgb_residual_regressor
from .common import (
    Timer,
    configure_logging,
    evaluate_all_splits,
    log_run_context,
    parse_args,
    plot_diagnostics,
    prepare_data,
    save_joblib_model,
    save_metrics,
    save_predictions,
)
from .train_xgboost_direct import _feature_importance, _fit_xgb


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    prepared = prepare_data(config)
    configure_mlflow(config, "eta_fixed_trip_xgboost_residual")

    features = prepared["feature_list"]
    train_df = prepared["train_df"]
    val_df = prepared["val_df"]
    test_df = prepared["test_df"]
    preprocessor = build_tabular_preprocessor(prepared["numeric_features"], prepared["categorical_features"])
    x_train = preprocessor.fit_transform(train_df[features])
    x_val = preprocessor.transform(val_df[features])
    x_test = preprocessor.transform(test_df[features])
    y_train = train_df["residual_secs"].to_numpy(dtype=float)
    y_val = val_df["residual_secs"].to_numpy(dtype=float)
    model = build_xgb_residual_regressor(config)

    with mlflow.start_run(run_name="xgb_residual_eta"):
        log_run_context(config, prepared, model_type="xgboost", target_type="residual_secs", extra_params={"xgboost": config.get("xgboost", {})})
        _fit_xgb(model, x_train, y_train, x_val, y_val)
        pred_train_residual = model.predict(x_train)
        pred_val_residual = model.predict(x_val)
        with Timer() as timer:
            pred_test_residual = model.predict(x_test)
        pred_train = np.maximum(train_df["baseline_eta_secs"].to_numpy(dtype=float) + pred_train_residual, 0.0)
        pred_val = np.maximum(val_df["baseline_eta_secs"].to_numpy(dtype=float) + pred_val_residual, 0.0)
        pred_test = np.maximum(test_df["baseline_eta_secs"].to_numpy(dtype=float) + pred_test_residual, 0.0)
        metrics = evaluate_all_splits(train_df, val_df, test_df, pred_train, pred_val, pred_test)
        metrics["inference_time_ms_per_sample"] = timer.elapsed / max(len(test_df), 1) * 1000.0
        metrics["residual_train_mean"] = float(np.mean(y_train))
        metrics["residual_prediction_test_mean"] = float(np.mean(pred_test_residual))
        log_metrics_safe(metrics)

        paths = prepared["paths"]
        save_predictions(paths, "xgb_residual_eta", train_df, val_df, test_df, pred_train, pred_val, pred_test)
        save_metrics(paths, "xgb_residual_eta", metrics)
        plot_diagnostics(paths, "xgb_residual_eta", test_df, pred_test)
        save_joblib_model(paths, "xgb_residual_eta.joblib", {"preprocessor": preprocessor, "model": model, "features": features})

        fi = _feature_importance(preprocessor, model)
        fi_path = paths["metrics_dir"] / "xgb_residual_feature_importance.csv"
        fi.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path))

        residual_path = paths["plots_dir"] / "xgb_residual_target_vs_prediction.csv"
        pd.DataFrame({"target_residual": test_df["residual_secs"], "pred_residual": pred_test_residual}).to_csv(residual_path, index=False)
        mlflow.log_artifact(str(residual_path))


if __name__ == "__main__":
    main()
