from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import mlflow
import numpy as np
import pandas as pd

from ..data_loading import load_config
from ..features import build_tabular_preprocessor
from ..mlflow_utils import configure_mlflow, log_metrics_safe
from ..models.xgboost_direct import build_xgb_regressor
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
    configure_mlflow(config, "eta_fixed_trip_xgboost_log_ratio")

    features = prepared["feature_list"]
    train_df = prepared["train_df"]
    val_df = prepared["val_df"]
    test_df = prepared["test_df"]
    preprocessor = build_tabular_preprocessor(prepared["numeric_features"], prepared["categorical_features"])
    x_train = preprocessor.fit_transform(train_df[features])
    x_val = preprocessor.transform(val_df[features])
    x_test = preprocessor.transform(test_df[features])
    
    # Calculate log ratio: ln(actual_eta / baseline_eta)
    y_train = np.log(train_df["actual_eta_secs"].to_numpy(dtype=float) / np.maximum(train_df["baseline_eta_secs"].to_numpy(dtype=float), 1.0))
    y_val = np.log(val_df["actual_eta_secs"].to_numpy(dtype=float) / np.maximum(val_df["baseline_eta_secs"].to_numpy(dtype=float), 1.0))
    
    model = build_xgb_regressor(config)

    with mlflow.start_run(run_name="xgb_log_ratio_eta"):
        log_run_context(config, prepared, model_type="xgboost", target_type="log_ratio", extra_params={"xgboost": config.get("xgboost", {})})
        _fit_xgb(model, x_train, y_train, x_val, y_val)
        
        pred_train_log_ratio = model.predict(x_train)
        pred_val_log_ratio = model.predict(x_val)
        with Timer() as timer:
            pred_test_log_ratio = model.predict(x_test)
            
        # Convert log ratio back to ETA: pred_eta = baseline * exp(log_ratio)
        pred_train = np.maximum(train_df["baseline_eta_secs"].to_numpy(dtype=float) * np.exp(pred_train_log_ratio), 0.0)
        pred_val = np.maximum(val_df["baseline_eta_secs"].to_numpy(dtype=float) * np.exp(pred_val_log_ratio), 0.0)
        pred_test = np.maximum(test_df["baseline_eta_secs"].to_numpy(dtype=float) * np.exp(pred_test_log_ratio), 0.0)
        
        metrics = evaluate_all_splits(train_df, val_df, test_df, pred_train, pred_val, pred_test)
        metrics["inference_time_ms_per_sample"] = timer.elapsed / max(len(test_df), 1) * 1000.0
        metrics["log_ratio_train_mean"] = float(np.mean(y_train))
        metrics["log_ratio_prediction_test_mean"] = float(np.mean(pred_test_log_ratio))
        log_metrics_safe(metrics)

        paths = prepared["paths"]
        save_predictions(paths, "xgb_log_ratio_eta", train_df, val_df, test_df, pred_train, pred_val, pred_test)
        save_metrics(paths, "xgb_log_ratio_eta", metrics)
        plot_diagnostics(paths, "xgb_log_ratio_eta", test_df, pred_test)
        save_joblib_model(paths, "xgb_log_ratio_eta.joblib", {"preprocessor": preprocessor, "model": model, "features": features})

        fi = _feature_importance(preprocessor, model)
        fi_path = paths["metrics_dir"] / "xgb_log_ratio_feature_importance.csv"
        fi.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path))

        log_ratio_path = paths["plots_dir"] / "xgb_log_ratio_target_vs_prediction.csv"
        pd.DataFrame({
            "target_log_ratio": np.log(test_df["actual_eta_secs"].to_numpy(dtype=float) / np.maximum(test_df["baseline_eta_secs"].to_numpy(dtype=float), 1.0)), 
            "pred_log_ratio": pred_test_log_ratio
        }).to_csv(log_ratio_path, index=False)
        mlflow.log_artifact(str(log_ratio_path))


if __name__ == "__main__":
    main()
