from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd

from ..data_loading import load_config
from ..features import build_tabular_preprocessor
from ..mlflow_utils import configure_mlflow, log_metrics_safe
from ..models.xgboost_direct import build_xgb_regressor
from ..models.random_forest import build_rf_regressor
from .common import (
    Timer,
    configure_logging,
    evaluate_all_splits,
    log_run_context,
    parse_args,
    plot_diagnostics,
    prepare_hour_binned_data,
    save_joblib_model,
    save_metrics,
    save_predictions,
)

def _fit_model(model, x_train, y_train, x_val, y_val, model_name):
    if model_name == "xgboost":
        try:
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_train, y_train), (x_val, y_val)],
                verbose=False,
            )
        except TypeError:
            model.fit(x_train, y_train)
    else:
        model.fit(x_train, y_train)


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    prepared = prepare_hour_binned_data(config)
    
    configure_mlflow(config, "eta_formula_comparison")

    features = prepared["feature_list"]
    train_df = prepared["train_df"]
    val_df = prepared["val_df"]
    test_df = prepared["test_df"]
    preprocessor = build_tabular_preprocessor(prepared["numeric_features"], prepared["categorical_features"])
    
    x_train = preprocessor.fit_transform(train_df[features])
    x_val = preprocessor.transform(val_df[features])
    x_test = preprocessor.transform(test_df[features])
    
    actual_train = train_df["actual_eta_secs"].to_numpy(dtype=float)
    actual_val = val_df["actual_eta_secs"].to_numpy(dtype=float)
    actual_test = test_df["actual_eta_secs"].to_numpy(dtype=float)
    
    base_train = train_df["baseline_eta_secs"].to_numpy(dtype=float)
    base_val = val_df["baseline_eta_secs"].to_numpy(dtype=float)
    base_test = test_df["baseline_eta_secs"].to_numpy(dtype=float)
    
    safe_base_train = np.maximum(base_train, 1.0)
    safe_base_val = np.maximum(base_val, 1.0)
    safe_base_test = np.maximum(base_test, 1.0)

    formulas = ["additive", "ratio", "log_ratio"]
    models = ["xgboost", "random_forest"]

    results = []

    for model_type in models:
        for formula in formulas:
            model_name_full = f"{model_type}_{formula}"
            print(f"Training {model_name_full}...")
            
            # Prepare targets based on formula
            if formula == "additive":
                y_train = actual_train - base_train
                y_val = actual_val - base_val
            elif formula == "ratio":
                y_train = actual_train / safe_base_train
                y_val = actual_val / safe_base_val
            elif formula == "log_ratio":
                y_train = np.log(actual_train / safe_base_train)
                y_val = np.log(actual_val / safe_base_val)
                
            if model_type == "xgboost":
                model = build_xgb_regressor(config)
            else:
                model = build_rf_regressor(config)

            with mlflow.start_run(run_name=model_name_full):
                log_run_context(
                    config,
                    prepared,
                    model_type=model_type,
                    target_type=formula,
                    extra_params={"formula": formula},
                )
                
                with Timer() as timer_fit:
                    _fit_model(model, x_train, y_train, x_val, y_val, model_type)
                
                pred_train_target = model.predict(x_train)
                pred_val_target = model.predict(x_val)
                with Timer() as timer:
                    pred_test_target = model.predict(x_test)
                    
                # Convert prediction back to ETA
                if formula == "additive":
                    pred_train = np.maximum(base_train + pred_train_target, 0.0)
                    pred_val = np.maximum(base_val + pred_val_target, 0.0)
                    pred_test = np.maximum(base_test + pred_test_target, 0.0)
                elif formula == "ratio":
                    pred_train = np.maximum(base_train * pred_train_target, 0.0)
                    pred_val = np.maximum(base_val * pred_val_target, 0.0)
                    pred_test = np.maximum(base_test * pred_test_target, 0.0)
                elif formula == "log_ratio":
                    pred_train = np.maximum(base_train * np.exp(pred_train_target), 0.0)
                    pred_val = np.maximum(base_val * np.exp(pred_val_target), 0.0)
                    pred_test = np.maximum(base_test * np.exp(pred_test_target), 0.0)
                
                metrics = evaluate_all_splits(train_df, val_df, test_df, pred_train, pred_val, pred_test)
                metrics["inference_time_ms_per_sample"] = timer.elapsed / max(len(test_df), 1) * 1000.0
                metrics["fit_time_s"] = timer_fit.elapsed
                
                log_metrics_safe(metrics)

                paths = prepared["paths"]
                save_predictions(paths, model_name_full, train_df, val_df, test_df, pred_train, pred_val, pred_test)
                save_metrics(paths, model_name_full, metrics)
                plot_diagnostics(paths, model_name_full, test_df, pred_test)
                save_joblib_model(paths, f"{model_name_full}.joblib", {"preprocessor": preprocessor, "model": model, "features": features})
                
                results.append({
                    "model": model_type,
                    "formula": formula,
                    "test_mae": metrics.get("test_mae", 0),
                    "test_mape": metrics.get("test_mape", 0),
                    "test_p95": metrics.get("test_p95", 0),
                    "test_r2": metrics.get("test_r2", 0)
                })

    results_df = pd.DataFrame(results)
    print("\nComparison Results:")
    print(results_df.to_string(index=False))
    
    comp_path = paths["metrics_dir"] / "formula_comparison_summary.csv"
    results_df.to_csv(comp_path, index=False)
    mlflow.log_artifact(str(comp_path))

if __name__ == "__main__":
    main()
