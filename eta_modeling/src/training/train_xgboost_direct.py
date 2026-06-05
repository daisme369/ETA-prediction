from __future__ import annotations

import pandas as pd
import mlflow
import numpy as np

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


def _fit_xgb(model, x_train, y_train, x_val, y_val) -> None:
    try:
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    except TypeError:
        model.set_params(early_stopping_rounds=None)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)


def _feature_importance(preprocessor, model) -> pd.DataFrame:
    names = preprocessor.get_feature_names_out()
    importances = getattr(model, "feature_importances_", np.zeros(len(names)))
    return pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False)


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    prepared = prepare_data(config)
    configure_mlflow(config, "eta_fixed_trip_xgboost_direct")

    features = prepared["feature_list"]
    train_df = prepared["train_df"]
    val_df = prepared["val_df"]
    test_df = prepared["test_df"]
    preprocessor = build_tabular_preprocessor(prepared["numeric_features"], prepared["categorical_features"])
    x_train = preprocessor.fit_transform(train_df[features])
    x_val = preprocessor.transform(val_df[features])
    x_test = preprocessor.transform(test_df[features])
    y_train = train_df["actual_eta_secs"].to_numpy(dtype=float)
    y_val = val_df["actual_eta_secs"].to_numpy(dtype=float)
    model = build_xgb_regressor(config)

    with mlflow.start_run(run_name="xgb_direct_eta"):
        log_run_context(config, prepared, model_type="xgboost", target_type="actual_eta_secs", extra_params={"xgboost": config.get("xgboost", {})})
        _fit_xgb(model, x_train, y_train, x_val, y_val)
        with Timer() as timer:
            pred_test = np.maximum(model.predict(x_test), 0.0)
        pred_train = np.maximum(model.predict(x_train), 0.0)
        pred_val = np.maximum(model.predict(x_val), 0.0)
        metrics = evaluate_all_splits(train_df, val_df, test_df, pred_train, pred_val, pred_test)
        metrics["inference_time_ms_per_sample"] = timer.elapsed / max(len(test_df), 1) * 1000.0
        log_metrics_safe(metrics)

        paths = prepared["paths"]
        save_predictions(paths, "xgb_direct_eta", train_df, val_df, test_df, pred_train, pred_val, pred_test)
        save_metrics(paths, "xgb_direct_eta", metrics)
        plot_diagnostics(paths, "xgb_direct_eta", test_df, pred_test)
        save_joblib_model(paths, "xgb_direct_eta.joblib", {"preprocessor": preprocessor, "model": model, "features": features})

        fi = _feature_importance(preprocessor, model)
        fi_path = paths["metrics_dir"] / "xgb_direct_feature_importance.csv"
        fi.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path))


if __name__ == "__main__":
    main()
