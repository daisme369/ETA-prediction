from __future__ import annotations

import joblib
import mlflow
import numpy as np

from ..data_loading import load_config
from ..mlflow_utils import configure_mlflow, log_metrics_safe
from .common import (
    configure_logging,
    evaluate_all_splits,
    log_run_context,
    parse_args,
    plot_diagnostics,
    prepare_hour_binned_data,
    save_metrics,
    save_predictions,
)


MODEL_NAME = "hour_bin_median_eta"


def _predict_with_bin_medians(frame, medians, global_median: float) -> np.ndarray:
    residual = frame["hour_bin"].map(medians).fillna(global_median).to_numpy(dtype=float)
    baseline = frame["baseline_eta_secs"].to_numpy(dtype=float)
    return np.maximum(baseline + residual, 0.0)


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    prepared = prepare_hour_binned_data(config)
    configure_mlflow(config, "eta_fixed_trip_hour_bin_median")

    train_df = prepared["train_df"]
    val_df = prepared["val_df"]
    test_df = prepared["test_df"]
    medians = train_df.groupby("hour_bin")["residual_secs"].median()
    counts = train_df["hour_bin"].value_counts().to_dict()
    global_median = float(train_df["residual_secs"].median())

    with mlflow.start_run(run_name=MODEL_NAME):
        log_run_context(
            config,
            prepared,
            model_type="hour_bin_median",
            target_type="median_residual_secs_by_hour_bin",
            extra_params={
                "hour_binning": prepared["hour_binning"],
                "hour_bin_counts_train": counts,
                "global_median_residual_secs": global_median,
            },
        )
        pred_train = _predict_with_bin_medians(train_df, medians, global_median)
        pred_val = _predict_with_bin_medians(val_df, medians, global_median)
        pred_test = _predict_with_bin_medians(test_df, medians, global_median)
        metrics = evaluate_all_splits(train_df, val_df, test_df, pred_train, pred_val, pred_test)
        metrics["inference_time_ms_per_sample"] = 0.0
        log_metrics_safe(metrics)

        paths = prepared["paths"]
        save_predictions(paths, MODEL_NAME, train_df, val_df, test_df, pred_train, pred_val, pred_test)
        save_metrics(paths, MODEL_NAME, metrics)
        plot_diagnostics(paths, MODEL_NAME, test_df, pred_test)
        model_path = paths["models_dir"] / f"{MODEL_NAME}.joblib"
        joblib.dump(
            {
                "model_type": "hour_bin_median",
                "hour_bin_median_residual_secs": medians.to_dict(),
                "global_median_residual_secs": global_median,
                "hour_binning": prepared["hour_binning"],
            },
            model_path,
        )
        mlflow.log_artifact(str(model_path))


if __name__ == "__main__":
    main()
