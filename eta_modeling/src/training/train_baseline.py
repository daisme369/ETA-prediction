from __future__ import annotations

import mlflow

from ..data_loading import load_config
from ..metrics import eta_metrics_with_baseline
from ..mlflow_utils import configure_mlflow, log_metrics_safe
from .common import (
    configure_logging,
    evaluate_all_splits,
    log_run_context,
    parse_args,
    plot_diagnostics,
    prepare_data,
    save_metrics,
    save_predictions,
)


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    prepared = prepare_data(config)
    configure_mlflow(config, "eta_fixed_trip_baseline")

    train_df = prepared["train_df"]
    val_df = prepared["val_df"]
    test_df = prepared["test_df"]
    pred_train = train_df["baseline_eta_secs"].to_numpy(dtype=float)
    pred_val = val_df["baseline_eta_secs"].to_numpy(dtype=float)
    pred_test = test_df["baseline_eta_secs"].to_numpy(dtype=float)

    with mlflow.start_run(run_name="vietmap_baseline"):
        log_run_context(config, prepared, model_type="vietmap_baseline", target_type="baseline_eta")
        metrics = evaluate_all_splits(train_df, val_df, test_df, pred_train, pred_val, pred_test)
        baseline_only = eta_metrics_with_baseline(
            test_df["actual_eta_secs"].to_numpy(dtype=float),
            pred_test,
            pred_test,
            "test",
        )
        metrics.update({f"baseline_only_{key}": value for key, value in baseline_only.items()})
        log_metrics_safe(metrics)
        save_predictions(prepared["paths"], "vietmap_baseline", train_df, val_df, test_df, pred_train, pred_val, pred_test)
        save_metrics(prepared["paths"], "vietmap_baseline", metrics)
        plot_diagnostics(prepared["paths"], "vietmap_baseline", test_df, pred_test)


if __name__ == "__main__":
    main()
