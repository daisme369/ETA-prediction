from __future__ import annotations

import matplotlib.pyplot as plt
import mlflow
import torch

from ..data_loading import load_config
from ..mlflow_utils import configure_mlflow, log_metrics_safe
from ..models.deepr_eta_like import DeepFeatureEncoder, predict_deepr_eta, train_deepr_eta_like_model
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
from .train_deepr_eta_like import _build_deepr_feature_groups


MODEL_NAME = "hour_bin_deepr_eta_like"


def _plot_loss(paths, train_loss: list[float], val_mae: list[float]) -> None:
    path = paths["plots_dir"] / f"{MODEL_NAME}_loss_curve.png"
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="train_loss")
    plt.plot(val_mae, label="val_mae")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    mlflow.log_artifact(str(path))


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    prepared = prepare_hour_binned_data(config)
    configure_mlflow(config, "eta_fixed_trip_hour_bin_deepr_eta_like")

    train_df = prepared["train_df"]
    val_df = prepared["val_df"]
    test_df = prepared["test_df"]
    cfg = dict(config.get("deepr_eta_like", {}))
    cfg["use_hour_bin"] = True
    if cfg.get("calibration_feature") == "hour":
        cfg["calibration_feature"] = "hour_bin"
    feature_groups = _build_deepr_feature_groups(train_df, cfg)
    categorical_features = feature_groups["categorical_embedding_features"]
    bucket_features = feature_groups["bucket_features"]
    encoder = DeepFeatureEncoder(
        categorical_features=categorical_features,
        bucket_features=bucket_features,
        calibration_feature=cfg.get("calibration_feature") if cfg.get("use_calibration", True) else None,
    ).fit(train_df)

    with mlflow.start_run(run_name=MODEL_NAME):
        log_run_context(
            config,
            prepared,
            model_type="hour_bin_deepr_eta_like",
            target_type="residual_secs_final_eta_loss",
            extra_params={
                "deepr_eta_like": cfg,
                "hour_binning": prepared["hour_binning"],
                "embedding_features": categorical_features + list(bucket_features),
                "categorical_embedding_features": feature_groups["categorical_embedding_features"],
                "continuous_bucket_embedding_features": feature_groups["continuous_bucket_embedding_features"],
                "time_features": feature_groups["time_features"],
                "weather_features": feature_groups["weather_features"],
            },
        )
        result = train_deepr_eta_like_model(
            encoder,
            train_df,
            val_df,
            cfg,
            random_seed=int(config.get("project", {}).get("random_seed", 42)),
        )
        pred_train = predict_deepr_eta(result.model, encoder, train_df)
        pred_val = predict_deepr_eta(result.model, encoder, val_df)
        with Timer() as timer:
            pred_test = predict_deepr_eta(result.model, encoder, test_df)
        metrics = evaluate_all_splits(train_df, val_df, test_df, pred_train, pred_val, pred_test)
        metrics["best_epoch"] = float(result.best_epoch)
        metrics["inference_time_ms_per_sample"] = timer.elapsed / max(len(test_df), 1) * 1000.0
        log_metrics_safe(metrics)

        paths = prepared["paths"]
        save_predictions(paths, MODEL_NAME, train_df, val_df, test_df, pred_train, pred_val, pred_test)
        save_metrics(paths, MODEL_NAME, metrics)
        plot_diagnostics(paths, MODEL_NAME, test_df, pred_test)
        _plot_loss(paths, result.train_loss, result.val_mae)
        model_path = paths["models_dir"] / f"{MODEL_NAME}.pt"
        torch.save(
            {
                "state_dict": result.model.state_dict(),
                "cardinalities": encoder.cardinalities(),
                "config": cfg,
                "feature_groups": feature_groups,
            },
            model_path,
        )
        mlflow.log_artifact(str(model_path))
        save_joblib_model(paths, f"{MODEL_NAME}_encoder.joblib", encoder)
        save_joblib_model(paths, f"{MODEL_NAME}_bucketizers.joblib", encoder.bucketizer_artifact())


if __name__ == "__main__":
    main()
