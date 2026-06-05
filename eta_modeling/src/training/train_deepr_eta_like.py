from __future__ import annotations

import matplotlib.pyplot as plt
import mlflow
import numpy as np
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
    prepare_data,
    save_joblib_model,
    save_metrics,
    save_predictions,
)


DEEPR_CATEGORICAL_TIME_FEATURES = ["hour", "weekday", "month", "is_weekend", "is_rush_hour"]
DEEPR_CATEGORICAL_ROUTE_FEATURES = ["stationId", "destination_stationId", "od_pair"]
DEEPR_CYCLIC_TIME_FEATURES = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos"]
DEEPR_WEATHER_FEATURES = ["origin_rain", "destination_rain", "rain"]


def _present_columns(df, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def _build_deepr_feature_groups(train_df, cfg: dict) -> dict[str, object]:
    """Build DeeprETA-like embedding feature groups without target leakage."""
    if cfg.get("use_hour_bin", False) and "hour_bin" in train_df.columns:
        categorical_time_candidates = ["hour_bin", "weekday", "month", "is_weekend", "is_rush_hour"]
        cyclic_time_candidates = [feature for feature in DEEPR_CYCLIC_TIME_FEATURES if feature not in {"hour_sin", "hour_cos"}]
    else:
        categorical_time_candidates = DEEPR_CATEGORICAL_TIME_FEATURES
        cyclic_time_candidates = DEEPR_CYCLIC_TIME_FEATURES

    categorical_time_features = _present_columns(train_df, categorical_time_candidates)
    categorical_route_features = _present_columns(train_df, DEEPR_CATEGORICAL_ROUTE_FEATURES)
    cyclic_time_features = _present_columns(train_df, cyclic_time_candidates)
    weather_features = _present_columns(train_df, DEEPR_WEATHER_FEATURES)

    bucket_features = {
        "baseline_eta_secs": int(cfg.get("baseline_eta_buckets", 128)),
    }
    if "haversine_distance_meters" in train_df.columns:
        bucket_features["haversine_distance_meters"] = int(cfg.get("distance_buckets", 64))
    for feature in cyclic_time_features:
        bucket_features[feature] = int(cfg.get("cyclic_time_buckets", 32))
    for feature in weather_features:
        feature_key = f"{feature}_buckets"
        bucket_features[feature] = int(cfg.get(feature_key, cfg.get("rain_buckets", 64)))

    blocked_targets = {"actual_eta_secs", "residual_secs", "delta_time"}
    leaked = blocked_targets.intersection(set(categorical_time_features + categorical_route_features + list(bucket_features)))
    if leaked:
        raise ValueError(f"Target leakage in DeeprETA-like features: {sorted(leaked)}")

    categorical_embedding_features = categorical_time_features + categorical_route_features
    continuous_bucket_embedding_features = list(bucket_features)
    time_features = categorical_time_features + cyclic_time_features

    return {
        "categorical_embedding_features": categorical_embedding_features,
        "continuous_bucket_embedding_features": continuous_bucket_embedding_features,
        "time_features": time_features,
        "weather_features": weather_features,
        "bucket_features": bucket_features,
    }


def _plot_loss(paths, model_name: str, train_loss: list[float], val_mae: list[float]):
    path = paths["plots_dir"] / f"{model_name}_loss_curve.png"
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
    prepared = prepare_data(config)
    configure_mlflow(config, "eta_fixed_trip_deepr_eta_like")

    train_df = prepared["train_df"]
    val_df = prepared["val_df"]
    test_df = prepared["test_df"]
    cfg = config.get("deepr_eta_like", {})
    feature_groups = _build_deepr_feature_groups(train_df, cfg)
    categorical_features = feature_groups["categorical_embedding_features"]
    bucket_features = feature_groups["bucket_features"]
    encoder = DeepFeatureEncoder(
        categorical_features=categorical_features,
        bucket_features=bucket_features,
        calibration_feature=cfg.get("calibration_feature") if cfg.get("use_calibration", True) else None,
    ).fit(train_df)

    with mlflow.start_run(run_name="deepr_eta_like_residual"):
        log_run_context(
            config,
            prepared,
            model_type="deepr_eta_like",
            target_type="residual_secs_final_eta_loss",
            extra_params={
                "deepr_eta_like": cfg,
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
        save_predictions(paths, "deepr_eta_like", train_df, val_df, test_df, pred_train, pred_val, pred_test)
        save_metrics(paths, "deepr_eta_like", metrics)
        plot_diagnostics(paths, "deepr_eta_like", test_df, pred_test)
        _plot_loss(paths, "deepr_eta_like", result.train_loss, result.val_mae)
        model_path = paths["models_dir"] / "deepr_eta_like.pt"
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
        save_joblib_model(paths, "deepr_eta_like_encoder.joblib", encoder)
        save_joblib_model(paths, "deepr_eta_like_bucketizers.joblib", encoder.bucketizer_artifact())


if __name__ == "__main__":
    main()
