from __future__ import annotations

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch

from ..data_loading import load_config
from ..features import build_tabular_preprocessor
from ..mlflow_utils import configure_mlflow, log_metrics_safe
from ..models.mlp_residual import predict_residual, train_mlp_residual_model
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
    configure_mlflow(config, "eta_fixed_trip_mlp_residual")

    features = prepared["feature_list"]
    train_df = prepared["train_df"]
    val_df = prepared["val_df"]
    test_df = prepared["test_df"]
    preprocessor = build_tabular_preprocessor(prepared["numeric_features"], prepared["categorical_features"])
    x_train = np.asarray(preprocessor.fit_transform(train_df[features]), dtype=np.float32)
    x_val = np.asarray(preprocessor.transform(val_df[features]), dtype=np.float32)
    x_test = np.asarray(preprocessor.transform(test_df[features]), dtype=np.float32)
    cfg = config.get("mlp_residual", {})

    with mlflow.start_run(run_name="mlp_residual_eta"):
        log_run_context(config, prepared, model_type="pytorch_mlp", target_type="residual_secs", extra_params={"mlp_residual": cfg})
        result = train_mlp_residual_model(
            x_train,
            train_df["residual_secs"].to_numpy(dtype=float),
            train_df["baseline_eta_secs"].to_numpy(dtype=float),
            train_df["actual_eta_secs"].to_numpy(dtype=float),
            x_val,
            val_df["residual_secs"].to_numpy(dtype=float),
            val_df["baseline_eta_secs"].to_numpy(dtype=float),
            val_df["actual_eta_secs"].to_numpy(dtype=float),
            hidden_sizes=[int(x) for x in cfg.get("hidden_sizes", [128, 64, 32])],
            dropout=float(cfg.get("dropout", 0.15)),
            learning_rate=float(cfg.get("learning_rate", 0.001)),
            batch_size=int(cfg.get("batch_size", 32)),
            epochs=int(cfg.get("epochs", 200)),
            patience=int(cfg.get("patience", 25)),
            loss_function=str(cfg.get("loss_function", "huber")),
            random_seed=int(config.get("project", {}).get("random_seed", 42)),
        )
        pred_train = np.maximum(train_df["baseline_eta_secs"].to_numpy(dtype=float) + predict_residual(result.model, x_train), 0.0)
        pred_val = np.maximum(val_df["baseline_eta_secs"].to_numpy(dtype=float) + predict_residual(result.model, x_val), 0.0)
        with Timer() as timer:
            pred_test = np.maximum(test_df["baseline_eta_secs"].to_numpy(dtype=float) + predict_residual(result.model, x_test), 0.0)
        metrics = evaluate_all_splits(train_df, val_df, test_df, pred_train, pred_val, pred_test)
        metrics["best_epoch"] = float(result.best_epoch)
        metrics["inference_time_ms_per_sample"] = timer.elapsed / max(len(test_df), 1) * 1000.0
        log_metrics_safe(metrics)

        paths = prepared["paths"]
        save_predictions(paths, "mlp_residual_eta", train_df, val_df, test_df, pred_train, pred_val, pred_test)
        save_metrics(paths, "mlp_residual_eta", metrics)
        plot_diagnostics(paths, "mlp_residual_eta", test_df, pred_test)
        _plot_loss(paths, "mlp_residual_eta", result.train_loss, result.val_mae)
        model_path = paths["models_dir"] / "mlp_residual.pt"
        torch.save({"state_dict": result.model.state_dict(), "input_dim": x_train.shape[1], "config": cfg, "features": features}, model_path)
        mlflow.log_artifact(str(model_path))
        save_joblib_model(paths, "mlp_residual_preprocessor.joblib", preprocessor)


if __name__ == "__main__":
    main()
