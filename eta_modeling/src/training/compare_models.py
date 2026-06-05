from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from ..data_loading import load_config, resolve_path
from ..metrics import eta_metrics_with_baseline
from ..mlflow_utils import configure_mlflow, ensure_artifact_dirs, log_metrics_safe
from .common import configure_logging, parse_args


def _load_metric_json(metrics_dir: Path, model_name: str) -> dict:
    path = metrics_dir / f"{model_name}_metrics.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_artifact_dirs(config)
    configure_mlflow(config, "eta_fixed_trip_comparison")

    rows = []
    for pred_path in sorted(paths["predictions_dir"].glob("*_predictions.csv")):
        pred = pd.read_csv(pred_path)
        if "split" in pred.columns:
            pred = pred[pred["split"] == "test"].copy()
        if pred.empty:
            continue
        model_name = str(pred["model_name"].iloc[0])
        metric_payload = _load_metric_json(paths["metrics_dir"], model_name)
        metrics = eta_metrics_with_baseline(
            pred["actual_eta_secs"].to_numpy(dtype=float),
            pred["pred_eta_secs"].to_numpy(dtype=float),
            pred["baseline_eta_secs"].to_numpy(dtype=float),
            "test",
        )
        rows.append(
            {
                "model_name": model_name,
                "test_mae": metrics["test_mae"],
                "test_mape": metrics["test_mape"],
                "test_rmse": metrics["test_rmse"],
                "test_p50": metrics["test_p50"],
                "test_p95": metrics["test_p95"],
                "baseline_mae": metrics["test_baseline_mae"],
                "mae_improvement_pct": metrics["test_mae_improvement_pct"],
                "p95_improvement_pct": metrics["test_p95_improvement_pct"],
                "inference_time_ms_per_sample": metric_payload.get("inference_time_ms_per_sample"),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No prediction files found in {paths['predictions_dir']}")

    comparison = pd.DataFrame(rows).sort_values("test_mae")
    output_path = paths["metrics_dir"] / "model_comparison.csv"
    comparison.to_csv(output_path, index=False)

    plot_path = paths["plots_dir"] / "model_comparison_mae_p95.png"
    plt.figure(figsize=(9, 5))
    x = range(len(comparison))
    plt.bar([i - 0.2 for i in x], comparison["test_mae"], width=0.4, label="MAE")
    plt.bar([i + 0.2 for i in x], comparison["test_p95"], width=0.4, label="p95 abs error")
    plt.xticks(list(x), comparison["model_name"], rotation=30, ha="right")
    plt.ylabel("Seconds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=140)
    plt.close()

    with mlflow.start_run(run_name="model_comparison"):
        mlflow.log_artifact(str(output_path))
        mlflow.log_artifact(str(plot_path))
        best = comparison.iloc[0].to_dict()
        log_metrics_safe({"best_test_mae": best["test_mae"], "best_test_p95": best["test_p95"]})
        mlflow.log_param("best_model", best["model_name"])

    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
