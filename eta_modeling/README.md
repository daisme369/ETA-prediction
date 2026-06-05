# ETA Fixed-Trip Modeling Pipeline

This package runs end-to-end ETA experiments for one fixed origin-destination station pair. It starts with the Vietmap ETA baseline, then trains direct and residual models, ending with a simplified DeeprETA-like neural residual model.

## Objective

Predict actual trip travel time:

- `baseline_eta_secs = estimate_time` from Vietmap API output log.
- `actual_eta_secs = delta_time`.
- `residual_secs = actual_eta_secs - baseline_eta_secs`.

Direct models predict ETA directly. Residual models predict a correction over Vietmap:

```text
pred_eta = baseline_eta_secs + pred_residual
```

All reported business metrics are computed on `pred_eta` vs `actual_eta_secs`, not only residual error.

## Data

The default config reads:

- `../data/processed_data.csv`
- `../data/output_log.csv`

The loader also supports the common typos `proccessed_data.csv` and `ouput_log.csv` as fallbacks. Required columns include:

```text
stationId, destination_stationId, hour, lat, lng,
destination_lat, destination_lng, delta_time, timestamp,
weekday, month, is_weekend, is_rush_hour, date, time,
origin_rain, destination_rain, rain
```

The output log must contain `estimate_time`.

## Fixed OD Pair

Configure a fixed pair in `configs/config.yaml`:

```yaml
fixed_trip:
  stationId: null
  destination_stationId: null
```

When either value is `null`, the pipeline selects the most frequent OD pair and logs it as a warning and MLflow parameter.

## Run Experiments

From this directory:

```bash
pip install -r requirements.txt
python -m src.training.train_baseline --config configs/config.yaml
python -m src.training.train_xgboost_direct --config configs/config.yaml
python -m src.training.train_xgboost_residual --config configs/config.yaml
python -m src.training.train_mlp_residual --config configs/config.yaml
python -m src.training.train_deepr_eta_like --config configs/config.yaml
python -m src.training.train_hour_bin_median --config configs/config.yaml
python -m src.training.train_hour_bin_xgboost_residual --config configs/config.yaml
python -m src.training.train_hour_bin_mlp_residual --config configs/config.yaml
python -m src.training.train_hour_bin_deepr_eta_like --config configs/config.yaml
python -m src.training.compare_models --config configs/config.yaml
```

The model stages are:

- Stage 0: Vietmap baseline evaluation.
- Stage 1: XGBoost direct ETA model.
- Stage 2: XGBoost residual model.
- Stage 3: PyTorch MLP residual model.
- Stage 4: DeeprETA-like embedding residual model.
- Hour-bin variants: replace sparse raw `hour`, `hour_sin`, and `hour_cos` with a categorical `hour_bin`, then train bin median, XGBoost residual, MLP residual, and DeeprETA-like residual models.

Hour-bin mapping:

```text
5-6   -> early_morning
7-9   -> morning_peak
10-14 -> off_peak_midday
15-18 -> afternoon_evening_peak
19-21 -> late_evening_low_service
other -> outside configured service hours
```

## MLflow

Each stage logs parameters, metrics, predictions, plots, and model artifacts to local MLflow tracking:

```bash
mlflow ui
```

Open the UI at `http://127.0.0.1:5000`.

Experiment names:

```text
eta_fixed_trip_baseline
eta_fixed_trip_xgboost_direct
eta_fixed_trip_xgboost_residual
eta_fixed_trip_mlp_residual
eta_fixed_trip_deepr_eta_like
eta_fixed_trip_hour_bin_median
eta_fixed_trip_hour_bin_xgboost_residual
eta_fixed_trip_hour_bin_mlp_residual
eta_fixed_trip_hour_bin_deepr_eta_like
eta_fixed_trip_comparison
```

## Artifacts

Outputs are saved under:

```text
artifacts/
  plots/
  metrics/
    predictions/
  models/
```

Important files:

- `artifacts/models/xgb_direct_eta.joblib`
- `artifacts/models/xgb_residual_eta.joblib`
- `artifacts/models/mlp_residual.pt`
- `artifacts/models/deepr_eta_like.pt`
- `artifacts/models/hour_bin_median_eta.joblib`
- `artifacts/models/hour_bin_xgb_residual_eta.joblib`
- `artifacts/models/hour_bin_mlp_residual_eta.pt`
- `artifacts/models/hour_bin_deepr_eta_like.pt`
- `artifacts/metrics/model_comparison.csv`
- `artifacts/plots/model_comparison_mae_p95.png`

## Validation And Splitting

The loader validates missing targets, non-positive actual ETA, missing or negative baseline ETA, parseable timestamps, and minimum fixed-route sample count. Splits are chronological:

```text
train: first 70%
validation: next 15%
test: last 15%
```

Preprocessors, encoders, scalers, and quantile bucketizers are fit only on train data.

## Known Limitations

The current dataset is small for neural models, so MLP and DeeprETA-like results should be treated as experiment scaffolding, not production quality. The fixed OD pair keeps route features mostly constant; the embedding model becomes more useful when training across many OD pairs and richer traffic/request context.

## Next Steps

Add more OD pairs, include real traffic/context features, tune residual objectives, evaluate calibration by request context, and register the best model in an MLflow model registry when the dataset is large enough.
