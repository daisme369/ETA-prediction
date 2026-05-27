"""
DeepETA-inspired PyTorch MVP for Hanoi ETA residual prediction.

Run:
    python model/deepeta_hanoi_mvp.py

Dependencies:
    pip install torch pandas numpy h3

This script is intentionally self-contained:
1. Generates synthetic Hanoi-only trip data.
2. Converts origin/destination coordinates to H3 resolution 8 cells.
3. Bucketizes continuous values with pd.qcut-derived quantile edges.
4. Hashes H3 strings with three independent stable hash functions.
5. Trains a Transformer-based tabular model to predict ETA residual seconds.
6. Evaluates final ETA with:
       predicted_eta_secs = max(baseline_eta_secs + predicted_residual_secs, 0)
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import h3
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: h3. Install it with `pip install h3` and rerun."
    ) from exc


# Strict Hanoi bounding box requested in the prompt.
HANOI_LAT_MIN = 20.80
HANOI_LAT_MAX = 21.20
HANOI_LNG_MIN = 105.70
HANOI_LNG_MAX = 106.00
H3_RESOLUTION = 8

DEFAULT_BUCKETS = 20
DEFAULT_H3_HASH_BINS = 1_000
H3_HASH_SEEDS = (17, 43, 101)

# The prompt explicitly names the distance/ETA/rain features for qcut. The
# lat/lng columns are also bucketized so every input field enters the model via
# embeddings, matching the "all inputs are embeddings" DeepETA-style setup.
BUCKETIZE_COLUMNS = [
    "origin_lat",
    "origin_lng",
    "destination_lat",
    "destination_lng",
    "haversine_distance_meters",
    "baseline_distance_meters",
    "baseline_eta_secs",
    "rain_level",
]

DISCRETE_CARDINALITIES = {
    "hour_of_day": 24,
    "day_of_week": 7,
    "is_rush_hour": 2,
    "is_weekend": 2,
    "is_holiday": 2,
    "is_raining": 2,
    "traffic_level": 5,
    "weather_condition": 4,
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def latlng_to_h3_cell(lat: float, lng: float, resolution: int = H3_RESOLUTION) -> str:
    """Support both h3-py v3 and v4 API names."""
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lng, resolution)
    return h3.geo_to_h3(lat, lng, resolution)


def haversine_meters(
    origin_lat: np.ndarray,
    origin_lng: np.ndarray,
    destination_lat: np.ndarray,
    destination_lng: np.ndarray,
) -> np.ndarray:
    """Vectorized great-circle distance in meters."""
    earth_radius_m = 6_371_000.0
    lat1 = np.radians(origin_lat)
    lng1 = np.radians(origin_lng)
    lat2 = np.radians(destination_lat)
    lng2 = np.radians(destination_lng)
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return earth_radius_m * c


def generate_mock_hanoi_eta_data(n_rows: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic ETA data strictly inside the Hanoi bounding box."""
    rng = np.random.default_rng(seed)

    origin_lat = rng.uniform(HANOI_LAT_MIN, HANOI_LAT_MAX, n_rows)
    origin_lng = rng.uniform(HANOI_LNG_MIN, HANOI_LNG_MAX, n_rows)
    destination_lat = rng.uniform(HANOI_LAT_MIN, HANOI_LAT_MAX, n_rows)
    destination_lng = rng.uniform(HANOI_LNG_MIN, HANOI_LNG_MAX, n_rows)

    hour_of_day = rng.integers(0, 24, n_rows)
    day_of_week = rng.integers(0, 7, n_rows)
    is_weekend = (day_of_week >= 5).astype(np.int64)
    is_holiday = rng.binomial(1, 0.03, n_rows).astype(np.int64)
    is_rush_hour = (
        ((hour_of_day >= 7) & (hour_of_day <= 9))
        | ((hour_of_day >= 16) & (hour_of_day <= 19))
    ).astype(np.int64)
    is_rush_hour = (is_rush_hour * (1 - is_weekend)).astype(np.int64)

    is_raining = rng.binomial(1, 0.26, n_rows).astype(np.int64)
    rain_level = np.where(
        is_raining == 1,
        rng.choice([1, 2, 3, 4], size=n_rows, p=[0.45, 0.30, 0.18, 0.07]),
        0,
    ).astype(np.int64)
    weather_condition = np.select(
        [rain_level == 0, rain_level == 1, rain_level <= 3, rain_level == 4],
        [0, 1, 2, 3],
        default=0,
    ).astype(np.int64)

    traffic_latent = (
        0.75
        + 1.65 * is_rush_hour
        + 0.38 * rain_level
        + 0.70 * is_holiday
        - 0.35 * is_weekend
        + rng.normal(0.0, 0.65, n_rows)
    )
    traffic_level = np.clip(np.rint(traffic_latent), 0, 4).astype(np.int64)

    haversine_distance_meters = haversine_meters(
        origin_lat, origin_lng, destination_lat, destination_lng
    )
    # Routing distance is longer than straight-line distance, especially in dense
    # urban road networks. Keep a minimum so tiny trips still have sane ETAs.
    route_factor = rng.uniform(1.15, 1.75, n_rows) + 0.04 * traffic_level
    baseline_distance_meters = np.maximum(haversine_distance_meters * route_factor, 250.0)

    baseline_speed_kph = (
        36.0
        - 2.1 * traffic_level
        - 1.2 * rain_level
        - 2.5 * is_rush_hour
        + 1.0 * is_weekend
        + rng.normal(0.0, 2.0, n_rows)
    )
    baseline_speed_mps = np.clip(baseline_speed_kph, 7.0, 48.0) / 3.6
    baseline_eta_secs = (
        baseline_distance_meters / baseline_speed_mps
        + 18.0 * traffic_level
        + 20.0 * is_holiday
        + rng.normal(0.0, 25.0, n_rows)
    )
    baseline_eta_secs = np.maximum(baseline_eta_secs, 45.0)

    # Residual encodes what the routing engine missed: rush bottlenecks,
    # local traffic noise, weather drag, holiday/event effects, and occasional
    # route-engine optimism or pessimism.
    residual_secs = (
        42.0 * is_rush_hour
        + 28.0 * traffic_level
        + 24.0 * rain_level
        + 54.0 * is_holiday
        - 16.0 * is_weekend
        + 0.0020 * baseline_distance_meters * traffic_level
        + rng.normal(0.0, 55.0 + 12.0 * traffic_level + 8.0 * rain_level, n_rows)
    )
    actual_eta_secs = np.maximum(baseline_eta_secs + residual_secs, 30.0)
    residual_secs = actual_eta_secs - baseline_eta_secs

    origin_h3 = [
        latlng_to_h3_cell(float(lat), float(lng), H3_RESOLUTION)
        for lat, lng in zip(origin_lat, origin_lng)
    ]
    destination_h3 = [
        latlng_to_h3_cell(float(lat), float(lng), H3_RESOLUTION)
        for lat, lng in zip(destination_lat, destination_lng)
    ]

    df = pd.DataFrame(
        {
            "origin_h3": origin_h3,
            "destination_h3": destination_h3,
            "origin_lat": origin_lat,
            "origin_lng": origin_lng,
            "destination_lat": destination_lat,
            "destination_lng": destination_lng,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_rush_hour": is_rush_hour,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "is_raining": is_raining,
            "haversine_distance_meters": haversine_distance_meters,
            "baseline_distance_meters": baseline_distance_meters,
            "traffic_level": traffic_level,
            "rain_level": rain_level,
            "weather_condition": weather_condition,
            "baseline_eta_secs": baseline_eta_secs,
            "actual_eta_secs": actual_eta_secs,
            "residual_secs": residual_secs,
        }
    )

    validate_hanoi_scope(df)
    return df


def validate_hanoi_scope(df: pd.DataFrame) -> None:
    """Fail fast if synthetic coordinates leave the requested Hanoi box."""
    lat_cols = ["origin_lat", "destination_lat"]
    lng_cols = ["origin_lng", "destination_lng"]
    for col in lat_cols:
        if not df[col].between(HANOI_LAT_MIN, HANOI_LAT_MAX).all():
            raise ValueError(f"{col} contains values outside Hanoi latitude bounds.")
    for col in lng_cols:
        if not df[col].between(HANOI_LNG_MIN, HANOI_LNG_MAX).all():
            raise ValueError(f"{col} contains values outside Hanoi longitude bounds.")


def stable_hash_to_bin(value: str, seed: int, num_bins: int) -> int:
    """Stable seeded hash, unlike Python's process-randomized built-in hash()."""
    payload = f"{seed}|{value}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % num_bins


def add_h3_hash_columns(
    df: pd.DataFrame, num_bins: int = DEFAULT_H3_HASH_BINS
) -> pd.DataFrame:
    """Create three independent compact hash IDs for each H3 string."""
    out = df.copy()
    for prefix, source_col in [("origin", "origin_h3"), ("destination", "destination_h3")]:
        values = out[source_col].astype(str).tolist()
        for i, seed in enumerate(H3_HASH_SEEDS):
            out[f"{prefix}_h3_hash_{i}"] = [
                stable_hash_to_bin(value, seed, num_bins) for value in values
            ]
    return out


@dataclass
class QuantileBucketizer:
    """Fit pd.qcut quantile edges on train data, then reuse them for eval/inference."""

    columns: Sequence[str]
    num_buckets: int = DEFAULT_BUCKETS
    bin_edges_: Dict[str, np.ndarray] | None = None

    def fit(self, df: pd.DataFrame) -> "QuantileBucketizer":
        edges: Dict[str, np.ndarray] = {}
        for col in self.columns:
            _, bins = pd.qcut(
                df[col],
                q=self.num_buckets,
                labels=False,
                retbins=True,
                duplicates="drop",
            )
            bins = np.asarray(bins, dtype=np.float64)
            if bins.size < 2:
                bins = np.array([-np.inf, np.inf], dtype=np.float64)
            else:
                bins[0] = -np.inf
                bins[-1] = np.inf
            edges[col] = bins
        self.bin_edges_ = edges
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.bin_edges_ is None:
            raise RuntimeError("QuantileBucketizer must be fit before transform.")

        out = df.copy()
        for col in self.columns:
            bins = self.bin_edges_[col]
            # np.digitize with internal cut points yields integer IDs from
            # 0..actual_bucket_count-1. For low-cardinality columns like
            # rain_level, fewer than 20 buckets may be used; the embedding table
            # still reserves 20 IDs, leaving some unused.
            bucket_ids = np.digitize(out[col].to_numpy(), bins[1:-1], right=True)
            bucket_ids = np.clip(bucket_ids, 0, self.num_buckets - 1).astype(np.int64)
            out[f"{col}_bucket"] = bucket_ids
        return out


def build_model_frame(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    num_buckets: int,
    h3_hash_bins: int,
) -> Tuple[pd.DataFrame, QuantileBucketizer, List[str], List[int]]:
    """Return feature-engineered frame plus ordered feature metadata."""
    bucketizer = QuantileBucketizer(BUCKETIZE_COLUMNS, num_buckets=num_buckets)
    bucketizer.fit(df.iloc[train_indices])

    out = bucketizer.transform(df)
    out = add_h3_hash_columns(out, num_bins=h3_hash_bins)

    bucket_cols = [f"{col}_bucket" for col in BUCKETIZE_COLUMNS]
    feature_columns = list(DISCRETE_CARDINALITIES.keys()) + bucket_cols
    feature_cardinalities = [
        DISCRETE_CARDINALITIES[col] if col in DISCRETE_CARDINALITIES else num_buckets
        for col in feature_columns
    ]

    for col in DISCRETE_CARDINALITIES:
        out[col] = out[col].astype(np.int64).clip(0, DISCRETE_CARDINALITIES[col] - 1)
    for col in bucket_cols:
        out[col] = out[col].astype(np.int64).clip(0, num_buckets - 1)

    return out, bucketizer, feature_columns, feature_cardinalities


def tensors_from_frame(
    df: pd.DataFrame, feature_columns: Sequence[str]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    origin_hash_cols = [f"origin_h3_hash_{i}" for i in range(len(H3_HASH_SEEDS))]
    destination_hash_cols = [f"destination_h3_hash_{i}" for i in range(len(H3_HASH_SEEDS))]

    feature_ids = torch.as_tensor(df[list(feature_columns)].to_numpy(), dtype=torch.long)
    origin_hash_ids = torch.as_tensor(df[origin_hash_cols].to_numpy(), dtype=torch.long)
    destination_hash_ids = torch.as_tensor(
        df[destination_hash_cols].to_numpy(), dtype=torch.long
    )
    rush_hour_ids = torch.as_tensor(df["is_rush_hour"].to_numpy(), dtype=torch.long)
    residual_targets = torch.as_tensor(
        df["residual_secs"].to_numpy(dtype=np.float32).reshape(-1, 1),
        dtype=torch.float32,
    )
    return feature_ids, origin_hash_ids, destination_hash_ids, rush_hour_ids, residual_targets


class DeepETATabularTransformer(nn.Module):
    """DeepETA-style tabular model with hashed H3 embeddings and Transformer encoder."""

    def __init__(
        self,
        feature_cardinalities: Sequence[int],
        h3_hash_bins: int,
        embedding_dim: int = 16,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_feature_tokens = len(feature_cardinalities) + 2  # + origin and dest H3

        self.feature_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, embedding_dim) for cardinality in feature_cardinalities]
        )
        self.h3_embedding = nn.Embedding(h3_hash_bins, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        flattened_dim = self.num_feature_tokens * embedding_dim
        self.decoder = nn.Sequential(
            nn.LayerNorm(flattened_dim),
            nn.Linear(flattened_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        # Segment bias adjustment: one scalar for normal, one for rush hour.
        self.rush_hour_bias = nn.Embedding(2, 1)

    def forward(
        self,
        feature_ids: torch.Tensor,
        origin_hash_ids: torch.Tensor,
        destination_hash_ids: torch.Tensor,
        rush_hour_ids: torch.Tensor,
    ) -> torch.Tensor:
        feature_tokens = [
            embedding(feature_ids[:, i]) for i, embedding in enumerate(self.feature_embeddings)
        ]

        # Multiple feature hashing: look up three H3 embeddings and sum them into
        # one spatial representation for origin and destination respectively.
        origin_token = self.h3_embedding(origin_hash_ids).sum(dim=1)
        destination_token = self.h3_embedding(destination_hash_ids).sum(dim=1)

        tokens = torch.stack([origin_token, destination_token, *feature_tokens], dim=1)
        encoded = self.encoder(tokens)  # No positional encoding: tabular order is not semantic.
        flattened = encoded.flatten(start_dim=1)
        residual_pred = self.decoder(flattened)
        return residual_pred + self.rush_hour_bias(rush_hour_ids)


class AsymmetricHuberLoss(nn.Module):
    """
    Huber loss with asymmetric weighting.

    error = prediction - target
    Under-prediction has error < 0. Setting omega > 1 penalizes being late in
    the user's experience: predicting a shorter ETA than reality.
    """

    def __init__(self, delta: float = 120.0, omega: float = 1.35) -> None:
        super().__init__()
        self.delta = float(delta)
        self.omega = float(omega)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = prediction - target
        abs_error = torch.abs(error)
        quadratic = 0.5 * error.pow(2)
        linear = self.delta * (abs_error - 0.5 * self.delta)
        huber = torch.where(abs_error <= self.delta, quadratic, linear)
        weights = torch.where(error < 0.0, self.omega, 1.0)
        return (weights * huber).mean()


def make_loaders(
    tensors: Tuple[torch.Tensor, ...],
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = TensorDataset(*(tensor[train_indices] for tensor in tensors))
    eval_dataset = TensorDataset(*(tensor[eval_indices] for tensor in tensors))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, eval_loader


def move_batch_to_device(
    batch: Iterable[torch.Tensor], device: torch.device
) -> Tuple[torch.Tensor, ...]:
    return tuple(item.to(device) for item in batch)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    delta: float,
    omega: float,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = AsymmetricHuberLoss(delta=delta, omega=omega)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []
        for batch in train_loader:
            (
                feature_ids,
                origin_hash_ids,
                destination_hash_ids,
                rush_hour_ids,
                targets,
            ) = move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(
                feature_ids, origin_hash_ids, destination_hash_ids, rush_hour_ids
            )
            loss = loss_fn(predictions, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        eval_losses: List[float] = []
        with torch.no_grad():
            for batch in eval_loader:
                (
                    feature_ids,
                    origin_hash_ids,
                    destination_hash_ids,
                    rush_hour_ids,
                    targets,
                ) = move_batch_to_device(batch, device)
                predictions = model(
                    feature_ids, origin_hash_ids, destination_hash_ids, rush_hour_ids
                )
                eval_loss = loss_fn(predictions, targets)
                eval_losses.append(float(eval_loss.detach().cpu()))

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_asym_huber={np.mean(train_losses):.2f} | "
            f"eval_asym_huber={np.mean(eval_losses):.2f}"
        )


def predict_residuals(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            (
                feature_ids,
                origin_hash_ids,
                destination_hash_ids,
                rush_hour_ids,
                _targets,
            ) = move_batch_to_device(batch, device)
            pred = model(feature_ids, origin_hash_ids, destination_hash_ids, rush_hour_ids)
            predictions.append(pred.detach().cpu().numpy().reshape(-1))
    return np.concatenate(predictions)


def mean_absolute_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Hanoi DeepETA-style MVP.")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buckets", type=int, default=DEFAULT_BUCKETS)
    parser.add_argument("--h3-hash-bins", type=int, default=DEFAULT_H3_HASH_BINS)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--delta", type=float, default=120.0)
    parser.add_argument("--omega", type=float, default=1.35)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = generate_mock_hanoi_eta_data(n_rows=args.rows, seed=args.seed)
    print(f"Generated {len(df):,} Hanoi trips.")
    print(
        "Coordinate ranges: "
        f"lat={df[['origin_lat', 'destination_lat']].min().min():.4f}.."
        f"{df[['origin_lat', 'destination_lat']].max().max():.4f}, "
        f"lng={df[['origin_lng', 'destination_lng']].min().min():.4f}.."
        f"{df[['origin_lng', 'destination_lng']].max().max():.4f}"
    )

    indices = np.arange(len(df))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    split = int(0.80 * len(indices))
    train_indices = indices[:split]
    eval_indices = indices[split:]

    model_df, _bucketizer, feature_columns, feature_cardinalities = build_model_frame(
        df,
        train_indices=train_indices,
        num_buckets=args.buckets,
        h3_hash_bins=args.h3_hash_bins,
    )

    tensors = tensors_from_frame(model_df, feature_columns)
    train_loader, eval_loader = make_loaders(
        tensors=tensors,
        train_indices=train_indices,
        eval_indices=eval_indices,
        batch_size=args.batch_size,
    )

    model = DeepETATabularTransformer(
        feature_cardinalities=feature_cardinalities,
        h3_hash_bins=args.h3_hash_bins,
        embedding_dim=args.embedding_dim,
    ).to(device)

    print(
        f"Training on {device.type} with {len(feature_columns)} tabular tokens "
        f"+ origin/destination H3 tokens."
    )
    train_model(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.lr,
        delta=args.delta,
        omega=args.omega,
    )

    predicted_residual_secs = predict_residuals(model, eval_loader, device=device)
    eval_df = df.iloc[eval_indices].reset_index(drop=True)
    predicted_eta_secs = np.maximum(
        eval_df["baseline_eta_secs"].to_numpy() + predicted_residual_secs,
        0.0,
    )
    actual_eta_secs = eval_df["actual_eta_secs"].to_numpy()
    baseline_eta_secs = eval_df["baseline_eta_secs"].to_numpy()

    model_mae = mean_absolute_error(predicted_eta_secs, actual_eta_secs)
    baseline_mae = mean_absolute_error(baseline_eta_secs, actual_eta_secs)

    print("\nEvaluation")
    print(f"Baseline ETA MAE: {baseline_mae:.2f} seconds")
    print(f"Model ETA MAE:    {model_mae:.2f} seconds")
    print("\nSample predictions")
    preview = pd.DataFrame(
        {
            "baseline_eta_secs": baseline_eta_secs[:8],
            "predicted_residual_secs": predicted_residual_secs[:8],
            "predicted_eta_secs": predicted_eta_secs[:8],
            "actual_eta_secs": actual_eta_secs[:8],
        }
    )
    print(preview.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
