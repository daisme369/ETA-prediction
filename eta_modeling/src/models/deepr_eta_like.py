from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class QuantileBucketizer:
    """Quantile bucketizer fit on train data only."""

    def __init__(self, n_buckets: int) -> None:
        self.n_buckets = int(n_buckets)
        self.edges: np.ndarray | None = None

    def fit(self, values: pd.Series | np.ndarray) -> "QuantileBucketizer":
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            self.edges = np.array([], dtype=float)
            return self
        quantiles = np.linspace(0, 1, self.n_buckets + 1)[1:-1]
        self.edges = np.unique(np.quantile(arr, quantiles))
        return self

    def transform(self, values: pd.Series | np.ndarray) -> np.ndarray:
        if self.edges is None:
            raise RuntimeError("Bucketizer is not fitted.")
        arr = np.asarray(values, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return np.digitize(arr, self.edges, right=False).astype(np.int64)

    @property
    def cardinality(self) -> int:
        if self.edges is None:
            raise RuntimeError("Bucketizer is not fitted.")
        return int(len(self.edges) + 1)

    def to_artifact(self) -> dict[str, Any]:
        """Return serializable bucket metadata for reproducible inference."""
        if self.edges is None:
            raise RuntimeError("Bucketizer is not fitted.")
        return {
            "n_buckets_requested": self.n_buckets,
            "n_buckets_fitted": self.cardinality,
            "edges": self.edges.tolist(),
        }


@dataclass
class DeepFeatureEncoder:
    categorical_features: list[str]
    bucket_features: dict[str, int]
    calibration_feature: str | None = None

    def __post_init__(self) -> None:
        self.category_maps: dict[str, dict[str, int]] = {}
        self.bucketizers: dict[str, QuantileBucketizer] = {
            feature: QuantileBucketizer(n_buckets) for feature, n_buckets in self.bucket_features.items()
        }

    def fit(self, df: pd.DataFrame) -> "DeepFeatureEncoder":
        for feature in self.categorical_features:
            values = sorted(df[feature].astype(str).fillna("__missing__").unique().tolist())
            self.category_maps[feature] = {value: idx + 1 for idx, value in enumerate(values)}
        for feature, bucketizer in self.bucketizers.items():
            bucketizer.fit(df[feature])
        return self

    def transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        encoded: dict[str, np.ndarray] = {}
        for feature in self.categorical_features:
            mapping = self.category_maps[feature]
            encoded[feature] = (
                df[feature]
                .astype(str)
                .fillna("__missing__")
                .map(mapping)
                .fillna(0)
                .astype(np.int64)
                .to_numpy()
            )
        for feature, bucketizer in self.bucketizers.items():
            encoded[feature] = bucketizer.transform(df[feature])
        return encoded

    def cardinalities(self) -> dict[str, int]:
        cards = {feature: len(mapping) + 1 for feature, mapping in self.category_maps.items()}
        cards.update({feature: bucketizer.cardinality for feature, bucketizer in self.bucketizers.items()})
        return cards

    def bucketizer_artifact(self) -> dict[str, Any]:
        """Return fitted quantile bucket boundaries by feature."""
        return {
            feature: bucketizer.to_artifact()
            for feature, bucketizer in self.bucketizers.items()
        }


class DeepETADataset(Dataset):
    def __init__(self, encoded: dict[str, np.ndarray], baseline: np.ndarray, actual: np.ndarray) -> None:
        self.encoded = {key: torch.tensor(value, dtype=torch.long) for key, value in encoded.items()}
        self.baseline = torch.tensor(baseline, dtype=torch.float32)
        self.actual = torch.tensor(actual, dtype=torch.float32)
        self.length = len(actual)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        return {key: value[idx] for key, value in self.encoded.items()}, self.baseline[idx], self.actual[idx]


class DeepRETALikeModel(nn.Module):
    """Simplified DeeprETA-like residual model using feature embeddings."""

    def __init__(
        self,
        cardinalities: dict[str, int],
        embedding_dim: int,
        hidden_sizes: list[int],
        dropout: float,
        use_attention: bool,
        calibration_feature: str | None,
    ) -> None:
        super().__init__()
        self.feature_names = list(cardinalities)
        self.embeddings = nn.ModuleDict(
            {
                feature: nn.Embedding(cardinality, embedding_dim, padding_idx=0)
                for feature, cardinality in cardinalities.items()
            }
        )
        self.use_attention = bool(use_attention)
        self.attention = (
            nn.MultiheadAttention(embedding_dim, num_heads=1, batch_first=True)
            if self.use_attention
            else None
        )
        input_dim = len(self.feature_names) * embedding_dim
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.decoder = nn.Sequential(*layers)
        self.calibration_feature = calibration_feature
        self.calibration = (
            nn.Embedding(cardinalities[calibration_feature], 1, padding_idx=0)
            if calibration_feature and calibration_feature in cardinalities
            else None
        )

    def forward(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = torch.stack([self.embeddings[name](encoded[name]) for name in self.feature_names], dim=1)
        if self.attention is not None:
            attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
            tokens = tokens + attended
        flat = tokens.flatten(start_dim=1)
        residual = self.decoder(flat).squeeze(-1)
        if self.calibration is not None and self.calibration_feature is not None:
            residual = residual + self.calibration(encoded[self.calibration_feature]).squeeze(-1)
        return residual


class AsymmetricHuberLoss(nn.Module):
    def __init__(self, delta: float, underprediction_weight: float, overprediction_weight: float) -> None:
        super().__init__()
        self.delta = float(delta)
        self.underprediction_weight = float(underprediction_weight)
        self.overprediction_weight = float(overprediction_weight)

    def forward(self, pred_eta: torch.Tensor, actual_eta: torch.Tensor) -> torch.Tensor:
        error = actual_eta - pred_eta
        abs_error = torch.abs(error)
        base = torch.where(
            abs_error <= self.delta,
            0.5 * error.pow(2),
            self.delta * (abs_error - 0.5 * self.delta),
        )
        weights = torch.where(error > 0, self.underprediction_weight, self.overprediction_weight)
        return torch.mean(base * weights)


def make_loss(config: dict[str, Any]) -> nn.Module:
    asym = config.get("asymmetric_huber", {})
    if asym.get("enabled", False):
        return AsymmetricHuberLoss(
            delta=asym.get("delta", 30.0),
            underprediction_weight=asym.get("underprediction_weight", 1.25),
            overprediction_weight=asym.get("overprediction_weight", 1.0),
        )
    loss_name = str(config.get("loss_function", "huber")).lower()
    if loss_name in {"mae", "l1", "l1loss"}:
        return nn.L1Loss()
    return nn.HuberLoss()


@dataclass
class DeepTrainingResult:
    model: DeepRETALikeModel
    train_loss: list[float]
    val_mae: list[float]
    best_epoch: int


def _move_batch(encoded: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in encoded.items()}


def train_deepr_eta_like_model(
    encoder: DeepFeatureEncoder,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict[str, Any],
    random_seed: int,
) -> DeepTrainingResult:
    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = DeepETADataset(
        encoder.transform(train_df),
        train_df["baseline_eta_secs"].to_numpy(dtype=float),
        train_df["actual_eta_secs"].to_numpy(dtype=float),
    )
    val_encoded = encoder.transform(val_df)
    val_baseline = torch.tensor(val_df["baseline_eta_secs"].to_numpy(dtype=float), dtype=torch.float32, device=device)
    val_actual = torch.tensor(val_df["actual_eta_secs"].to_numpy(dtype=float), dtype=torch.float32, device=device)
    val_encoded_t = {key: torch.tensor(value, dtype=torch.long, device=device) for key, value in val_encoded.items()}

    model = DeepRETALikeModel(
        encoder.cardinalities(),
        embedding_dim=int(config.get("embedding_dim", 16)),
        hidden_sizes=[int(x) for x in config.get("hidden_sizes", [128, 64])],
        dropout=float(config.get("dropout", 0.15)),
        use_attention=bool(config.get("use_attention", False)),
        calibration_feature=config.get("calibration_feature") if config.get("use_calibration", True) else None,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 0.001)))
    criterion = make_loss(config)
    loader = DataLoader(train_dataset, batch_size=int(config.get("batch_size", 32)), shuffle=True)

    best_state = None
    best_epoch = 0
    best_val_mae = float("inf")
    stale_epochs = 0
    train_losses: list[float] = []
    val_maes: list[float] = []

    for epoch in range(int(config.get("epochs", 200))):
        model.train()
        losses = []
        for encoded, baseline, actual in loader:
            encoded = _move_batch(encoded, device)
            baseline = baseline.to(device)
            actual = actual.to(device)
            optimizer.zero_grad()
            pred_eta = torch.clamp(baseline + model(encoded), min=0.0)
            loss = criterion(pred_eta, actual)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            pred_eta_val = torch.clamp(val_baseline + model(val_encoded_t), min=0.0)
            val_mae = torch.mean(torch.abs(val_actual - pred_eta_val)).item()
        train_losses.append(float(np.mean(losses)))
        val_maes.append(float(val_mae))
        if val_mae < best_val_mae:
            best_val_mae = float(val_mae)
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config.get("patience", 25)):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return DeepTrainingResult(model=model, train_loss=train_losses, val_mae=val_maes, best_epoch=best_epoch)


def predict_deepr_eta(model: DeepRETALikeModel, encoder: DeepFeatureEncoder, df: pd.DataFrame) -> np.ndarray:
    device = next(model.parameters()).device
    encoded = {key: torch.tensor(value, dtype=torch.long, device=device) for key, value in encoder.transform(df).items()}
    baseline = torch.tensor(df["baseline_eta_secs"].to_numpy(dtype=float), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pred_eta = torch.clamp(baseline + model(encoded), min=0.0).detach().cpu().numpy()
    return pred_eta.astype(float)
