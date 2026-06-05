from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ResidualMLP(nn.Module):
    """Simple tabular MLP that predicts residual seconds."""

    def __init__(self, input_dim: int, hidden_sizes: Iterable[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(current_dim, int(hidden_dim)),
                    nn.ReLU(),
                    nn.BatchNorm1d(int(hidden_dim)),
                    nn.Dropout(float(dropout)),
                ]
            )
            current_dim = int(hidden_dim)
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


@dataclass
class MLPTrainingResult:
    model: ResidualMLP
    train_loss: list[float]
    val_mae: list[float]
    best_epoch: int


def _loss_fn(name: str) -> nn.Module:
    if name.lower() in {"mae", "l1", "l1loss"}:
        return nn.L1Loss()
    return nn.HuberLoss()


def train_mlp_residual_model(
    x_train: np.ndarray,
    residual_train: np.ndarray,
    baseline_train: np.ndarray,
    actual_train: np.ndarray,
    x_val: np.ndarray,
    residual_val: np.ndarray,
    baseline_val: np.ndarray,
    actual_val: np.ndarray,
    *,
    hidden_sizes: list[int],
    dropout: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    loss_function: str,
    random_seed: int,
) -> MLPTrainingResult:
    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualMLP(x_train.shape[1], hidden_sizes, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = _loss_fn(loss_function)

    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(residual_train, dtype=torch.float32),
        torch.tensor(baseline_train, dtype=torch.float32),
        torch.tensor(actual_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    baseline_val_t = torch.tensor(baseline_val, dtype=torch.float32, device=device)
    actual_val_t = torch.tensor(actual_val, dtype=torch.float32, device=device)

    train_losses: list[float] = []
    val_maes: list[float] = []
    best_state = None
    best_val_mae = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(int(epochs)):
        model.train()
        batch_losses = []
        for xb, residual_yb, baseline_yb, actual_yb in loader:
            xb = xb.to(device)
            residual_yb = residual_yb.to(device)
            baseline_yb = baseline_yb.to(device)
            actual_yb = actual_yb.to(device)
            optimizer.zero_grad()
            pred_residual = model(xb)
            pred_eta = torch.clamp(baseline_yb + pred_residual, min=0.0)
            residual_loss = criterion(pred_residual, residual_yb)
            eta_loss = criterion(pred_eta, actual_yb)
            loss = 0.5 * residual_loss + 0.5 * eta_loss
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred_eta = torch.clamp(baseline_val_t + model(x_val_t), min=0.0)
            val_mae = torch.mean(torch.abs(actual_val_t - val_pred_eta)).item()

        train_losses.append(float(np.mean(batch_losses)))
        val_maes.append(float(val_mae))
        if val_mae < best_val_mae:
            best_val_mae = float(val_mae)
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return MLPTrainingResult(model=model, train_loss=train_losses, val_mae=val_maes, best_epoch=best_epoch)


def predict_residual(model: ResidualMLP, x: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return pred.astype(float)
