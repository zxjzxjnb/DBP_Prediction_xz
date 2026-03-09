"""Training loop, early stopping, and CV ensemble utilities.

This module extracts the training logic that was duplicated across
``train_mlp.py``, ``train_kan.py``, ``tune_mlp.py``, ``tune_kan.py``,
and ``tune_kan_per_target.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from dbp_prediction.metrics import compute_metrics, compute_per_target_metrics

logger = logging.getLogger(__name__)


# ── Reproducibility ──────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    """Set random seeds for PyTorch and NumPy."""
    torch.manual_seed(seed)
    np.random.seed(seed)


# ── Optimizer / loss factories ───────────────────────────────────────────────


def make_optimizer(
    model: nn.Module,
    optimizer_name: str = "Adam",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Create an optimizer from a string name.

    Supported: ``"Adam"``, ``"AdamW"``.
    """
    if optimizer_name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_loss_fn(loss_name: str = "MSE", huber_delta: float = 1.0) -> nn.Module:
    """Create a loss function from a string name.

    Supported: ``"MSE"``, ``"Huber"``.
    """
    if loss_name == "Huber":
        return nn.SmoothL1Loss(beta=huber_delta)
    return nn.MSELoss()


# ── Single training run ─────────────────────────────────────────────────────


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    optimizer_name: str = "Adam",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    loss_name: str = "MSE",
    huber_delta: float = 1.0,
    batch_size: int = 16,
    max_epochs: int = 2000,
    patience: int = 100,
    max_grad_norm: float = 5.0,
    verbose_every: int = 100,
) -> tuple[nn.Module, float, int, np.ndarray]:
    """Train a model with early stopping and return the best checkpoint.

    Parameters
    ----------
    model : nn.Module
        The model to train (will be modified in-place).
    X_train, Y_train : torch.Tensor
        Scaled training data.
    X_val, Y_val : torch.Tensor
        Scaled validation data.
    optimizer_name : str
        ``"Adam"`` or ``"AdamW"``.
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularisation coefficient.
    loss_name : str
        ``"MSE"`` or ``"Huber"``.
    huber_delta : float
        Delta parameter for Huber loss.
    batch_size : int
        Mini-batch size.
    max_epochs : int
        Maximum training epochs.
    patience : int
        Early-stopping patience (epochs without improvement).
    max_grad_norm : float
        Maximum gradient norm for clipping.  Set to 0 to disable.
    verbose_every : int
        Print progress every N epochs.  Set to 0 to disable.

    Returns
    -------
    tuple of (model, best_val_loss, best_epoch, val_predictions_scaled)
        The model is loaded with the best checkpoint state.
    """
    optimizer = make_optimizer(model, optimizer_name, lr, weight_decay)
    train_loss_fn = make_loss_fn(loss_name, huber_delta)
    val_loss_fn = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_epoch = 0
    wait = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, max_epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = train_loss_fn(pred, yb)
            loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(dataset)

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_mse = val_loss_fn(val_pred, Y_val).item()

        # --- Early stopping ---
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if verbose_every > 0 and (epoch % verbose_every == 0 or wait == patience):
            logger.info(
                "Epoch %4d | train loss: %.4f | val MSE: %.4f | patience: %d/%d",
                epoch, epoch_loss, val_mse, wait, patience,
            )

        if wait >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    # Restore best checkpoint
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_best = model(X_val).detach().cpu().numpy()

    return model, best_val, best_epoch, pred_best


# ── Cross-validation fold helper ────────────────────────────────────────────


def fit_and_eval_fold(
    model_builder: Callable[[], nn.Module],
    X_all: np.ndarray,
    Y_all: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    params: dict[str, Any],
    seed: int,
    max_epochs: int = 2000,
    patience: int = 100,
    keep_member: bool = False,
) -> dict[str, Any]:
    """Train and evaluate a single CV fold.

    Parameters
    ----------
    model_builder : callable
        Zero-argument function that returns a fresh ``nn.Module``.
    X_all : np.ndarray, shape (n, n_features)
        All feature data (unscaled).
    Y_all : np.ndarray, shape (n, n_targets) or (n, 1)
        All target data (unscaled).
    train_idx, val_idx : np.ndarray
        Index arrays for the fold.
    params : dict
        Must contain keys: ``optimizer``, ``lr``, ``weight_decay``,
        ``batch_size``.  Optionally ``loss``, ``huber_delta``.
    seed : int
        Random seed for this fold.
    max_epochs : int
        Max training epochs.
    patience : int
        Early-stopping patience.
    keep_member : bool
        If ``True``, include model state and scalers in the return dict.

    Returns
    -------
    dict with fold-level metrics and (optionally) model state.
    """
    set_seed(seed)

    scaler_x = StandardScaler().fit(X_all[train_idx])
    scaler_y = StandardScaler().fit(Y_all[train_idx])

    X_tr = torch.tensor(scaler_x.transform(X_all[train_idx]), dtype=torch.float32)
    Y_tr = torch.tensor(scaler_y.transform(Y_all[train_idx]), dtype=torch.float32)
    X_va = torch.tensor(scaler_x.transform(X_all[val_idx]), dtype=torch.float32)
    Y_va = torch.tensor(scaler_y.transform(Y_all[val_idx]), dtype=torch.float32)

    model = model_builder()

    model, val_mse_scaled, best_epoch, pred_scaled = train_model(
        model=model,
        X_train=X_tr,
        Y_train=Y_tr,
        X_val=X_va,
        Y_val=Y_va,
        optimizer_name=params.get("optimizer", "Adam"),
        lr=params.get("lr", 1e-3),
        weight_decay=params.get("weight_decay", 1e-4),
        loss_name=params.get("loss", "MSE"),
        huber_delta=params.get("huber_delta", 1.0),
        batch_size=params.get("batch_size", 16),
        max_epochs=max_epochs,
        patience=patience,
        verbose_every=0,
    )

    pred_raw = scaler_y.inverse_transform(pred_scaled)
    y_true_raw = Y_all[val_idx]

    # Compute metrics
    n_targets = Y_all.shape[1] if Y_all.ndim > 1 else 1
    if n_targets == 1:
        metrics = compute_metrics(y_true_raw.ravel(), pred_raw.ravel())
    else:
        per_target = compute_per_target_metrics(y_true_raw, pred_raw)
        rmse_vals = [m["rmse"] for m in per_target.values()]
        mae_vals = [m["mae"] for m in per_target.values()]
        r2_vals = [m["r2"] for m in per_target.values()]
        metrics = {
            "rmse": float(np.mean(rmse_vals)),
            "mae": float(np.mean(mae_vals)),
            "r2": float(np.mean(r2_vals)),
            "rmse_per_target": rmse_vals,
            "mae_per_target": mae_vals,
            "r2_per_target": r2_vals,
        }

    result: dict[str, Any] = {
        "val_mse_scaled": float(val_mse_scaled),
        "best_epoch": int(best_epoch),
        **metrics,
    }

    if keep_member:
        result["member"] = {
            "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "scaler_x": scaler_x,
            "scaler_y": scaler_y,
            "best_epoch": int(best_epoch),
        }

    return result


# ── CV ensemble ──────────────────────────────────────────────────────────────


def train_cv_ensemble(
    model_builder: Callable[[], nn.Module],
    X_train_all: np.ndarray,
    Y_train_all: np.ndarray,
    params: dict[str, Any],
    seed: int,
    folds: int = 5,
    max_epochs: int = 2000,
    patience: int = 100,
) -> tuple[list[dict], dict[str, Any]]:
    """Train a K-fold CV ensemble and return members + summary metrics.

    Parameters
    ----------
    model_builder : callable
        Zero-argument function that returns a fresh ``nn.Module``.
    X_train_all, Y_train_all : np.ndarray
        Full training data (unscaled).
    params : dict
        Hyperparameters (passed to :func:`fit_and_eval_fold`).
    seed : int
        Random seed for K-fold splitting.
    folds : int
        Number of CV folds.
    max_epochs, patience : int
        Training constraints.

    Returns
    -------
    tuple of (members_list, summary_dict)
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    members: list[dict] = []
    fold_rmses: list[float] = []
    fold_maes: list[float] = []
    fold_r2s: list[float] = []
    fold_epochs: list[int] = []

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_train_all), start=1):
        fold_result = fit_and_eval_fold(
            model_builder=model_builder,
            X_all=X_train_all,
            Y_all=Y_train_all,
            train_idx=tr_idx,
            val_idx=va_idx,
            params=params,
            seed=seed + fold_id,
            max_epochs=max_epochs,
            patience=patience,
            keep_member=True,
        )

        members.append(fold_result["member"])
        fold_rmses.append(fold_result["rmse"])
        fold_maes.append(fold_result["mae"])
        fold_r2s.append(fold_result["r2"])
        fold_epochs.append(fold_result["best_epoch"])

        print(
            f"    Fold {fold_id}/{folds} | RMSE={fold_result['rmse']:.3f} "
            f"MAE={fold_result['mae']:.3f} R²={fold_result['r2']:.4f} "
            f"best_epoch={fold_result['best_epoch']}"
        )

    summary = {
        "cv_rmse_mean": float(np.mean(fold_rmses)),
        "cv_rmse_std": float(np.std(fold_rmses)),
        "cv_mae_mean": float(np.mean(fold_maes)),
        "cv_r2_mean": float(np.mean(fold_r2s)),
        "cv_best_epochs": fold_epochs,
    }
    return members, summary


def predict_with_ensemble(
    model_builder: Callable[[], nn.Module],
    members: list[dict],
    X_test_raw: np.ndarray,
) -> np.ndarray:
    """Generate predictions by averaging across CV ensemble members.

    Parameters
    ----------
    model_builder : callable
        Zero-argument function that returns a fresh ``nn.Module``
        (architecture must match the saved states).
    members : list of dict
        Each dict contains ``model_state``, ``scaler_x``, ``scaler_y``.
    X_test_raw : np.ndarray
        Unscaled test features.

    Returns
    -------
    np.ndarray
        Ensemble-averaged predictions in original scale.
    """
    preds = []

    for member in members:
        model = model_builder()
        model.load_state_dict(member["model_state"])
        model.eval()

        X_te = torch.tensor(member["scaler_x"].transform(X_test_raw), dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = model(X_te).detach().cpu().numpy()

        pred_raw = member["scaler_y"].inverse_transform(pred_scaled)
        preds.append(pred_raw)

    return np.mean(np.stack(preds, axis=0), axis=0)
