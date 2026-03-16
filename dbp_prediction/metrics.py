"""Evaluation metrics for regression tasks.

Provides a single source of truth for RMSE, MAE, and R² computation,
replacing the inline metric code duplicated across every script.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, and R² for a single target.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Ground-truth values.
    y_pred : np.ndarray, shape (n,)
        Predicted values.

    Returns
    -------
    dict with keys ``rmse``, ``mae``, ``r2``.
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_per_target_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute RMSE, MAE, R² for each target column independently.

    Parameters
    ----------
    y_true : np.ndarray, shape (n, num_targets)
        Ground-truth values, one column per target.
    y_pred : np.ndarray, shape (n, num_targets)
        Predicted values.
    target_names : list of str, optional
        Column labels.  Defaults to ``target_0``, ``target_1``, etc.

    Returns
    -------
    dict mapping target name → metrics dict.
    """
    num_targets = y_true.shape[1]
    if target_names is None:
        target_names = [f"target_{i}" for i in range(num_targets)]

    results: dict[str, dict[str, float]] = {}
    for i, name in enumerate(target_names):
        results[name] = compute_metrics(y_true[:, i], y_pred[:, i])
    return results


def macro_average(per_target: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute macro-averaged metrics across targets.

    Parameters
    ----------
    per_target : dict
        Output of :func:`compute_per_target_metrics`.

    Returns
    -------
    dict with keys ``rmse``, ``mae``, ``r2`` (averages).
    """
    rows = list(per_target.values())
    return {
        "rmse": float(np.mean([r["rmse"] for r in rows])),
        "mae": float(np.mean([r["mae"] for r in rows])),
        "r2": float(np.mean([r["r2"] for r in rows])),
    }


def print_metrics_table(
    per_target: dict[str, dict[str, float]],
    header: str = "Evaluation on test set (original scale)",
) -> None:
    """Pretty-print a metrics table to stdout.

    Parameters
    ----------
    per_target : dict
        Output of :func:`compute_per_target_metrics`.
    header : str
        Table header text.
    """
    print("\n" + "=" * 60)
    print(header)
    print("=" * 60)
    for name, m in per_target.items():
        print(f"  {name:15s}  RMSE={m['rmse']:7.3f}  MAE={m['mae']:7.3f}  R²={m['r2']:.4f}")

    avg = macro_average(per_target)
    print("-" * 60)
    print(f"  {'Macro Average':15s}  RMSE={avg['rmse']:7.3f}  MAE={avg['mae']:7.3f}  R²={avg['r2']:.4f}")
