"""KAN (Kolmogorov-Arnold Network) model builder.

Wraps the ``pykan`` library's ``KAN`` class with sensible project defaults.
"""

from __future__ import annotations

import os
from typing import Any

import torch.nn as nn

from dbp_prediction.models.base import TorchModelAdapter, register_model_adapter

# KAN is lazily imported to avoid triggering matplotlib initialisation on
# package load.  The actual import happens inside ``_get_kan_class()``.
_KAN_CLASS = None


def _get_kan_class() -> type:
    """Lazily import and return the KAN class from pykan."""
    global _KAN_CLASS
    if _KAN_CLASS is None:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
        from kan import KAN
        _KAN_CLASS = KAN
    return _KAN_CLASS

KAN_HIDDEN_DIMS_MAP = {
    "8": [8],
    "16": [16],
    "32": [32],
    "16-8": [16, 8],
    "24-12": [24, 12],
    "32-16": [32, 16],
}


def build_kan(
    in_dim: int,
    out_dim: int,
    hidden_dims: list[int] | tuple[int, ...] = (32, 16),
    grid: int = 8,
    k: int = 3,
    base_fun: str = "silu",
    symbolic_enabled: bool = False,
    seed: int = 42,
    device: str = "cpu",
) -> nn.Module:
    """Build a KAN model.

    Parameters
    ----------
    in_dim : int
        Number of input features.
    out_dim : int
        Number of output targets.
    hidden_dims : list or tuple of int
        Hidden layer widths.
    grid : int
        Number of grid intervals for the B-spline basis.
    k : int
        Spline order.
    base_fun : str
        Base activation function (e.g. ``"silu"``).
    symbolic_enabled : bool
        Whether to enable symbolic regression in KAN.
    seed : int
        Random seed for KAN initialization.
    device : str
        Device string (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    nn.Module
        A KAN model instance.
    """
    width = [in_dim] + list(hidden_dims) + [out_dim]
    return _get_kan_class()(
        width=width,
        grid=grid,
        k=k,
        base_fun=base_fun,
        symbolic_enabled=symbolic_enabled,
        save_act=False,
        auto_save=False,
        seed=seed,
        device=device,
    )


def build_kan_from_params(
    in_dim: int,
    out_dim: int,
    params: dict,
    seed: int = 42,
) -> nn.Module:
    """Build a KAN model from a hyperparameter dict (used by Optuna tuning).

    Parameters
    ----------
    in_dim, out_dim : int
        Input/output dimensions.
    params : dict
        Must contain ``hidden_dims``, ``grid``, ``k``.
    seed : int
        Random seed.

    Returns
    -------
    nn.Module
    """
    return build_kan(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dims=params["hidden_dims"],
        grid=params["grid"],
        k=params["k"],
        seed=seed,
    )


@register_model_adapter("kan")
class KANAdapter(TorchModelAdapter):
    """ModelAdapter implementation for the baseline KAN."""

    def search_space(self) -> dict[str, Any]:
        return {
            "model": {
                "hidden_dims": {
                    "type": "categorical",
                    "choices": list(KAN_HIDDEN_DIMS_MAP),
                    "value_map": KAN_HIDDEN_DIMS_MAP,
                },
                "grid": {"type": "categorical", "choices": [3, 5, 8]},
                "k": {"type": "categorical", "choices": [3, 5]},
            },
            "training": {
                "lr": {"type": "float", "low": 2e-4, "high": 8e-3, "log": True},
                "weight_decay": {"type": "float", "low": 1e-7, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [8, 16, 32]},
                "optimizer": {"type": "categorical", "choices": ["Adam", "AdamW"]},
            },
            "study": {
                "n_startup_trials": 10,
                "n_warmup_steps": 2,
            },
        }

    def build_model(
        self,
        *,
        in_dim: int,
        out_dim: int,
        model_params: dict[str, Any],
        seed: int,
    ) -> nn.Module:
        return build_kan(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=list(model_params.get("hidden_dims", [32, 16])),
            grid=int(model_params.get("grid", 8)),
            k=int(model_params.get("k", 3)),
            base_fun=str(model_params.get("base_fun", "silu")),
            seed=seed,
        )
