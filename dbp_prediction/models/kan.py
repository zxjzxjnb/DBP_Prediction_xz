"""KAN (Kolmogorov-Arnold Network) model builder.

Wraps the ``pykan`` library's ``KAN`` class with sensible project defaults.
"""

from __future__ import annotations

import os

import torch.nn as nn

# Keep KAN/matplotlib cache in writable paths to avoid permission errors.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from kan import KAN  # noqa: E402 — must be imported after env setup


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
    return KAN(
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
