"""MLP model definition and builder function."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from dbp_prediction.models.base import TorchModelAdapter, register_model_adapter

MLP_HIDDEN_DIMS_MAP = {
    "linear": [],
    "8": [8],
    "16": [16],
    "24": [24],
    "32": [32],
    "48": [48],
    "64": [64],
    "8-8": [8, 8],
    "16-16": [16, 16],
    "24-24": [24, 24],
    "32-32": [32, 32],
    "48-48": [48, 48],
    "64-64": [64, 64],
    "8-8-8": [8, 8, 8],
    "16-16-16": [16, 16, 16],
    "24-24-24": [24, 24, 24],
    "32-32-32": [32, 32, 32],
    "48-48-48": [48, 48, 48],
    "64-64-64": [64, 64, 64],
}


class MLP(nn.Module):
    """Multi-layer perceptron for regression.

    Parameters
    ----------
    in_dim : int
        Number of input features.
    out_dim : int
        Number of output targets.
    hidden_dims : list of int
        Hidden layer widths.
    dropout : float
        Dropout probability after each hidden layer.
    activation : str
        Activation function name: ``"ReLU"``, ``"LeakyReLU"``, ``"SiLU"``, ``"Tanh"``.
    """

    ACTIVATIONS = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "SiLU": nn.SiLU,
        "Tanh": nn.Tanh,
    }

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.2,
        activation: str = "ReLU",
    ) -> None:
        super().__init__()

        act_cls = self.ACTIVATIONS.get(activation)
        if act_cls is None:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Supported: {list(self.ACTIVATIONS.keys())}"
            )

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), act_cls(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_mlp(
    in_dim: int,
    out_dim: int,
    n_layers: int = 2,
    hidden_dim: int = 32,
    dropout: float = 0.2,
    activation: str = "ReLU",
) -> nn.Module:
    """Build an MLP from flat hyperparameters (convenience for Optuna).

    If ``n_layers == 0``, returns a plain ``nn.Linear``.

    Parameters
    ----------
    in_dim : int
        Number of input features.
    out_dim : int
        Number of output targets.
    n_layers : int
        Number of hidden layers.
    hidden_dim : int
        Width of each hidden layer (uniform).
    dropout : float
        Dropout probability.
    activation : str
        Activation function name.

    Returns
    -------
    nn.Module
    """
    if n_layers == 0:
        return nn.Linear(in_dim, out_dim)

    hidden_dims = [hidden_dim] * n_layers
    return MLP(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
    )


@register_model_adapter("mlp")
class MLPAdapter(TorchModelAdapter):
    """ModelAdapter implementation for the baseline MLP."""

    def search_space(self) -> dict[str, Any]:
        return {
            "model": {
                "hidden_dims": {
                    "type": "categorical",
                    "choices": list(MLP_HIDDEN_DIMS_MAP),
                    "value_map": MLP_HIDDEN_DIMS_MAP,
                },
                "dropout": {
                    "type": "float",
                    "low": 0.0,
                    "high": 0.5,
                    "step": 0.05,
                },
                "activation": {
                    "type": "categorical",
                    "choices": ["ReLU", "LeakyReLU", "SiLU", "Tanh"],
                },
            },
            "training": {
                "lr": {"type": "float", "low": 3e-4, "high": 2e-2, "log": True},
                "weight_decay": {"type": "float", "low": 1e-7, "high": 2e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [8, 16, 32]},
                "optimizer": {"type": "categorical", "choices": ["Adam", "AdamW"]},
                "loss": {"type": "categorical", "choices": ["MSE", "Huber"]},
                "huber_delta": {
                    "type": "categorical",
                    "choices": [0.5, 1.0, 2.0, 4.0],
                },
            },
            "study": {
                "n_startup_trials": 15,
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
        del seed
        return MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=list(model_params.get("hidden_dims", [32, 16])),
            dropout=float(model_params.get("dropout", 0.2)),
            activation=str(model_params.get("activation", "ReLU")),
        )
