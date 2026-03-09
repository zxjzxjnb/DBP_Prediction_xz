"""MLP model definition and builder function."""

from __future__ import annotations

import torch.nn as nn


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
