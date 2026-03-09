"""Tests for model construction and forward passes."""

from __future__ import annotations

import pytest
import torch

from dbp_prediction.models.mlp import MLP, build_mlp


class TestMLP:
    """Tests for the MLP model."""

    def test_forward_shape(self) -> None:
        model = MLP(in_dim=9, out_dim=3, hidden_dims=[32, 16])
        x = torch.randn(5, 9)
        y = model(x)
        assert y.shape == (5, 3)

    def test_single_layer(self) -> None:
        model = MLP(in_dim=4, out_dim=1, hidden_dims=[8])
        x = torch.randn(3, 4)
        y = model(x)
        assert y.shape == (3, 1)

    def test_invalid_activation_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown activation"):
            MLP(in_dim=4, out_dim=1, hidden_dims=[8], activation="GELU")

    def test_build_mlp_zero_layers(self) -> None:
        model = build_mlp(in_dim=9, out_dim=3, n_layers=0)
        x = torch.randn(4, 9)
        y = model(x)
        assert y.shape == (4, 3)

    def test_build_mlp_multiple_layers(self) -> None:
        model = build_mlp(in_dim=9, out_dim=1, n_layers=3, hidden_dim=16)
        x = torch.randn(4, 9)
        y = model(x)
        assert y.shape == (4, 1)

    def test_all_activations(self) -> None:
        for act in ["ReLU", "LeakyReLU", "SiLU", "Tanh"]:
            model = MLP(in_dim=4, out_dim=1, hidden_dims=[8], activation=act)
            y = model(torch.randn(2, 4))
            assert y.shape == (2, 1), f"Failed for activation {act}"

    def test_parameter_count_increases_with_layers(self) -> None:
        small = build_mlp(9, 3, n_layers=1, hidden_dim=8)
        large = build_mlp(9, 3, n_layers=3, hidden_dim=32)
        small_params = sum(p.numel() for p in small.parameters())
        large_params = sum(p.numel() for p in large.parameters())
        assert large_params > small_params
