"""Tests for training utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from dbp_prediction.models.mlp import MLP
from dbp_prediction.training import make_loss_fn, make_optimizer, set_seed, train_model


class TestSetSeed:
    """Tests for set_seed()."""

    def test_reproducibility(self) -> None:
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)


class TestFactories:
    """Tests for optimizer and loss factories."""

    def test_make_optimizer_adam(self) -> None:
        m = nn.Linear(4, 2)
        opt = make_optimizer(m, "Adam", lr=1e-3)
        assert isinstance(opt, torch.optim.Adam)

    def test_make_optimizer_adamw(self) -> None:
        m = nn.Linear(4, 2)
        opt = make_optimizer(m, "AdamW", lr=1e-3)
        assert isinstance(opt, torch.optim.AdamW)

    def test_make_loss_mse(self) -> None:
        loss = make_loss_fn("MSE")
        assert isinstance(loss, nn.MSELoss)

    def test_make_loss_huber(self) -> None:
        loss = make_loss_fn("Huber", huber_delta=2.0)
        assert isinstance(loss, nn.SmoothL1Loss)


class TestTrainModel:
    """Integration tests for the training loop."""

    def test_loss_decreases(self) -> None:
        set_seed(42)
        model = MLP(in_dim=4, out_dim=1, hidden_dims=[8])

        X = torch.randn(30, 4)
        Y = torch.randn(30, 1)

        # Compute initial loss
        model.eval()
        with torch.no_grad():
            initial_loss = nn.MSELoss()(model(X), Y).item()

        # Train for a few epochs
        model, best_val, best_step, _ = train_model(
            model=model,
            X_train=X[:20],
            Y_train=Y[:20],
            X_val=X[20:],
            Y_val=Y[20:],
            max_epochs=50,
            patience=50,
            batch_size=10,
            verbose_every=0,
        )

        # Loss should have decreased
        model.eval()
        with torch.no_grad():
            final_loss = nn.MSELoss()(model(X[:20]), Y[:20]).item()
        assert final_loss < initial_loss

    def test_early_stopping_triggers(self) -> None:
        set_seed(42)
        model = MLP(in_dim=4, out_dim=1, hidden_dims=[8])

        X = torch.randn(20, 4)
        Y = torch.randn(20, 1)

        _, _, best_step, _ = train_model(
            model=model,
            X_train=X[:15],
            Y_train=Y[:15],
            X_val=X[15:],
            Y_val=Y[15:],
            max_epochs=5000,
            patience=10,
            batch_size=8,
            verbose_every=0,
        )
        # Should stop well before max_epochs
        assert best_step < 5000

    def test_returns_numpy_predictions(self) -> None:
        set_seed(42)
        model = MLP(in_dim=4, out_dim=2, hidden_dims=[8])

        X = torch.randn(20, 4)
        Y = torch.randn(20, 2)

        _, _, _, preds = train_model(
            model=model,
            X_train=X[:15],
            Y_train=Y[:15],
            X_val=X[15:],
            Y_val=Y[15:],
            max_epochs=10,
            patience=10,
            batch_size=8,
            verbose_every=0,
        )
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (5, 2)
