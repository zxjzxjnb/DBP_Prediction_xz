"""Tests for model adapters and registry behavior."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dbp_prediction.models import MODEL_REGISTRY, TrainedModelArtifact, get_model_adapter


class TestModelRegistry:
    """Tests for the Phase 4 model registry contract."""

    def test_registry_contains_mlp_and_kan(self) -> None:
        assert "mlp" in MODEL_REGISTRY
        assert "kan" in MODEL_REGISTRY
        assert "random_forest" in MODEL_REGISTRY
        assert "xgboost" in MODEL_REGISTRY

    def test_adapters_expose_search_spaces_for_phase6_tuning(self) -> None:
        mlp = get_model_adapter("mlp")
        kan = get_model_adapter("kan")
        random_forest = get_model_adapter("random_forest")
        xgboost = get_model_adapter("xgboost")

        assert "model" in mlp.search_space()
        assert "training" in mlp.search_space()
        assert "hidden_dims" in mlp.search_space()["model"]
        assert "model" in kan.search_space()
        assert "training" in kan.search_space()
        assert "grid" in kan.search_space()["model"]
        assert "n_estimators" in random_forest.search_space()["model"]
        assert "learning_rate" in xgboost.search_space()["model"]

    def test_mlp_adapter_supports_fit_predict_save_and_load(self, tmp_path) -> None:
        adapter = get_model_adapter("mlp")
        X_train = torch.randn(12, 3)
        Y_train = torch.randn(12, 1)
        X_val = torch.randn(4, 3)
        Y_val = torch.randn(4, 1)

        artifact = adapter.fit(
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            in_dim=3,
            out_dim=1,
            model_params={"hidden_dims": [8], "dropout": 0.0, "activation": "ReLU"},
            training_params={
                "optimizer": "Adam",
                "loss": "MSE",
                "huber_delta": 1.0,
                "max_grad_norm": 5.0,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 4,
                "max_epochs": 2,
                "patience": 1,
                "val_fraction": 0.2,
            },
            seed=42,
        )

        assert isinstance(artifact, TrainedModelArtifact)

        preds_before = adapter.predict(artifact, X_val)
        saved_path = adapter.save(artifact, tmp_path / "mlp_artifact.pt")
        loaded = adapter.load(saved_path)
        preds_after = adapter.predict(loaded, X_val)

        assert saved_path.exists()
        assert preds_before.shape == (4, 1)
        assert np.allclose(preds_before, preds_after)

    def test_random_forest_adapter_supports_fit_predict_save_and_load(self, tmp_path) -> None:
        adapter = get_model_adapter("random_forest")
        X_train = torch.randn(16, 3)
        Y_train = torch.randn(16, 1)
        X_val = torch.randn(5, 3)
        Y_val = torch.randn(5, 1)

        artifact = adapter.fit(
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            in_dim=3,
            out_dim=1,
            model_params={
                "n_estimators": 20,
                "max_depth": 4,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "bootstrap": True,
            },
            training_params={
                "optimizer": "Adam",
                "loss": "MSE",
                "huber_delta": 1.0,
                "max_grad_norm": 5.0,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 4,
                "max_epochs": 2,
                "patience": 1,
                "val_fraction": 0.2,
            },
            seed=42,
        )

        preds_before = adapter.predict(artifact, X_val)
        saved_path = adapter.save(artifact, tmp_path / "rf_artifact.joblib")
        loaded = adapter.load(saved_path)
        preds_after = adapter.predict(loaded, X_val)

        assert isinstance(artifact, TrainedModelArtifact)
        assert artifact.parameter_count > 0
        assert saved_path.exists()
        assert preds_before.shape == (5, 1)
        assert np.allclose(preds_before, preds_after)

    def test_xgboost_adapter_supports_fit_predict_save_and_load(self, tmp_path) -> None:
        pytest.importorskip("xgboost")
        adapter = get_model_adapter("xgboost")
        X_train = torch.randn(20, 3)
        Y_train = torch.randn(20, 1)
        X_val = torch.randn(6, 3)
        Y_val = torch.randn(6, 1)

        artifact = adapter.fit(
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            in_dim=3,
            out_dim=1,
            model_params={
                "n_estimators": 25,
                "max_depth": 3,
                "learning_rate": 0.1,
                "min_child_weight": 1.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0.0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "early_stopping_rounds": 5,
            },
            training_params={
                "optimizer": "Adam",
                "loss": "MSE",
                "huber_delta": 1.0,
                "max_grad_norm": 5.0,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 4,
                "max_epochs": 2,
                "patience": 1,
                "val_fraction": 0.2,
            },
            seed=42,
        )

        preds_before = adapter.predict(artifact, X_val)
        saved_path = adapter.save(artifact, tmp_path / "xgb_artifact.joblib")
        loaded = adapter.load(saved_path)
        preds_after = adapter.predict(loaded, X_val)

        assert isinstance(artifact, TrainedModelArtifact)
        assert artifact.best_step > 0
        assert saved_path.exists()
        assert preds_before.shape == (6, 1)
        assert np.allclose(preds_before, preds_after)
