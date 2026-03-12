"""Model adapter interfaces and registry utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from dbp_prediction.training import train_model


@dataclass
class TrainedModelArtifact:
    """Serialized state for a trained model instance."""

    family: str
    in_dim: int
    out_dim: int
    model_params: dict[str, Any]
    training_params: dict[str, Any]
    seed: int
    model_state: dict[str, Any]
    best_val: float
    best_epoch: int
    parameter_count: int


class ModelAdapter(ABC):
    """Unified lifecycle interface for trainable models."""

    name: str = ""

    @abstractmethod
    def fit(
        self,
        *,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        in_dim: int,
        out_dim: int,
        model_params: dict[str, Any],
        training_params: dict[str, Any],
        seed: int,
    ) -> TrainedModelArtifact:
        """Train a model and return the captured artifact."""

    @abstractmethod
    def predict(
        self,
        artifact: TrainedModelArtifact,
        X: torch.Tensor,
    ) -> np.ndarray:
        """Run inference from a trained artifact."""

    @abstractmethod
    def save(
        self,
        artifact: TrainedModelArtifact,
        path: str | Path,
    ) -> Path:
        """Persist a trained artifact to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> TrainedModelArtifact:
        """Load a trained artifact from disk."""

    def search_space(self) -> dict[str, Any]:
        """Return an adapter-owned hyperparameter search space description."""
        return {}


class TorchModelAdapter(ModelAdapter):
    """Shared Torch fit/predict/save/load behavior for current models."""

    @abstractmethod
    def build_model(
        self,
        *,
        in_dim: int,
        out_dim: int,
        model_params: dict[str, Any],
        seed: int,
    ) -> nn.Module:
        """Construct a fresh model instance."""

    def fit(
        self,
        *,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        in_dim: int,
        out_dim: int,
        model_params: dict[str, Any],
        training_params: dict[str, Any],
        seed: int,
    ) -> TrainedModelArtifact:
        model = self.build_model(
            in_dim=in_dim,
            out_dim=out_dim,
            model_params=model_params,
            seed=seed,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        model, best_val, best_epoch, _ = train_model(
            model=model,
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            optimizer_name=str(training_params.get("optimizer", "Adam")),
            lr=float(training_params.get("lr", 1e-3)),
            weight_decay=float(training_params.get("weight_decay", 1e-4)),
            loss_name=str(training_params.get("loss", "MSE")),
            huber_delta=float(training_params.get("huber_delta", 1.0)),
            batch_size=int(training_params.get("batch_size", 16)),
            max_epochs=int(training_params.get("max_epochs", 2000)),
            patience=int(training_params.get("patience", 100)),
            max_grad_norm=float(training_params.get("max_grad_norm", 5.0)),
        )
        return TrainedModelArtifact(
            family=self.name,
            in_dim=in_dim,
            out_dim=out_dim,
            model_params=dict(model_params),
            training_params=dict(training_params),
            seed=seed,
            model_state={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            best_val=float(best_val),
            best_epoch=int(best_epoch),
            parameter_count=parameter_count,
        )

    def _restore_model(self, artifact: TrainedModelArtifact) -> nn.Module:
        model = self.build_model(
            in_dim=artifact.in_dim,
            out_dim=artifact.out_dim,
            model_params=artifact.model_params,
            seed=artifact.seed,
        )
        model.load_state_dict(artifact.model_state)
        model.eval()
        return model

    def predict(
        self,
        artifact: TrainedModelArtifact,
        X: torch.Tensor,
    ) -> np.ndarray:
        model = self._restore_model(artifact)
        with torch.no_grad():
            return model(X).detach().cpu().numpy()

    def save(
        self,
        artifact: TrainedModelArtifact,
        path: str | Path,
    ) -> Path:
        resolved_path = Path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "family": artifact.family,
                "in_dim": artifact.in_dim,
                "out_dim": artifact.out_dim,
                "model_params": artifact.model_params,
                "training_params": artifact.training_params,
                "seed": artifact.seed,
                "model_state": artifact.model_state,
                "best_val": artifact.best_val,
                "best_epoch": artifact.best_epoch,
                "parameter_count": artifact.parameter_count,
            },
            resolved_path,
        )
        return resolved_path

    def load(self, path: str | Path) -> TrainedModelArtifact:
        # weights_only=False is required because adapter checkpoints may contain
        # sklearn objects (e.g. StandardScaler).  Only load checkpoints from
        # trusted sources.
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        return TrainedModelArtifact(
            family=str(payload["family"]),
            in_dim=int(payload["in_dim"]),
            out_dim=int(payload["out_dim"]),
            model_params=dict(payload["model_params"]),
            training_params=dict(payload["training_params"]),
            seed=int(payload["seed"]),
            model_state=dict(payload["model_state"]),
            best_val=float(payload["best_val"]),
            best_epoch=int(payload["best_epoch"]),
            parameter_count=int(payload["parameter_count"]),
        )


MODEL_REGISTRY: dict[str, type[ModelAdapter]] = {}


def register_model_adapter(name: str):
    """Register a model adapter class under a normalized name."""

    def decorator(adapter_cls: type[ModelAdapter]) -> type[ModelAdapter]:
        key = name.strip().lower()
        if key in MODEL_REGISTRY and MODEL_REGISTRY[key] is not adapter_cls:
            raise ValueError(f"Model adapter '{key}' is already registered")
        adapter_cls.name = key
        MODEL_REGISTRY[key] = adapter_cls
        return adapter_cls

    return decorator


def get_model_adapter(name: str) -> ModelAdapter:
    """Instantiate a registered model adapter by name."""
    key = name.strip().lower()
    adapter_cls = MODEL_REGISTRY.get(key)
    if adapter_cls is None:
        raise ValueError(
            f"Unknown model adapter '{key}'. "
            f"Registered: {sorted(MODEL_REGISTRY)}"
        )
    return adapter_cls()
