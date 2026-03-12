"""Model definitions, adapters, and registry exports."""

from dbp_prediction.models.base import (
    MODEL_REGISTRY,
    ModelAdapter,
    TorchModelAdapter,
    TrainedModelArtifact,
    get_model_adapter,
)
from dbp_prediction.models.kan import KANAdapter, build_kan, build_kan_from_params
from dbp_prediction.models.mlp import MLP, MLPAdapter, build_mlp
from dbp_prediction.models.tree import RandomForestAdapter, XGBoostAdapter

__all__ = [
    "KANAdapter",
    "MLP",
    "MLPAdapter",
    "MODEL_REGISTRY",
    "ModelAdapter",
    "TorchModelAdapter",
    "TrainedModelArtifact",
    "build_kan",
    "build_kan_from_params",
    "build_mlp",
    "get_model_adapter",
    "RandomForestAdapter",
    "XGBoostAdapter",
]
