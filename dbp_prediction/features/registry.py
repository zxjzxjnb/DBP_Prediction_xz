"""Registry for feature pipeline transforms."""

from __future__ import annotations

from typing import Any

from dbp_prediction.features.base import BaseTransformer

TRANSFORM_REGISTRY: dict[str, type[BaseTransformer]] = {}


def register_transform(name: str):
    """Register a transformer class under a normalized name."""

    def decorator(transformer_cls: type[BaseTransformer]) -> type[BaseTransformer]:
        key = name.strip().lower()
        if key in TRANSFORM_REGISTRY and TRANSFORM_REGISTRY[key] is not transformer_cls:
            raise ValueError(f"Transform '{key}' is already registered")
        transformer_cls.name = key
        TRANSFORM_REGISTRY[key] = transformer_cls
        return transformer_cls

    return decorator


def create_transform(name: str, params: dict[str, Any] | None = None) -> BaseTransformer:
    """Instantiate a registered transform by name."""
    from dbp_prediction.features import transforms as _transforms  # noqa: F401

    key = name.strip().lower()
    transformer_cls = TRANSFORM_REGISTRY.get(key)
    if transformer_cls is None:
        raise ValueError(
            f"Unknown feature transform '{key}'. "
            f"Registered: {sorted(TRANSFORM_REGISTRY)}"
        )
    return transformer_cls(**dict(params or {}))
