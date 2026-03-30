"""Ratio feature generation transform."""

from __future__ import annotations

from typing import Any

import pandas as pd

from dbp_prediction.features.base import BaseTransformer, ensure_columns_exist
from dbp_prediction.features.registry import register_transform


def _normalize_specs(specs: list[Any]) -> list[tuple[str, str, str | None]]:
    normalized: list[tuple[str, str, str | None]] = []
    for spec in specs:
        if isinstance(spec, dict):
            left = str(spec.get("left"))
            right = str(spec.get("right"))
            name = spec.get("name")
            normalized.append((left, right, None if name is None else str(name)))
            continue
        if isinstance(spec, (list, tuple)) and len(spec) in {2, 3}:
            left = str(spec[0])
            right = str(spec[1])
            name = None if len(spec) == 2 or spec[2] is None else str(spec[2])
            normalized.append((left, right, name))
            continue
        raise ValueError(
            "'ratio.pairs' entries must be [left, right], [left, right, name], "
            "or a mapping with keys 'left', 'right', and optional 'name'"
        )
    return normalized


@register_transform("ratio")
class RatioTransformer(BaseTransformer):
    """Create ratio features from column pairs."""

    def __init__(self, pairs: list[Any] | None = None) -> None:
        if not pairs:
            raise ValueError("'ratio' requires non-empty 'pairs'")
        super().__init__(pairs=pairs)
        self.pairs = _normalize_specs(list(pairs))

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> "RatioTransformer":
        del y
        validated: list[tuple[str, str, str]] = []
        for left, right, name in self.pairs:
            ensure_columns_exist(X, [left, right], "ratio.pairs")
            output_name = name or f"{left}__div__{right}"
            validated.append((left, right, output_name))
        self.pairs_ = validated
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        transformed = X.copy()
        for left, right, output_name in getattr(self, "pairs_", []):
            if (transformed[right] == 0).any():
                raise ValueError(
                    f"'ratio' denominator column '{right}' contains zero values, "
                    f"cannot create '{output_name}'"
                )
            transformed[output_name] = transformed[left] / transformed[right]
        return transformed, None if y is None else y.copy()
