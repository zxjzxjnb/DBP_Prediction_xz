"""Interaction feature generation transform."""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from dbp_prediction.features.base import BaseTransformer, ensure_columns_exist, resolve_columns
from dbp_prediction.features.registry import register_transform


def _normalize_pairs(pairs: list[list[str]] | list[tuple[str, str]]) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("'interaction.pairs' entries must contain exactly two columns")
        normalized.append((str(pair[0]), str(pair[1])))
    return normalized


@register_transform("interaction")
class InteractionTransformer(BaseTransformer):
    """Create pairwise multiplicative interaction features."""

    def __init__(
        self,
        columns: list[str] | None = None,
        pairs: list[list[str]] | list[tuple[str, str]] | None = None,
    ) -> None:
        if columns is None and pairs is None:
            raise ValueError("'interaction' requires either 'columns' or 'pairs'")
        super().__init__(columns=columns, pairs=pairs)
        self.columns = None if columns is None else [str(column) for column in columns]
        self.pairs = None if pairs is None else _normalize_pairs(pairs)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> "InteractionTransformer":
        del y
        if self.pairs is not None:
            pairs = []
            for left, right in self.pairs:
                ensure_columns_exist(X, [left, right], "interaction.pairs")
                pairs.append((left, right))
            self.pairs_ = pairs
        else:
            columns = resolve_columns(X, self.columns, "interaction.columns")
            self.pairs_ = list(combinations(columns, 2))
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        transformed = X.copy()
        for left, right in getattr(self, "pairs_", []):
            transformed[f"{left}__x__{right}"] = transformed[left] * transformed[right]
        return transformed, None if y is None else y.copy()
