"""Row filtering transform for missing feature/target values."""

from __future__ import annotations

import pandas as pd

from dbp_prediction.features.base import BaseTransformer, resolve_columns
from dbp_prediction.features.registry import register_transform


@register_transform("drop_missing")
class DropMissingTransformer(BaseTransformer):
    """Drop rows with missing values in selected feature/target columns."""

    def __init__(
        self,
        columns: list[str] | None = None,
        target_columns: list[str] | None = None,
    ) -> None:
        super().__init__(columns=columns, target_columns=target_columns)
        self.columns = None if columns is None else [str(column) for column in columns]
        self.target_columns = (
            None if target_columns is None else [str(column) for column in target_columns]
        )

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        feature_columns = resolve_columns(X, self.columns, "drop_missing.columns")
        keep_mask = ~X[feature_columns].isna().any(axis=1)

        if y is not None:
            target_columns = resolve_columns(
                y,
                self.target_columns,
                "drop_missing.target_columns",
            )
            keep_mask = keep_mask & ~y[target_columns].isna().any(axis=1)
            return X.loc[keep_mask].copy(), y.loc[keep_mask].copy()

        return X.loc[keep_mask].copy(), None
