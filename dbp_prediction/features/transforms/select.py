"""Column selection transform."""

from __future__ import annotations

import pandas as pd

from dbp_prediction.features.base import BaseTransformer, resolve_columns
from dbp_prediction.features.registry import register_transform


@register_transform("select_columns")
class SelectColumnsTransformer(BaseTransformer):
    """Restrict the feature frame to an explicit ordered subset of columns."""

    def __init__(self, columns: list[str]) -> None:
        if not isinstance(columns, list) or not columns:
            raise ValueError("'select_columns.columns' must be a non-empty list")
        super().__init__(columns=columns)
        self.columns = [str(column) for column in columns]

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        columns = resolve_columns(X, self.columns, "select_columns.columns")
        return X.loc[:, columns].copy(), None if y is None else y.copy()
