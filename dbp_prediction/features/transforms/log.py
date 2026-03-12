"""Log-style transforms for features."""

from __future__ import annotations

import pandas as pd

from dbp_prediction.features.base import BaseTransformer, resolve_columns, signed_log1p
from dbp_prediction.features.registry import register_transform


@register_transform("log1p")
class Log1PTransformer(BaseTransformer):
    """Apply signed log1p to selected feature columns."""

    def __init__(self, columns: list[str] | None = None) -> None:
        super().__init__(columns=columns)
        self.columns = None if columns is None else [str(column) for column in columns]

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        columns = resolve_columns(X, self.columns, "log1p.columns")
        transformed = X.copy()
        transformed.loc[:, columns] = signed_log1p(transformed[columns])
        return transformed, None if y is None else y.copy()
