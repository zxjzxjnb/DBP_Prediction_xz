"""Feature scaling transform."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from dbp_prediction.features.base import BaseTransformer, resolve_columns
from dbp_prediction.features.registry import register_transform


@register_transform("scale")
class ScaleTransformer(BaseTransformer):
    """Apply a fitted standard scaler to selected feature columns."""

    is_feature_scaler = True

    def __init__(self, columns: list[str] | None = None) -> None:
        super().__init__(columns=columns)
        self.columns = None if columns is None else [str(column) for column in columns]
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> "ScaleTransformer":
        del y
        columns = resolve_columns(X, self.columns, "scale.columns")
        self.columns_ = columns
        self.scaler.fit(X[columns])
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        columns = resolve_columns(X, getattr(self, "columns_", self.columns), "scale.columns")
        transformed = X.copy()
        transformed.loc[:, columns] = self.scaler.transform(transformed[columns])
        return transformed, None if y is None else y.copy()
