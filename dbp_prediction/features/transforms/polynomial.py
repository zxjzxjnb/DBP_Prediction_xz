"""Polynomial feature generation transform."""

from __future__ import annotations

import pandas as pd

from dbp_prediction.features.base import BaseTransformer, resolve_columns
from dbp_prediction.features.registry import register_transform


@register_transform("polynomial")
class PolynomialTransformer(BaseTransformer):
    """Add power terms for selected feature columns."""

    def __init__(self, columns: list[str] | None = None, degree: int = 2) -> None:
        if int(degree) < 2:
            raise ValueError("'polynomial.degree' must be >= 2")
        super().__init__(columns=columns, degree=int(degree))
        self.columns = None if columns is None else [str(column) for column in columns]
        self.degree = int(degree)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> "PolynomialTransformer":
        del y
        self.columns_ = resolve_columns(X, self.columns, "polynomial.columns")
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        transformed = X.copy()
        for column in getattr(self, "columns_", resolve_columns(X, self.columns, "polynomial.columns")):
            for power in range(2, self.degree + 1):
                transformed[f"{column}__pow_{power}"] = transformed[column] ** power
        return transformed, None if y is None else y.copy()
