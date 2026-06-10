"""Missing-value imputation transform."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer

from dbp_prediction.features.base import BaseTransformer, resolve_columns
from dbp_prediction.features.registry import register_transform


@register_transform("impute")
class ImputeTransformer(BaseTransformer):
    """Fill missing feature values with a fitted imputer."""

    def __init__(
        self,
        columns: list[str] | None = None,
        strategy: str = "mean",
        fill_value: float | str | None = None,
    ) -> None:
        super().__init__(columns=columns, strategy=strategy, fill_value=fill_value)
        self.columns = None if columns is None else [str(column) for column in columns]
        self.strategy = str(strategy)
        self.fill_value = fill_value
        self.imputer = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> ImputeTransformer:
        del y
        columns = resolve_columns(X, self.columns, "impute.columns")
        self.columns_ = columns
        self.imputer.fit(X[columns])
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        columns = resolve_columns(X, getattr(self, "columns_", self.columns), "impute.columns")
        transformed = X.copy()
        transformed.loc[:, columns] = self.imputer.transform(transformed[columns])
        return transformed, None if y is None else y.copy()
