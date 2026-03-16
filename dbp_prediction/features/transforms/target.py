"""Target-only transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from dbp_prediction.features.base import (
    BaseTransformer,
    resolve_columns,
    signed_expm1,
    signed_log1p,
)
from dbp_prediction.features.registry import register_transform





@register_transform("target_transform")
class TargetTransformTransformer(BaseTransformer):
    """Transform targets while keeping inverse-transform support."""

    is_target_transformer = True

    def __init__(
        self,
        method: str = "standard_scale",
        columns: list[str] | None = None,
    ) -> None:
        normalized_method = str(method).strip().lower()
        if normalized_method not in {"standard_scale", "log1p"}:
            raise ValueError(
                "'target_transform.method' must be one of ['standard_scale', 'log1p']"
            )
        super().__init__(method=normalized_method, columns=columns)
        self.method = normalized_method
        self.columns = None if columns is None else [str(column) for column in columns]
        self.scaler = StandardScaler() if self.method == "standard_scale" else None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
    ) -> "TargetTransformTransformer":
        del X
        if y is None:
            raise ValueError("'target_transform' requires target columns")
        self.columns_ = resolve_columns(y, self.columns, "target_transform.columns")
        if self.scaler is not None:
            self.scaler.fit(y[self.columns_])
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        if y is None:
            return X.copy(), None
        columns = resolve_columns(y, getattr(self, "columns_", self.columns), "target_transform.columns")
        transformed_y = y.copy()
        if self.method == "standard_scale":
            transformed_y.loc[:, columns] = self.scaler.transform(transformed_y[columns])
        else:
            transformed_y.loc[:, columns] = signed_log1p(transformed_y[columns])
        return X.copy(), transformed_y

    def _available_columns(self, y: pd.DataFrame) -> list[str]:
        configured = getattr(self, "columns_", self.columns)
        if configured is None:
            return list(y.columns)
        return [column for column in configured if column in y.columns]

    def inverse_transform_targets(self, y: pd.DataFrame) -> pd.DataFrame:
        columns = self._available_columns(y)
        if not columns:
            return y.copy()

        restored = y.copy()
        if self.method == "standard_scale":
            fitted_columns = list(getattr(self, "columns_", self.columns) or columns)
            column_index = {column: index for index, column in enumerate(fitted_columns)}
            scale = getattr(self.scaler, "scale_", None)
            mean = getattr(self.scaler, "mean_", None)

            # StandardScaler is column-wise, so we can safely invert only the
            # target subset present in the current frame.
            for column in columns:
                values = restored[column].to_numpy(copy=True)
                index = column_index[column]
                if scale is not None:
                    values = values * scale[index]
                if mean is not None:
                    values = values + mean[index]
                restored.loc[:, column] = values
        else:
            restored.loc[:, columns] = signed_expm1(restored[columns])
        return restored
