"""Composable feature engineering pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dbp_prediction.features.base import BaseTransformer
from dbp_prediction.features.registry import create_transform


class FeaturePipeline:
    """Sequentially fit and apply feature/target transforms."""

    def __init__(self, steps: list[BaseTransformer] | None = None) -> None:
        self.steps = list(steps or [])
        self.feature_columns_: list[str] = []
        self.target_columns_: list[str] = []
        self._is_fitted = False

    @classmethod
    def from_specs(cls, steps: list[Any] | None) -> FeaturePipeline:
        """Build a pipeline from schema step objects or plain mappings."""
        transformers: list[BaseTransformer] = []
        for step in steps or []:
            if hasattr(step, "name") and hasattr(step, "params"):
                name = step.name
                params = step.params
            elif isinstance(step, dict):
                name = step.get("name")
                params = step.get("params", {})
            else:
                raise ValueError("Feature pipeline steps must be config objects or mappings")
            transformers.append(create_transform(str(name), dict(params or {})))
        return cls(transformers)

    @property
    def has_feature_scaler(self) -> bool:
        return any(step.is_feature_scaler for step in self.steps)

    @property
    def has_target_transformer(self) -> bool:
        return any(step.is_target_transformer for step in self.steps)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
    ) -> FeaturePipeline:
        """Fit all transforms on train data in sequence."""
        current_X = X.copy()
        current_y = None if y is None else y.copy()
        for step in self.steps:
            step.fit(current_X, current_y)
            current_X, current_y = step.transform(current_X, current_y)
        self.feature_columns_ = list(current_X.columns)
        self.target_columns_ = [] if current_y is None else list(current_y.columns)
        self._is_fitted = True
        self._last_fit_result = (current_X, current_y)
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Apply fitted transforms in sequence."""
        current_X = X.copy()
        current_y = None if y is None else y.copy()
        for step in self.steps:
            current_X, current_y = step.transform(current_X, current_y)
        return current_X, current_y

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Fit the pipeline and return the transformed data without recomputing."""
        self.fit(X, y)
        return self._last_fit_result

    def inverse_transform_targets(
        self,
        y: pd.DataFrame | np.ndarray,
        columns: list[str] | None = None,
    ) -> pd.DataFrame | np.ndarray:
        """Undo target transforms in reverse order."""
        is_array = isinstance(y, np.ndarray)
        target_columns = list(columns or self.target_columns_)
        target_frame = (
            pd.DataFrame(y, columns=target_columns)
            if is_array
            else y.copy()
        )
        for step in reversed(self.steps):
            target_frame = step.inverse_transform_targets(target_frame)
        return target_frame.to_numpy() if is_array else target_frame

    def to_config(self) -> list[dict[str, Any]]:
        """Serialize the pipeline steps back to config specs."""
        return [step.to_config() for step in self.steps]
