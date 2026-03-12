"""Base classes and helpers for feature/target transformations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


# ── Shared math helpers ─────────────────────────────────────────────────────


def signed_log1p(values: pd.DataFrame) -> pd.DataFrame:
    """Element-wise signed log1p: sign(x) * log1p(|x|)."""
    return np.sign(values) * np.log1p(np.abs(values))


def signed_expm1(values: pd.DataFrame) -> pd.DataFrame:
    """Element-wise signed expm1: sign(x) * expm1(|x|).  Inverse of signed_log1p."""
    return np.sign(values) * np.expm1(np.abs(values))


def ensure_columns_exist(df: pd.DataFrame, columns: list[str], field_name: str) -> list[str]:
    """Validate that requested columns exist in a dataframe."""
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{field_name} references missing columns: {missing}")
    return columns


def resolve_columns(
    df: pd.DataFrame,
    columns: list[str] | None,
    field_name: str,
) -> list[str]:
    """Return selected columns or all dataframe columns when omitted."""
    selected = list(df.columns) if columns is None else [str(column) for column in columns]
    if not selected:
        raise ValueError(f"{field_name} must not be empty")
    return ensure_columns_exist(df, selected, field_name)


class BaseTransformer(ABC):
    """Base interface for sequential feature/target transformations."""

    name: str = ""
    is_feature_scaler: bool = False
    is_target_transformer: bool = False

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> "BaseTransformer":
        """Learn parameters from train data."""
        del X, y
        return self

    @abstractmethod
    def transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Apply the transform to feature and optional target frames."""

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Fit the transform and apply it immediately."""
        return self.fit(X, y).transform(X, y)

    def inverse_transform_targets(self, y: pd.DataFrame) -> pd.DataFrame:
        """Undo any target-space transform. No-op for feature-only steps."""
        return y

    def to_config(self) -> dict[str, Any]:
        """Serialize the transform back to a config-friendly structure."""
        return {"name": self.name, "params": dict(self.params)}
