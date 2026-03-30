"""Shared data-preparation and prediction utilities for engine modules."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from dbp_prediction.datasets import get_train_test_split, load_dataset
from dbp_prediction.features import FeaturePipeline
from dbp_prediction.schemas import DatasetSchema

logger = logging.getLogger(__name__)


def dataset_payload(
    dataset: DatasetSchema | None,
    feature_cols: list[str],
    allowed_targets: list[str],
) -> dict[str, Any]:
    """Build a serializable dataset metadata dict for checkpoint storage."""
    return {
        "path": str(dataset.path) if dataset else None,
        "format": dataset.format if dataset else "csv",
        "feature_cols": feature_cols,
        "target_cols": allowed_targets,
        "split": {
            "strategy": dataset.split.strategy if dataset else "predefined",
            "column": dataset.split.column if dataset else "split",
            "train_label": dataset.split.train_label if dataset else "train",
            "test_label": dataset.split.test_label if dataset else "test",
        },
    }


def scale_frame(frame: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply a fitted scaler to a DataFrame, preserving column names and index."""
    return pd.DataFrame(
        scaler.transform(frame),
        columns=list(frame.columns),
        index=frame.index,
    )


def inverse_predictions(
    pred_scaled: np.ndarray,
    data: dict[str, Any],
    target_name: str,
) -> np.ndarray:
    """Undo target scaling or pipeline transforms on model predictions."""
    pipeline = data.get("feature_pipeline")
    if pipeline is not None and pipeline.has_target_transformer:
        return pipeline.inverse_transform_targets(
            pred_scaled,
            columns=[target_name],
        ).ravel()
    if data["scaler_y"] is not None:
        return data["scaler_y"].inverse_transform(pred_scaled).ravel()
    return pred_scaled.ravel()


def _ensure_non_empty_split(frame: pd.DataFrame, split_name: str) -> None:
    """Raise a clear error when preprocessing removes every row from a split."""
    if frame.empty:
        raise ValueError(
            f"No rows remain in the '{split_name}' split after feature pipeline preprocessing. "
            "Check drop_missing/select_columns settings or the dataset coverage for this target."
        )


def prepare_pipeline_data(
    feature_cols: list[str],
    target_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_steps: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Prepare pipeline-transformed and scaled tensors for a single target.

    This is the shared core used by both the training runner and the tuner.  It
    handles optional feature pipeline execution, fallback StandardScaler for
    features that have no pipeline scaler, and fallback StandardScaler for
    targets that have no pipeline target transformer.
    """
    pipeline = FeaturePipeline.from_specs(feature_steps)

    if feature_steps:
        train_X_df, train_y_df = pipeline.fit_transform(
            train_df[feature_cols],
            train_df[[target_name]],
        )
        val_X_df, val_y_df = pipeline.transform(
            val_df[feature_cols],
            val_df[[target_name]],
        )
        test_X_df, _ = pipeline.transform(
            test_df[feature_cols],
            test_df[[target_name]],
        )
        train_frame = train_df.loc[train_X_df.index].copy()
        val_frame = val_df.loc[val_X_df.index].copy()
        test_frame = test_df.loc[test_X_df.index].copy()
    else:
        pipeline = None
        train_X_df = train_df[feature_cols].copy()
        train_y_df = train_df[[target_name]].copy()
        val_X_df = val_df[feature_cols].copy()
        val_y_df = val_df[[target_name]].copy()
        test_X_df = test_df[feature_cols].copy()
        train_frame = train_df.copy()
        val_frame = val_df.copy()
        test_frame = test_df.copy()

    _ensure_non_empty_split(train_X_df, "train")
    _ensure_non_empty_split(val_X_df, "validation")
    _ensure_non_empty_split(test_X_df, "test")

    scaler_x = None
    if pipeline is None or not pipeline.has_feature_scaler:
        scaler_x = StandardScaler().fit(train_X_df)
        train_X_df = scale_frame(train_X_df, scaler_x)
        val_X_df = scale_frame(val_X_df, scaler_x)
        test_X_df = scale_frame(test_X_df, scaler_x)

    scaler_y = None
    if pipeline is None or not pipeline.has_target_transformer:
        scaler_y = StandardScaler().fit(train_y_df)
        train_y_df = scale_frame(train_y_df, scaler_y)
        val_y_df = scale_frame(val_y_df, scaler_y)

    return {
        "X_train": torch.tensor(train_X_df.to_numpy(), dtype=torch.float32),
        "Y_train": torch.tensor(train_y_df.to_numpy(), dtype=torch.float32),
        "X_val": torch.tensor(val_X_df.to_numpy(), dtype=torch.float32),
        "Y_val": torch.tensor(val_y_df.to_numpy(), dtype=torch.float32),
        "X_test": torch.tensor(test_X_df.to_numpy(), dtype=torch.float32),
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "feature_pipeline": pipeline,
        "feature_cols_processed": list(train_X_df.columns),
        "Y_val_raw": val_frame[[target_name]].to_numpy(),
        "Y_test_raw": test_frame[[target_name]].to_numpy(),
        "train_frame": train_frame,
        "val_frame": val_frame,
        "test_frame": test_frame,
    }


def load_train_test_frames(
    dataset: DatasetSchema | None,
    feature_cols: list[str],
    target_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a dataset and split it into train/test DataFrames."""
    df = load_dataset(
        path=dataset.path if dataset else None,
        feature_cols=feature_cols,
        target_cols=target_cols,
        split_col=dataset.split.column if dataset else "split",
        file_format=dataset.format if dataset else None,
        read_options=dataset.reader_options if dataset else None,
    )
    train_df, test_df = get_train_test_split(
        df,
        split_col=dataset.split.column if dataset else "split",
        train_label=dataset.split.train_label if dataset else "train",
        test_label=dataset.split.test_label if dataset else "test",
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
