"""Scaling, tensor conversion, and end-to-end data preparation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from sklearn.preprocessing import StandardScaler

from dbp_prediction.config import FEATURE_COLS, SPLIT_COL, TARGET_COLS, TEST_LABEL, TRAIN_LABEL
from dbp_prediction.datasets.loaders import load_dataset
from dbp_prediction.datasets.splitters import get_train_test_split, get_train_val_split


def fit_scalers(
    train_df,
    feature_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
) -> tuple[StandardScaler, StandardScaler]:
    """Fit feature and target scalers on training data only."""
    feature_cols = feature_cols or FEATURE_COLS
    target_cols = target_cols or TARGET_COLS

    scaler_x = StandardScaler().fit(train_df[feature_cols])
    scaler_y = StandardScaler().fit(train_df[target_cols])
    return scaler_x, scaler_y


def df_to_tensors(
    df,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler | None = None,
    feature_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Scale features and optional targets and convert them to float32 tensors."""
    feature_cols = feature_cols or FEATURE_COLS
    target_cols = target_cols or TARGET_COLS

    X = torch.tensor(
        scaler_x.transform(df[feature_cols]),
        dtype=torch.float32,
    )

    Y = None
    if scaler_y is not None:
        Y = torch.tensor(
            scaler_y.transform(df[target_cols]),
            dtype=torch.float32,
        )

    return X, Y


def prepare_data(
    data_path: Path | None = None,
    val_fraction: float = 0.15,
    seed: int = 42,
    feature_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
    split_col: str = SPLIT_COL,
    train_label: str = TRAIN_LABEL,
    test_label: str = TEST_LABEL,
    file_format: str | None = None,
    read_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load, split, scale, and tensorize a dataset for model training."""
    feature_cols = feature_cols or FEATURE_COLS
    target_cols = target_cols or TARGET_COLS

    df = load_dataset(
        path=data_path,
        feature_cols=feature_cols,
        target_cols=target_cols,
        split_col=split_col,
        file_format=file_format,
        read_options=read_options,
    )
    train_df, test_df = get_train_test_split(
        df,
        split_col=split_col,
        train_label=train_label,
        test_label=test_label,
    )
    train_sub_df, val_df = get_train_val_split(train_df, val_fraction, seed)

    scaler_x, scaler_y = fit_scalers(train_sub_df, feature_cols, target_cols)

    X_train, Y_train = df_to_tensors(train_sub_df, scaler_x, scaler_y, feature_cols, target_cols)
    X_val, Y_val = df_to_tensors(val_df, scaler_x, scaler_y, feature_cols, target_cols)
    X_test, _ = df_to_tensors(test_df, scaler_x, feature_cols=feature_cols)

    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_val": X_val,
        "Y_val": Y_val,
        "X_test": X_test,
        "Y_test_raw": test_df[target_cols].values,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "train_df": train_df,
        "train_sub_df": train_sub_df,
        "val_df": val_df,
        "test_df": test_df,
    }
