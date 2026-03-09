"""Data loading, splitting, and preprocessing utilities.

All functions operate on pandas DataFrames and numpy arrays so that they
remain framework-agnostic until the final tensor conversion step.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dbp_prediction.config import (
    DEFAULT_DATA_PATH,
    FEATURE_COLS,
    SPLIT_COL,
    TARGET_COLS,
    TEST_LABEL,
    TRAIN_LABEL,
)

logger = logging.getLogger(__name__)


# ── Loading ──────────────────────────────────────────────────────────────────


def load_dataset(
    path: Path | None = None,
    feature_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load the CSV dataset and validate that required columns exist.

    Parameters
    ----------
    path : Path, optional
        Path to the CSV file.  Defaults to ``DEFAULT_DATA_PATH``.
    feature_cols : list of str, optional
        Feature column names to validate.  Defaults to ``FEATURE_COLS``.
    target_cols : list of str, optional
        Target column names to validate.  Defaults to ``TARGET_COLS``.

    Returns
    -------
    pd.DataFrame
        The loaded dataframe.

    Raises
    ------
    FileNotFoundError
        If the data file does not exist.
    ValueError
        If required columns are missing.
    """
    path = path or DEFAULT_DATA_PATH
    feature_cols = feature_cols or FEATURE_COLS
    target_cols = target_cols or TARGET_COLS

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    required = set(feature_cols) | set(target_cols) | {SPLIT_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {sorted(missing)}")

    logger.info("Loaded %d rows from %s", len(df), path)
    return df


# ── Splitting ────────────────────────────────────────────────────────────────


def get_train_test_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into train and test sets using the ``split`` column.

    Returns
    -------
    tuple of (train_df, test_df)
    """
    train_df = df[df[SPLIT_COL] == TRAIN_LABEL].reset_index(drop=True)
    test_df = df[df[SPLIT_COL] == TEST_LABEL].reset_index(drop=True)
    logger.info("Train: %d samples, Test: %d samples", len(train_df), len(test_df))
    return train_df, test_df


def get_train_val_split(
    train_df: pd.DataFrame,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Further split the training set into train-subset and validation set.

    Parameters
    ----------
    train_df : pd.DataFrame
        Full training dataframe.
    val_fraction : float
        Fraction to hold out for validation.
    seed : int
        Random state for reproducibility.

    Returns
    -------
    tuple of (train_sub_df, val_df)
    """
    train_sub_df, val_df = train_test_split(
        train_df,
        test_size=val_fraction,
        random_state=seed,
    )
    logger.info("Train subset: %d, Validation: %d", len(train_sub_df), len(val_df))
    return train_sub_df, val_df


# ── Scaling ──────────────────────────────────────────────────────────────────


def fit_scalers(
    train_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
) -> tuple[StandardScaler, StandardScaler]:
    """Fit StandardScalers on training data only.

    Returns
    -------
    tuple of (scaler_x, scaler_y)
    """
    feature_cols = feature_cols or FEATURE_COLS
    target_cols = target_cols or TARGET_COLS

    scaler_x = StandardScaler().fit(train_df[feature_cols])
    scaler_y = StandardScaler().fit(train_df[target_cols])
    return scaler_x, scaler_y


# ── Tensor conversion ───────────────────────────────────────────────────────


def df_to_tensors(
    df: pd.DataFrame,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler | None = None,
    feature_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Scale features (and optionally targets) and convert to float32 tensors.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    scaler_x : StandardScaler
        Pre-fitted feature scaler.
    scaler_y : StandardScaler, optional
        Pre-fitted target scaler.  If ``None``, no Y tensor is returned.
    feature_cols : list of str, optional
        Feature columns.  Defaults to ``FEATURE_COLS``.
    target_cols : list of str, optional
        Target columns.  Defaults to ``TARGET_COLS``.

    Returns
    -------
    tuple of (X_tensor, Y_tensor or None)
    """
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
) -> dict:
    """One-call convenience: load → split → scale → tensorise.

    Returns
    -------
    dict with keys:
        X_train, Y_train, X_val, Y_val, X_test,
        Y_test_raw (numpy), scaler_x, scaler_y,
        train_sub_df, val_df, test_df
    """
    feature_cols = feature_cols or FEATURE_COLS
    target_cols = target_cols or TARGET_COLS

    df = load_dataset(data_path, feature_cols, target_cols)
    train_df, test_df = get_train_test_split(df)
    train_sub_df, val_df = get_train_val_split(train_df, val_fraction, seed)

    scaler_x, scaler_y = fit_scalers(train_sub_df, feature_cols, target_cols)

    X_train, Y_train = df_to_tensors(train_sub_df, scaler_x, scaler_y, feature_cols, target_cols)
    X_val, Y_val = df_to_tensors(val_df, scaler_x, scaler_y, feature_cols, target_cols)
    X_test, _ = df_to_tensors(test_df, scaler_x, feature_cols=feature_cols)

    Y_test_raw = test_df[target_cols].values

    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_val": X_val,
        "Y_val": Y_val,
        "X_test": X_test,
        "Y_test_raw": Y_test_raw,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "train_df": train_df,
        "train_sub_df": train_sub_df,
        "val_df": val_df,
        "test_df": test_df,
    }
