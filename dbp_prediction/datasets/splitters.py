"""Train/test and train/validation split helpers."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from dbp_prediction.config import SPLIT_COL, TEST_LABEL, TRAIN_LABEL

logger = logging.getLogger(__name__)


def get_train_test_split(
    df: pd.DataFrame,
    split_col: str = SPLIT_COL,
    train_label: str = TRAIN_LABEL,
    test_label: str = TEST_LABEL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into train and test sets using a label column."""
    train_df = df[df[split_col] == train_label].reset_index(drop=True)
    test_df = df[df[split_col] == test_label].reset_index(drop=True)

    if train_df.empty or test_df.empty:
        available = sorted(df[split_col].dropna().astype(str).unique().tolist())
        raise ValueError(
            "Split produced an empty partition. "
            f"split_col={split_col!r}, train_label={train_label!r}, "
            f"test_label={test_label!r}, available_labels={available}"
        )

    logger.info("Train: %d samples, Test: %d samples", len(train_df), len(test_df))
    return train_df, test_df


def get_train_val_split(
    train_df: pd.DataFrame,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Further split a training dataframe into train/validation subsets."""
    train_sub_df, val_df = train_test_split(
        train_df,
        test_size=val_fraction,
        random_state=seed,
    )
    logger.info("Train subset: %d, Validation: %d", len(train_sub_df), len(val_df))
    return train_sub_df, val_df
