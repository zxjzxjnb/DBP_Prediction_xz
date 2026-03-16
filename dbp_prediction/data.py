"""Legacy compatibility exports for dataset loading and preprocessing.

The canonical Phase 3 implementation lives under ``dbp_prediction.datasets``.
This module remains as a thin re-export layer so existing training scripts and
tests keep working while the project migrates to the new package structure.
"""

from __future__ import annotations

from dbp_prediction.datasets import (
    READERS,
    df_to_tensors,
    fit_scalers,
    get_train_test_split,
    get_train_val_split,
    load_dataset,
    prepare_data,
)

__all__ = [
    "READERS",
    "df_to_tensors",
    "fit_scalers",
    "get_train_test_split",
    "get_train_val_split",
    "load_dataset",
    "prepare_data",
]
