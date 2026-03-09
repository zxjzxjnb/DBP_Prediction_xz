"""Shared test fixtures for the DBP prediction test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dbp_prediction.config import FEATURE_COLS, SPLIT_COL, TARGET_COLS


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a small synthetic dataframe mimicking the real dataset."""
    rng = np.random.RandomState(42)
    n_train, n_test = 20, 5
    n = n_train + n_test

    data = {col: rng.randn(n).astype(np.float32) for col in FEATURE_COLS}
    for col in TARGET_COLS:
        data[col] = rng.rand(n).astype(np.float32) * 100
    data[SPLIT_COL] = ["train"] * n_train + ["test"] * n_test

    return pd.DataFrame(data)


@pytest.fixture
def sample_csv(tmp_path: Path, sample_dataframe: pd.DataFrame) -> Path:
    """Write the sample dataframe to a temporary CSV and return its path."""
    csv_path = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_arrays() -> dict:
    """Create small numpy arrays for metric testing."""
    rng = np.random.RandomState(42)
    n, targets = 10, 3
    y_true = rng.rand(n, targets) * 100
    y_pred = y_true + rng.randn(n, targets) * 5
    return {"y_true": y_true, "y_pred": y_pred, "n": n, "targets": targets}
