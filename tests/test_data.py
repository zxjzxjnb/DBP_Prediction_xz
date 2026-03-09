"""Tests for the data loading and preprocessing module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from dbp_prediction.config import FEATURE_COLS, TARGET_COLS
from dbp_prediction.data import (
    df_to_tensors,
    fit_scalers,
    get_train_test_split,
    get_train_val_split,
    load_dataset,
)


class TestLoadDataset:
    """Tests for load_dataset()."""

    def test_loads_valid_csv(self, sample_csv: Path) -> None:
        df = load_dataset(sample_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 25

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nonexistent.csv")

    def test_raises_on_missing_columns(self, tmp_path: Path) -> None:
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"a": [1]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="Missing columns"):
            load_dataset(bad_csv)


class TestSplitting:
    """Tests for train/test and train/val splitting."""

    def test_train_test_split_sizes(self, sample_dataframe: pd.DataFrame) -> None:
        train, test = get_train_test_split(sample_dataframe)
        assert len(train) == 20
        assert len(test) == 5

    def test_train_val_split(self, sample_dataframe: pd.DataFrame) -> None:
        train, _ = get_train_test_split(sample_dataframe)
        sub, val = get_train_val_split(train, val_fraction=0.2, seed=42)
        assert len(sub) + len(val) == len(train)
        assert len(val) > 0

    def test_val_split_is_reproducible(self, sample_dataframe: pd.DataFrame) -> None:
        train, _ = get_train_test_split(sample_dataframe)
        sub1, val1 = get_train_val_split(train, seed=42)
        sub2, val2 = get_train_val_split(train, seed=42)
        pd.testing.assert_frame_equal(sub1, sub2)
        pd.testing.assert_frame_equal(val1, val2)


class TestScaling:
    """Tests for scaler fitting."""

    def test_scalers_return_correct_types(self, sample_dataframe: pd.DataFrame) -> None:
        train, _ = get_train_test_split(sample_dataframe)
        sx, sy = fit_scalers(train)
        assert hasattr(sx, "transform")
        assert hasattr(sy, "transform")

    def test_scalers_match_feature_count(self, sample_dataframe: pd.DataFrame) -> None:
        train, _ = get_train_test_split(sample_dataframe)
        sx, sy = fit_scalers(train)
        assert sx.n_features_in_ == len(FEATURE_COLS)
        assert sy.n_features_in_ == len(TARGET_COLS)


class TestTensorConversion:
    """Tests for df_to_tensors()."""

    def test_output_shapes(self, sample_dataframe: pd.DataFrame) -> None:
        train, _ = get_train_test_split(sample_dataframe)
        sx, sy = fit_scalers(train)
        X, Y = df_to_tensors(train, sx, sy)
        assert X.shape == (len(train), len(FEATURE_COLS))
        assert Y.shape == (len(train), len(TARGET_COLS))
        assert X.dtype == torch.float32

    def test_no_target_scaler(self, sample_dataframe: pd.DataFrame) -> None:
        train, _ = get_train_test_split(sample_dataframe)
        sx, _ = fit_scalers(train)
        X, Y = df_to_tensors(train, sx)
        assert X is not None
        assert Y is None
