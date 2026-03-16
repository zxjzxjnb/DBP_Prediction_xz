"""Tests for the structured datasets package introduced in Phase 3."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from dbp_prediction.datasets import (
    READERS,
    build_predictions_frame,
    df_to_tensors,
    export_predictions,
    fit_scalers,
    get_train_test_split,
    get_train_val_split,
    load_dataset,
    prepare_data,
)


class TestDatasetsPackage:
    """Tests for the new datasets package exports."""

    def test_load_dataset_via_package_export(self, sample_csv: Path) -> None:
        df = load_dataset(sample_csv)

        assert len(df) == 25
        assert "csv" in READERS

    def test_split_and_preprocess_via_package_exports(self, sample_dataframe) -> None:
        train_df, _ = get_train_test_split(sample_dataframe)
        train_sub_df, val_df = get_train_val_split(train_df, val_fraction=0.2, seed=42)
        scaler_x, scaler_y = fit_scalers(train_sub_df)
        X_val, Y_val = df_to_tensors(val_df, scaler_x, scaler_y)

        assert len(train_sub_df) + len(val_df) == len(train_df)
        assert X_val.dtype == torch.float32
        assert Y_val is not None

    def test_prepare_data_via_package_export(self, sample_csv: Path) -> None:
        prepared = prepare_data(
            data_path=sample_csv,
            val_fraction=0.2,
            seed=42,
        )

        assert prepared["X_train"].ndim == 2
        assert prepared["Y_train"].ndim == 2
        assert prepared["X_test"].ndim == 2
        assert prepared["Y_test_raw"].shape[0] == len(prepared["test_df"])

    def test_load_dataset_reports_missing_optional_reader_dependency(
        self,
        sample_csv: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise_import_error(*args, **kwargs):
            raise ImportError("missing optional dependency")

        monkeypatch.setitem(READERS, "excel", _raise_import_error)

        with pytest.raises(ModuleNotFoundError, match="openpyxl"):
            load_dataset(sample_csv, file_format="excel")

    def test_export_predictions_writes_wide_csv(self, tmp_path: Path) -> None:
        predictions = {
            "T_THMs_ug_L": {"y_true": [1.0, 2.0], "y_pred": [1.1, 1.9]},
            "DBCM_ug_L": {"y_true": [3.0, 4.0], "y_pred": [2.8, 4.2]},
        }

        frame = build_predictions_frame(predictions)
        output_path = export_predictions(tmp_path / "predictions.csv", predictions)
        written = pd.read_csv(output_path)

        assert list(frame.columns) == [
            "row_id",
            "T_THMs_ug_L__actual",
            "T_THMs_ug_L__prediction",
            "DBCM_ug_L__actual",
            "DBCM_ug_L__prediction",
        ]
        assert written.shape == (2, 5)
