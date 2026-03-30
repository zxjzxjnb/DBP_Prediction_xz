"""Tests for the Phase 5 feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dbp_prediction.features import TRANSFORM_REGISTRY, FeaturePipeline


class TestFeaturePipeline:
    """Tests for built-in feature transforms and target inversion."""

    def test_registry_contains_minimal_phase5_transforms(self) -> None:
        for name in [
            "select_columns",
            "drop_missing",
            "impute",
            "scale",
            "log1p",
            "interaction",
            "polynomial",
            "target_transform",
        ]:
            assert name in TRANSFORM_REGISTRY

    def test_pipeline_applies_feature_steps_in_order(self) -> None:
        X = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0],
                "b": [2.0, 4.0, 8.0],
                "c": [9.0, 10.0, 11.0],
            }
        )
        y = pd.DataFrame({"target": [10.0, 20.0, 30.0]})
        pipeline = FeaturePipeline.from_specs(
            [
                {"name": "select_columns", "params": {"columns": ["a", "b"]}},
                {"name": "impute", "params": {"columns": ["a"], "strategy": "mean"}},
                {"name": "log1p", "params": {"columns": ["a"]}},
                {"name": "interaction", "params": {"columns": ["a", "b"]}},
                {"name": "polynomial", "params": {"columns": ["b"], "degree": 3}},
            ]
        )

        X_out, y_out = pipeline.fit_transform(X, y)

        assert list(X_out.columns) == ["a", "b", "a__x__b", "b__pow_2", "b__pow_3"]
        assert not X_out.isna().any().any()
        assert y_out is not None
        assert list(y_out.columns) == ["target"]

    def test_drop_missing_filters_rows_using_features_and_targets(self) -> None:
        X = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0, 4.0],
                "b": [2.0, 4.0, np.nan, 8.0],
            }
        )
        y = pd.DataFrame({"target": [10.0, 20.0, 30.0, np.nan]})
        pipeline = FeaturePipeline.from_specs(
            [{"name": "drop_missing", "params": {"columns": ["a", "b"]}}]
        )

        X_out, y_out = pipeline.fit_transform(X, y)

        assert list(X_out.index) == [0]
        assert list(y_out.index) == [0]
        assert X_out.iloc[0].to_dict() == {"a": 1.0, "b": 2.0}
        assert y_out.iloc[0].to_dict() == {"target": 10.0}

    def test_target_transform_inverse_restores_original_targets(self) -> None:
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = pd.DataFrame({"target": [10.0, 20.0, 40.0]})
        pipeline = FeaturePipeline.from_specs(
            [{"name": "target_transform", "params": {"method": "log1p"}}]
        )

        _, y_scaled = pipeline.fit_transform(X, y)
        restored = pipeline.inverse_transform_targets(y_scaled)

        assert np.allclose(restored["target"].to_numpy(), y["target"].to_numpy())

    def test_target_transform_inverse_supports_target_subsets(self) -> None:
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = pd.DataFrame(
            {
                "y1": [10.0, 20.0, 30.0],
                "y2": [5.0, 6.0, 7.0],
            }
        )
        pipeline = FeaturePipeline.from_specs(
            [{"name": "target_transform", "params": {"method": "standard_scale"}}]
        )

        _, y_scaled = pipeline.fit_transform(X, y)
        restored = pipeline.inverse_transform_targets(y_scaled[["y1"]])

        assert list(restored.columns) == ["y1"]
        assert np.allclose(restored["y1"].to_numpy(), y["y1"].to_numpy())

    def test_pipeline_flags_scaling_and_target_transforms(self) -> None:
        pipeline = FeaturePipeline.from_specs(
            [
                {"name": "scale", "params": {}},
                {"name": "target_transform", "params": {"method": "standard_scale"}},
            ]
        )

        assert pipeline.has_feature_scaler is True
        assert pipeline.has_target_transformer is True
