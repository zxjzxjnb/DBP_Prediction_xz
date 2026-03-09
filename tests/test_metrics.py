"""Tests for the evaluation metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from dbp_prediction.metrics import compute_metrics, compute_per_target_metrics, macro_average


class TestComputeMetrics:
    """Tests for compute_metrics()."""

    def test_perfect_prediction(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(y, y)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert m["mae"] == pytest.approx(0.0, abs=1e-10)
        assert m["r2"] == pytest.approx(1.0, abs=1e-10)

    def test_known_values(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
        m = compute_metrics(y_true, y_pred)
        assert m["rmse"] > 0
        assert m["mae"] > 0
        assert 0 < m["r2"] < 1

    def test_returns_floats(self) -> None:
        m = compute_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        for v in m.values():
            assert isinstance(v, float)

    def test_constant_prediction(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        m = compute_metrics(y_true, y_pred)
        assert m["rmse"] > 0
        assert m["r2"] == pytest.approx(0.0, abs=1e-10)


class TestPerTargetMetrics:
    """Tests for compute_per_target_metrics()."""

    def test_output_structure(self, sample_arrays: dict) -> None:
        result = compute_per_target_metrics(
            sample_arrays["y_true"],
            sample_arrays["y_pred"],
            ["a", "b", "c"],
        )
        assert set(result.keys()) == {"a", "b", "c"}
        for m in result.values():
            assert "rmse" in m
            assert "mae" in m
            assert "r2" in m

    def test_default_names(self, sample_arrays: dict) -> None:
        result = compute_per_target_metrics(
            sample_arrays["y_true"],
            sample_arrays["y_pred"],
        )
        assert "target_0" in result
        assert "target_2" in result


class TestMacroAverage:
    """Tests for macro_average()."""

    def test_averages_correctly(self) -> None:
        per_target = {
            "a": {"rmse": 2.0, "mae": 1.0, "r2": 0.8},
            "b": {"rmse": 4.0, "mae": 3.0, "r2": 0.6},
        }
        avg = macro_average(per_target)
        assert avg["rmse"] == pytest.approx(3.0)
        assert avg["mae"] == pytest.approx(2.0)
        assert avg["r2"] == pytest.approx(0.7)
