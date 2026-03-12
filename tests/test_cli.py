"""Tests for CLI helper logic that formats reports and resolves checkpoint shapes."""

from __future__ import annotations

import pytest

from dbp_prediction.cli import generate_report


class TestGenerateReportHelpers:
    """Tests for report checkpoint parsing and metadata rendering."""

    def test_extract_supports_per_target_checkpoint(self) -> None:
        ckpt = {
            "target_payloads": {
                "foo": {"test_metrics": {"rmse": 1.0, "mae": 0.5, "r2": 0.9}},
            },
        }

        assert generate_report._extract(ckpt) == {
            "foo": {"rmse": 1.0, "mae": 0.5, "r2": 0.9},
        }

    def test_extract_supports_multi_output_checkpoint(self) -> None:
        ckpt = {
            "model_family": "kan",
            "paradigm": "multi_output",
            "test_metrics": {
                "foo": {"rmse": 1.0, "mae": 0.5, "r2": 0.9},
            },
        }

        assert generate_report._extract(ckpt) == {
            "foo": {"rmse": 1.0, "mae": 0.5, "r2": 0.9},
        }

    def test_extract_raises_on_unknown_checkpoint_shape(self) -> None:
        with pytest.raises(ValueError, match="Cannot extract per-target metrics"):
            generate_report._extract({"unexpected": True})

    def test_infer_label_uses_kan_paradigm(self) -> None:
        assert generate_report._infer_label(
            {"model_family": "kan", "paradigm": "multi_output"},
            "fallback",
        ) == "KAN (Multi-output)"

    def test_infer_label_supports_per_target_baseline(self) -> None:
        assert generate_report._infer_label(
            {"model_family": "mlp", "paradigm": "per_target_baseline"},
            "fallback",
        ) == "MLP (Per-target)"

    def test_render_uses_separate_protocol_lines_when_paradigms_differ(self) -> None:
        baseline_ckpt = {
            "target_payloads": {
                "foo": {"test_metrics": {"rmse": 1.0, "mae": 0.5, "r2": 0.9}},
            },
            "folds": 5,
            "seed": 42,
        }
        candidate_ckpt = {
            "model_family": "kan",
            "paradigm": "multi_output",
            "test_metrics": {
                "foo": {"rmse": 0.8, "mae": 0.4, "r2": 0.92},
            },
            "folds": 5,
            "seed": 42,
        }

        report = generate_report.render(
            "Title",
            "Baseline MLP",
            "KAN (Multi-output)",
            baseline_ckpt,
            candidate_ckpt,
            generate_report._extract(baseline_ckpt),
            generate_report._extract(candidate_ckpt),
        )

        assert "- Baseline protocol: per-target tuning, 5 folds, seed=42" in report
        assert "- Candidate protocol: multi-output tuning, 5 folds, seed=42" in report

    def test_describe_protocol_supports_per_target_baseline(self) -> None:
        ckpt = {
            "model_family": "kan",
            "paradigm": "per_target_baseline",
            "seed": 42,
        }

        assert (
            generate_report._describe_protocol(ckpt)
            == "per-target baseline training, seed=42"
        )
