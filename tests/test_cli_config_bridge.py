"""Tests for bridging experiment configs into legacy CLIs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dbp_prediction.cli.config_bridge import bind_legacy_model_config


class TestLegacyConfigBridge:
    """Tests for model-specific config extraction used by current train CLIs."""

    def test_bind_legacy_model_config_uses_matching_model_and_output_dir(
        self,
        tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "experiment.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset": {
                        "path": "data/custom.csv",
                        "features": ["f1", "f2"],
                        "targets": ["y1", "y2"],
                    },
                    "task": {
                        "strategy": "per_target",
                        "targets": ["y2"],
                    },
                    "models": [
                        {"name": "mlp", "params": {"hidden_dims": [64, 32]}},
                        {"name": "kan", "params": {"grid": 8, "k": 3}},
                    ],
                    "outputs": {
                        "dir": str(tmp_path / "artifacts"),
                    },
                }
            ),
            encoding="utf-8",
        )

        binding = bind_legacy_model_config(config_path, "mlp", Path("checkpoints/mlp_checkpoint.pt"))

        assert binding.model.name == "mlp"
        assert binding.selected_targets == ["y2"]
        assert binding.output_path == tmp_path / "artifacts" / "mlp_checkpoint.pt"
        assert binding.save_models is True

    def test_bind_legacy_model_config_rejects_multi_output_strategy(self, tmp_path: Path) -> None:
        config_path = tmp_path / "experiment.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset": {
                        "path": "data/custom.csv",
                        "features": ["f1"],
                        "targets": ["y1", "y2"],
                    },
                    "task": {
                        "strategy": "multi_output",
                    },
                    "models": [{"name": "kan"}],
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="only support task.strategy='per_target'"):
            bind_legacy_model_config(config_path, "kan", Path("checkpoints/kan_checkpoint.pt"))

    def test_bind_legacy_model_config_accepts_feature_steps(self, tmp_path: Path) -> None:
        config_path = tmp_path / "experiment.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset": {
                        "path": "data/custom.csv",
                        "features": ["f1"],
                        "targets": ["y1"],
                    },
                    "features": {
                        "steps": [{"name": "log1p"}],
                    },
                    "models": [{"name": "mlp"}],
                }
            ),
            encoding="utf-8",
        )

        binding = bind_legacy_model_config(config_path, "mlp", Path("checkpoints/mlp_checkpoint.pt"))

        assert binding.model.name == "mlp"

    def test_bind_legacy_model_config_rejects_tuning_enabled(self, tmp_path: Path) -> None:
        config_path = tmp_path / "experiment.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset": {
                        "path": "data/custom.csv",
                        "features": ["f1"],
                        "targets": ["y1"],
                    },
                    "models": [{"name": "kan"}],
                    "tuning": {"enabled": True},
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="do not support 'tuning.enabled=true'"):
            bind_legacy_model_config(config_path, "kan", Path("checkpoints/kan_checkpoint.pt"))

    def test_bind_legacy_model_config_rejects_save_predictions(self, tmp_path: Path) -> None:
        config_path = tmp_path / "experiment.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset": {
                        "path": "data/custom.csv",
                        "features": ["f1"],
                        "targets": ["y1"],
                    },
                    "models": [{"name": "mlp"}],
                    "outputs": {"save_predictions": True},
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="do not support 'outputs.save_predictions=true'"):
            bind_legacy_model_config(config_path, "mlp", Path("checkpoints/mlp_checkpoint.pt"))

    def test_bind_legacy_model_config_supports_disabling_checkpoint_saves(
        self,
        tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "experiment.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset": {
                        "path": "data/custom.csv",
                        "features": ["f1"],
                        "targets": ["y1"],
                    },
                    "models": [{"name": "mlp"}],
                    "outputs": {"save_models": False},
                }
            ),
            encoding="utf-8",
        )

        binding = bind_legacy_model_config(config_path, "mlp", Path("checkpoints/mlp_checkpoint.pt"))

        assert binding.save_models is False

    def test_bind_legacy_model_config_rejects_ambiguous_duplicate_models(self, tmp_path: Path) -> None:
        config_path = tmp_path / "experiment.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset": {
                        "path": "data/custom.csv",
                        "features": ["f1"],
                        "targets": ["y1"],
                    },
                    "models": [
                        {"name": "mlp", "alias": "baseline"},
                        {"name": "mlp", "alias": "wide"},
                    ],
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="require exactly one enabled model"):
            bind_legacy_model_config(config_path, "mlp", Path("checkpoints/mlp_checkpoint.pt"))
