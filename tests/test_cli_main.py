"""Tests for the unified Phase 2 CLI parser."""

from __future__ import annotations

from dbp_prediction.cli.main import build_parser


class TestCliMain:
    """Tests for the config-driven CLI entrypoint."""

    def test_run_parser_accepts_expected_arguments(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "run",
                "experiments/per_target_baseline.yaml",
                "--output-dir",
                "results/demo",
                "--run-id",
                "phase2-demo",
                "--prepare-only",
                "--skip-data-inspection",
                "--print-plan",
            ]
        )

        assert args.command == "run"
        assert args.config == "experiments/per_target_baseline.yaml"
        assert args.output_dir == "results/demo"
        assert args.run_id == "phase2-demo"
        assert args.prepare_only is True
        assert args.skip_data_inspection is True
        assert args.print_plan is True
