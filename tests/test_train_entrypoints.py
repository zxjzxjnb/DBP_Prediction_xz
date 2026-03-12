"""Tests for legacy train CLI helpers."""

from __future__ import annotations

from argparse import Namespace

from dbp_prediction.cli.train_kan import _get_arg as get_kan_arg
from dbp_prediction.cli.train_mlp import _get_arg as get_mlp_arg


class TestTrainCliHelpers:
    """Tests for config-to-CLI default fallback behavior."""

    def test_mlp_get_arg_uses_default_when_namespace_value_is_none(self) -> None:
        args = Namespace(seed=None)

        assert get_mlp_arg(args, "seed", 42) == 42

    def test_kan_get_arg_uses_default_when_namespace_value_is_none(self) -> None:
        args = Namespace(seed=None)

        assert get_kan_arg(args, "seed", 42) == 42
