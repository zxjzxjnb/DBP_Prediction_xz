# ruff: noqa: I001
"""Deprecated wrapper for ``dbp_prediction.cli.tune_kan_per_target``."""

from __future__ import annotations

import importlib
import warnings


if __name__ == "__main__":
    warnings.warn(
        (
            "The scripts/ entry points are deprecated and kept only for "
            "compatibility. Use `python -m dbp_prediction.cli.tune_kan_per_target` "
            "instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    importlib.import_module("dbp_prediction.cli.tune_kan_per_target").main()
