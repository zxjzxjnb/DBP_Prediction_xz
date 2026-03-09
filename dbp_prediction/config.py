"""Shared configuration constants and dataclasses for the DBP prediction project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ── Project paths ────────────────────────────────────────────────────────────


def _find_project_root() -> Path:
    """Locate the project root directory.

    Strategy:
    1. Walk up from this file looking for ``data/`` — works for source checkouts
       and editable installs.
    2. Fall back to the current working directory — works for normal pip installs
       where the user runs commands from their project directory.
    """
    candidate = Path(__file__).resolve().parents[1]
    if (candidate / "data").is_dir():
        return candidate

    cwd = Path.cwd()
    if (cwd / "data").is_dir():
        return cwd

    # Last resort: use CWD so outputs don't land in site-packages.
    return cwd


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_DATA_PATH = DATA_DIR / "DBP_dataset_DWTP_B.csv"

# ── Column definitions ──────────────────────────────────────────────────────

FEATURE_COLS: list[str] = [
    "pH",
    "COD_mg_L",
    "NH4_N_mg_L",
    "NO2_N_mg_L",
    "NO3_N_mg_L",
    "Br_mg_L",
    "TOC_mg_L",
    "UV254_A_cm",
    "temp_C",
]

TARGET_COLS: list[str] = [
    "T_THMs_ug_L",
    "DBCM_ug_L",
    "BDCM_ug_L",
]

SPLIT_COL = "split"
TRAIN_LABEL = "train"
TEST_LABEL = "test"


# ── Training configuration ──────────────────────────────────────────────────


@dataclass
class TrainingConfig:
    """Configuration for a single training run."""

    seed: int = 42
    max_epochs: int = 2000
    patience: int = 100
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.15


@dataclass
class TuningConfig:
    """Configuration for Optuna hyperparameter search."""

    seed: int = 42
    trials: int = 60
    folds: int = 5
    max_epochs: int = 2000
    patience: int = 100
    stability_penalty: float = 0.10
    targets: list[str] = field(default_factory=lambda: list(TARGET_COLS))
