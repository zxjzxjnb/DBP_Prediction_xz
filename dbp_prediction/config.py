"""Legacy compatibility exports for shared configuration values."""

from __future__ import annotations

from dataclasses import dataclass, field
from dbp_prediction.settings import (
    CHECKPOINT_DIR,
    DATA_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_FOLDS,
    DEFAULT_LR,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_STABILITY_PENALTY,
    DEFAULT_TRIALS,
    DEFAULT_VAL_FRACTION,
    DEFAULT_WEIGHT_DECAY,
    PACKAGE_DATA_DIR,
    PACKAGED_DATA_PATH,
    PROJECT_ROOT,
    RESULTS_DIR,
    first_existing_path,
    resolve_artifact_path,
)

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

    seed: int = DEFAULT_SEED
    max_epochs: int = DEFAULT_MAX_EPOCHS
    patience: int = DEFAULT_PATIENCE
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    val_fraction: float = DEFAULT_VAL_FRACTION


@dataclass
class TuningConfig:
    """Configuration for Optuna hyperparameter search."""

    seed: int = DEFAULT_SEED
    trials: int = DEFAULT_TRIALS
    folds: int = DEFAULT_FOLDS
    max_epochs: int = DEFAULT_MAX_EPOCHS
    patience: int = DEFAULT_PATIENCE
    stability_penalty: float = DEFAULT_STABILITY_PENALTY
    targets: list[str] = field(default_factory=lambda: list(TARGET_COLS))
