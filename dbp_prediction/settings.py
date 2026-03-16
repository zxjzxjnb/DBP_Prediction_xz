"""Project settings, paths, and shared default values."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def _find_project_root() -> Path:
    """Locate the project root directory."""
    candidate = Path(__file__).resolve().parents[1]
    if (candidate / "data").is_dir():
        return candidate

    cwd = Path.cwd()
    if (cwd / "data").is_dir():
        return cwd

    # Last resort: use CWD so outputs do not land in site-packages.
    return cwd


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "_data"
PACKAGED_DATA_PATH = PACKAGE_DATA_DIR / "DBP_dataset_DWTP_B.csv"

DEFAULT_SEED = 42
DEFAULT_MAX_EPOCHS = 2000
DEFAULT_PATIENCE = 100
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_TRIALS = 60
DEFAULT_FOLDS = 5
DEFAULT_STABILITY_PENALTY = 0.10


def first_existing_path(candidates: Iterable[Path]) -> Path | None:
    """Return the first existing path from an ordered candidate list."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_artifact_path(
    explicit_path: str | None,
    candidates: list[Path],
    label: str,
) -> Path:
    """Resolve a CLI artifact path, supporting legacy defaults."""
    if explicit_path is not None:
        return Path(explicit_path)

    resolved = first_existing_path(candidates)
    if resolved is not None:
        return resolved

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No {label} found. Searched: {searched}")


def _resolve_default_data_path() -> Path:
    candidates = [
        DATA_DIR / "DBP_dataset_DWTP_B.csv",
        PACKAGED_DATA_PATH,
    ]
    return first_existing_path(candidates) or candidates[0]


DEFAULT_DATA_PATH = _resolve_default_data_path()
