# DBP Prediction

> Predicting disinfection by-products (DBPs) in drinking water using MLP and KAN (Kolmogorov-Arnold Network) models.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project standardizes on **per-target prediction** with **Multi-Layer Perceptron (MLP)** and **Kolmogorov-Arnold Network (KAN)** models for three disinfection by-products from 9 water quality features:

| Target | Description |
| --- | --- |
| `T_THMs_ug_L` | Total trihalomethanes |
| `DBCM_ug_L` | Dibromochloromethane |
| `BDCM_ug_L` | Bromodichloromethane |

Key features:
- **Optuna Bayesian hyperparameter tuning** with TPE sampler and median pruning
- **5-fold CV ensemble** for robust predictions on small datasets (175 samples)
- **Per-target training and tuning** for all three DBP targets
- **Historical multi-output KAN comparison tools** retained for backtesting only

## Project Structure

```
├── dbp_prediction/              # Main Python package
│   ├── config.py                # Shared constants and config dataclasses
│   ├── data.py                  # Data loading, splitting, scaling
│   ├── metrics.py               # RMSE, MAE, R² computation
│   ├── settings.py              # Shared paths and default values
│   ├── schemas/                 # Dataset/experiment config contracts
│   ├── datasets/                # Structured data loading/splitting/preprocess layer
│   ├── artifacts/               # Run artifact storage helpers
│   ├── engine/                  # Experiment runner skeleton
│   ├── training.py              # Training loop, CV ensemble, prediction
│   ├── models/
│   │   ├── mlp.py               # MLP model definition
│   │   └── kan.py               # KAN model builder
│   └── cli/                     # Command-line entry points
│       ├── main.py              # Unified config-driven CLI (`dbp run`)
│       ├── train_mlp.py         # Per-target baseline MLP training
│       ├── train_kan.py         # Per-target baseline KAN training
│       ├── tune_mlp.py          # Per-target MLP Optuna tuning
│       ├── tune_kan.py          # Legacy multi-output KAN tuning
│       ├── tune_kan_per_target.py   # Per-target KAN Optuna tuning
│       ├── compare_kan_paradigms.py # Legacy KAN paradigm comparison
│       ├── sweep_kan_paradigms.py   # Legacy multi-seed KAN sweep
│       └── generate_report.py   # Report table from tuned checkpoints
├── tests/                       # Pytest test suite
├── scripts/                     # Deprecated compatibility wrappers
├── data/                        # Dataset
├── checkpoints/                 # Model checkpoints (.pt)
├── results/                     # Metrics logs and reports
├── references/                  # Upstream reference code
├── pyproject.toml               # Package metadata and tooling config
└── environment.yml              # Conda environment specification
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/zxjzxjnb/DBPs-prediction-by-kan.git
cd DBPs-prediction-by-kan

# Option 1: editable install (recommended for development)
pip install -e ".[dev]"

# Option 2: standard install (dataset is bundled with the package)
pip install .

# Option 3: conda
conda env create -f environment.yml
conda activate kan_model
pip install -e ".[dev]"
```

### Training

```bash
# Train baseline MLP (per-target)
python -m dbp_prediction.cli.train_mlp

# Train baseline KAN (per-target)
python -m dbp_prediction.cli.train_kan

# Train only selected targets
python -m dbp_prediction.cli.train_mlp --targets T_THMs_ug_L,DBCM_ug_L
python -m dbp_prediction.cli.train_kan --targets T_THMs_ug_L

# Or use installed CLI commands
dbp-train-mlp --seed 42
dbp-train-kan --seed 42 --grid 8
```

Phase 1 also adds an experiment-config bridge so the current MLP/KAN CLIs can
read a shared YAML experiment contract without changing the training engine yet:

```bash
python -m dbp_prediction.cli.train_mlp --config experiments/per_target_baseline.yaml
python -m dbp_prediction.cli.train_kan --config experiments/per_target_baseline.yaml
```

Phase 2 adds a unified runner skeleton that prepares a run directory, snapshots
the resolved config, inspects the dataset, and writes a run plan:

```bash
dbp run experiments/per_target_baseline.yaml --print-plan
```

Paths inside experiment configs are resolved relative to the config file itself.

### Hyperparameter Tuning

```bash
# Tune MLP (per-target, Optuna)
python -m dbp_prediction.cli.tune_mlp --trials 120

# Tune KAN (per-target, Optuna)
python -m dbp_prediction.cli.tune_kan_per_target --trials 60
```

### Evaluation & Reporting

```bash
# Generate MLP vs per-target KAN comparison report
python -m dbp_prediction.cli.generate_report
```

### Optional Historical Comparison

These commands are retained only if you want to reproduce the older
multi-output-vs-per-target KAN comparison.

```bash
# Tune legacy multi-output KAN
python -m dbp_prediction.cli.tune_kan --trials 60

# Compare legacy multi-output KAN against current per-target KAN
python -m dbp_prediction.cli.compare_kan_paradigms

# Run multi-seed historical comparison
python -m dbp_prediction.cli.sweep_kan_paradigms --seeds 42,2024,3407 --trials 30
```

The `scripts/` directory now contains compatibility wrappers only. New work
should use `python -m dbp_prediction.cli.<command>` or the installed
`dbp-*` console scripts.

### Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test module
pytest tests/test_models.py
```

### Linting

```bash
# Check code style
ruff check dbp_prediction/ tests/

# Auto-fix issues
ruff check --fix dbp_prediction/ tests/

# Format code
ruff format dbp_prediction/ tests/
```

## Dataset

- **Source**: `data/DBP_dataset_DWTP_B.csv`
- **Samples**: 175 (141 train / 34 test)
- **Features**: 9 water quality parameters (pH, COD, NH₄-N, NO₂-N, NO₃-N, Br⁻, TOC, UV254, temperature)
- **Targets**: 3 DBP concentrations (T-THMs, DBCM, BDCM)

## Acknowledgments

This project is based on [DBPs-prediction-by-kan](https://github.com/XiaoyanLi-enviro/DBPs-prediction-by-kan) by **Xiaoyan Li** (2024), licensed under the [MIT License](LICENSE).
