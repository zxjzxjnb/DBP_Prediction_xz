# DBP Prediction

> Predicting disinfection by-products (DBPs) in drinking water with reproducible tabular machine-learning experiments.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project standardizes on **per-target prediction** for three
disinfection by-products from 9 water quality features. The current research
codebase supports neural and tree-based baselines through a shared
configuration, feature-pipeline, training, and tuning workflow.

| Target | Description |
| --- | --- |
| `T_THMs_ug_L` | Total trihalomethanes |
| `DBCM_ug_L` | Dibromochloromethane |
| `BDCM_ug_L` | Bromodichloromethane |

Key features:
- **Optuna Bayesian hyperparameter tuning** with TPE sampler and median pruning
- **5-fold CV ensemble** for robust predictions on small datasets (175 samples)
- **Per-target training and tuning** for all three DBP targets
- **Config-driven experiments** with reusable dataset and feature-pipeline contracts
- **Multiple model families** including MLP, KAN, Random Forest, and XGBoost

## Project Structure

```
├── dbp_prediction/              # Main Python package
│   ├── datasets/                # Tabular loading and split helpers
│   ├── features/                # Feature engineering pipeline and transforms
│   ├── models/                  # MLP, KAN, Random Forest, XGBoost adapters
│   ├── engine/                  # Experiment runner, evaluation, tuning
│   ├── schemas/                 # Dataset and experiment config contracts
│   ├── cli/                     # Command-line entry points
│   ├── artifacts/               # Run artifact storage helpers
│   ├── training.py              # Shared torch training utilities
│   ├── config.py                # Shared constants and compatibility exports
│   ├── data.py                  # Legacy-compatible data helpers
│   ├── metrics.py               # RMSE, MAE, R² computation
│   └── settings.py              # Shared paths and default values
├── tests/                       # Pytest test suite
├── experiments/                 # Example experiment configs
├── data/                        # Dataset
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
# Prepare or run a config-driven experiment
dbp run experiments/per_target_baseline.yaml --print-plan
dbp run experiments/per_target_tree_feature_starter.yaml --print-plan
```

Model-specific CLIs are still available when you want a focused workflow:

```bash
python -m dbp_prediction.cli.train_mlp --config experiments/per_target_baseline.yaml
python -m dbp_prediction.cli.train_kan --config experiments/per_target_baseline.yaml
```

Paths inside experiment configs are resolved relative to the config file itself.

Run outputs are written to `checkpoints/` and `results/` at execution time and
are ignored by Git.

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

### Research UI

An English-language Streamlit workbench is included for researchers who prefer
an interactive workflow over hand-editing YAML configs.

```bash
# Install the optional UI dependency
pip install -e ".[ui]"

# Launch the workbench
dbp-ui
```

The UI guides users through dataset selection, model setup, shared training
defaults, hyperparameter tuning controls, config preview, and experiment
execution using the same backend runner as the CLI.

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
