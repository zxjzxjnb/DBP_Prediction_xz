# DBP Prediction

> Predicting disinfection by-products (DBPs) in drinking water with reproducible tabular machine-learning experiments.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project standardizes on **per-target prediction** for three
disinfection by-products from tabular water-quality features. The primary
formal experiments use the **U.S. Dataset** with chlorine dose and contact-time
features, while the packaged 175-row benchmark is retained for lightweight
examples and legacy comparisons. The current research codebase supports neural
and tree-based baselines through a shared configuration, feature-pipeline,
training, and tuning workflow.

| Target | Description |
| --- | --- |
| `T_THMs_ug_L` | Total trihalomethanes |
| `DBCM_ug_L` | Dibromochloromethane |
| `BDCM_ug_L` | Bromodichloromethane |

Key features:
- **Optuna Bayesian hyperparameter tuning** with TPE sampler and median pruning
- **5-fold CV ensemble** for robust held-out evaluation on small tabular datasets
- **Per-target training and tuning** for all three DBP targets
- **Config-driven experiments** with reusable dataset and feature-pipeline contracts
- **Multiple model families** including MLP, KAN, Random Forest, and XGBoost
- **SHAP attribution and interaction analysis** for model interpretation

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
git clone https://github.com/zxjzxjnb/DBP_Prediction_xz.git
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

## Results

Reported metrics below are from the formal 7-feature **U.S. Dataset** runs,
ranked by held-out test RMSE. The 7-feature setting uses pH, UV254,
temperature, TOC, bromide, chlorine dose, and contact time.

| Target | Best model | RMSE | MAE | R² | Key SHAP features |
| --- | --- | ---: | ---: | ---: | --- |
| THM4 | Random Forest | 37.378 | 25.934 | 0.855 | Chlorine dose, TOC, bromide |
| BDCM | Random Forest | 8.430 | 6.321 | 0.843 | Bromide, chlorine dose, UV254 |
| DBCM | MLP | 4.924 | 3.338 | 0.713 | Bromide, UV254, chlorine dose |

The interpretation workflow also exports SHAP interaction summaries, including
prominent bromide/chlorine-dose and TOC/chlorine-dose interaction patterns
across the U.S. Dataset targets.

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

## Datasets

- **U.S. Dataset, formal experiments**: `data/dataset1_dbp_formation_with_split.csv`
  contains 514 records with a predefined 406 train / 108 test split. The
  formal 7-feature runs use pH, UV254, temperature, TOC, bromide, chlorine
  dose, and contact time to predict THM4, DBCM, and BDCM. Complete-case
  evaluation artifacts contain 106 held-out test rows after `drop_missing`.
- **Packaged benchmark / legacy demo**: `data/DBP_dataset_DWTP_B.csv` contains
  175 records with a predefined 141 train / 34 test split, 9 water-quality
  features, and 3 DBP targets.

## Acknowledgments

This project is based on [DBPs-prediction-by-kan](https://github.com/XiaoyanLi-enviro/DBPs-prediction-by-kan) by **Xiaoyan Li** (2024), licensed under the [MIT License](LICENSE).
