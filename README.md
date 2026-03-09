# DBP Prediction

> Predicting disinfection by-products (DBPs) in drinking water using MLP and KAN (Kolmogorov-Arnold Network) models.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project compares **Multi-Layer Perceptron (MLP)** and **Kolmogorov-Arnold Network (KAN)** architectures for predicting three disinfection by-products from 9 water quality features:

| Target | Description |
| --- | --- |
| `T_THMs_ug_L` | Total trihalomethanes |
| `DBCM_ug_L` | Dibromochloromethane |
| `BDCM_ug_L` | Bromodichloromethane |

Key features:
- **Optuna Bayesian hyperparameter tuning** with TPE sampler and median pruning
- **5-fold CV ensemble** for robust predictions on small datasets (175 samples)
- **Multi-seed sweep** for statistically rigorous model comparison
- **Per-target vs multi-output** KAN paradigm comparison

## Project Structure

```
├── dbp_prediction/              # Main Python package
│   ├── config.py                # Shared constants and config dataclasses
│   ├── data.py                  # Data loading, splitting, scaling
│   ├── metrics.py               # RMSE, MAE, R² computation
│   ├── training.py              # Training loop, CV ensemble, prediction
│   ├── models/
│   │   ├── mlp.py               # MLP model definition
│   │   └── kan.py               # KAN model builder
│   └── cli/                     # Command-line entry points
│       ├── train_mlp.py
│       ├── train_kan.py
│       ├── tune_mlp.py
│       ├── tune_kan.py
│       ├── tune_kan_per_target.py
│       ├── compare_kan_paradigms.py
│       ├── sweep_kan_paradigms.py
│       └── generate_report.py
├── tests/                       # Pytest test suite
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
git clone https://github.com/your-username/DBPs-prediction-by-kan.git
cd DBPs-prediction-by-kan

# Option 1: pip install (recommended)
pip install -e ".[dev]"

# Option 2: conda
conda env create -f environment.yml
conda activate kan_model
pip install -e ".[dev]"
```

### Training

```bash
# Train baseline MLP
python -m dbp_prediction.cli.train_mlp

# Train baseline KAN
python -m dbp_prediction.cli.train_kan

# Or use installed CLI commands
dbp-train-mlp --seed 42
dbp-train-kan --seed 42 --grid 8
```

### Hyperparameter Tuning

```bash
# Tune MLP (per-target, Optuna)
python -m dbp_prediction.cli.tune_mlp --trials 120

# Tune KAN (multi-output)
python -m dbp_prediction.cli.tune_kan --trials 60

# Tune KAN (per-target)
python -m dbp_prediction.cli.tune_kan_per_target --trials 60
```

### Evaluation & Reporting

```bash
# Compare KAN paradigms
python -m dbp_prediction.cli.compare_kan_paradigms

# Multi-seed sweep
python -m dbp_prediction.cli.sweep_kan_paradigms --seeds 42,2024,3407 --trials 30

# Generate comparison report
python -m dbp_prediction.cli.generate_report
```

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
