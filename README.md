# DBPs Prediction by MLP Baseline

Predicting disinfection by-products (DBPs) in drinking water using MLP neural networks.

## Dataset

`DBP_dataset_DWTP_B.csv` — 176 water quality samples (141 train / 35 test) from DWTP-B, with 9 input features and 3 target DBP concentrations (T_THMs, DBCM, BDCM).

## Usage

```bash
conda activate kan_model
python train_mlp.py
```

## Files

| File | Description |
|------|-------------|
| `train_mlp.py` | MLP training with StandardScaler, early stopping, per-target evaluation |
| `base_model.py` | Original MLP baseline (with code review annotations) |
| `KAN_model.py` | Original KAN model from upstream repo |
| `kan_example.ipynb` | KAN tutorial notebook |

## Acknowledgments

This project is based on [DBPs-prediction-by-kan](https://github.com/XiaoyanLi-enviro/DBPs-prediction-by-kan) by **Xiaoyan Li** (2024), licensed under the [MIT License](LICENSE).
