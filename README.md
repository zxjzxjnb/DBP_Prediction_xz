# DBPs Prediction Project

Predicting disinfection by-products (DBPs) in drinking water with MLP and KAN baselines.

## Project Structure

- `data/`: dataset files
- `scripts/`: training and tuning scripts
- `results/`: text logs / reported metrics
- `checkpoints/`: model checkpoint files (`.pt`)
- `references/models_from_original_project/`: upstream reference code and notebook

## Dataset

- `data/DBP_dataset_DWTP_B.csv`
- 175 samples (141 train / 34 test), 9 input features, 3 targets:
  `T_THMs_ug_L`, `DBCM_ug_L`, `BDCM_ug_L`

## Usage

Activate environment:

```bash
conda activate kan_model
```

Train MLP:

```bash
python scripts/train_mlp.py
```

Tune MLP:

```bash
python scripts/tune_mlp.py --trials 60
```

Train KAN:

```bash
python scripts/train_kan.py
```

Tune KAN (multi-output):

```bash
python scripts/tune_kan.py --trials 30
```

Tune KAN (per-target):

```bash
python scripts/tune_kan_per_target.py --trials 30
```

Compare KAN paradigms:

```bash
python scripts/compare_kan_paradigms.py
```

Sweep KAN paradigms across multiple random seeds:

```bash
python scripts/sweep_kan_paradigms.py --seeds 42,2024,3407,7777,10086 --trials 30 --skip-existing
```

All scripts read data from `data/` and save checkpoints to `checkpoints/` by default.

## Acknowledgments

This project is based on [DBPs-prediction-by-kan](https://github.com/XiaoyanLi-enviro/DBPs-prediction-by-kan) by **Xiaoyan Li** (2024), licensed under the [MIT License](LICENSE).
