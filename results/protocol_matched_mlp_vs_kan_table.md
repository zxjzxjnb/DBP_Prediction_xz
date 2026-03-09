# Protocol-Matched MLP vs KAN Comparison (Test Set)

## Setup

- Baseline: `Baseline MLP`
- Candidate: `KAN (Per-target)`
- Shared protocol: per-target tuning, 5 folds, seed=42, stability penalty=0.1
- Search budget: baseline trials=60, candidate trials=60

## Report-Ready Table

| Target | Baseline MLP RMSE | KAN (Per-target) RMSE | Better RMSE | Baseline MLP MAE | KAN (Per-target) MAE | Better MAE | Baseline MLP R² | KAN (Per-target) R² | Better R² |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| T_THMs_ug_L | 7.418 | 7.369 | KAN (Per-target) | 6.160 | 5.810 | KAN (Per-target) | 0.4735 | 0.4804 | KAN (Per-target) |
| DBCM_ug_L | 2.723 | 2.682 | KAN (Per-target) | 2.197 | 2.233 | Baseline MLP | 0.5076 | 0.5221 | KAN (Per-target) |
| BDCM_ug_L | 1.663 | 1.594 | KAN (Per-target) | 1.275 | 1.267 | KAN (Per-target) | 0.3556 | 0.4079 | KAN (Per-target) |
| Macro Average | 3.935 | 3.882 | KAN (Per-target) | 3.211 | 3.103 | KAN (Per-target) | 0.4456 | 0.4701 | KAN (Per-target) |

## Suggested Caption

Comparison between Baseline MLP and KAN (Per-target) under a protocol-matched per-target tuning setup on the held-out test set.
