# SHAP Attribution Study

Controlled experiment to disentangle why the dominant SHAP feature shifted
from **Temperature** (Tai Lake) to **Cl₂ dose** (Dataset1).

## Three confounded hypotheses

1. **Feature set** — Dataset1 adds Cl₂ dose / contact time, which capture variance
   that Temperature proxied in the Tai Lake model.
2. **Sample size** — 175 vs ~488; small-sample models may latch onto covariance
   proxies (e.g. Temperature ↔ seasonal chlorination).
3. **Data source chemistry** — Different water sources, treatment processes, geography.

## Experimental conditions

```
              5-common (pH,UV254,T,TOC,Br)       6-feat (+Cl₂)
            ┌─────────────────────────────────┬──────────────────────┐
  Full N    │  B  (D1, ~488, 5)   [exists]    │  C (D1, ~488, 6)    │
            │                                 │        [exists]      │
            ├─────────────────────────────────┼──────────────────────┤
  N = 175   │  D  (D1, 175×5, 5)  [new]      │  E (D1, 175×5, 6)   │
            │                                 │        [new]         │
            └─────────────────────────────────┴──────────────────────┘
                External ref:  A'  (Tai Lake, 175, 5-common)  [new]
```

## Key comparisons

| Comparison | Controls                | Tests              |
|------------|------------------------|--------------------|
| D vs B     | data source, features  | **sample size**    |
| E vs C     | data source, features  | **sample size**    |
| D vs E     | data source, N         | **Cl₂ feature**    |
| B vs C     | data source, N         | **Cl₂ feature**    |
| D vs A'    | N, features            | **data source**    |

## Models

RF + XGBoost only (TreeExplainer — fast & exact).

## Sub-sampling protocol (D & E)

- K = 5 random seeds
- Each draw: N_TL rows from Dataset1, 80/20 train/test split
- Fixed hyperparameters from B (for D) and C (for E)
- 5-fold CV ensemble, same as existing pipeline
- Report mean ± std of SHAP rankings across seeds
