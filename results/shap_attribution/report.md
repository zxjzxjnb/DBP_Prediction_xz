# SHAP Attribution Study: Disentangling the Drivers of Feature Importance Shifts in DBP Prediction

---

## 1. Background and Motivation

Machine learning models trained to predict disinfection by-product (DBP) concentrations in drinking water rely on SHAP (SHapley Additive exPlanations) values to identify which input features drive each prediction. A striking pattern emerged when comparing two independently collected datasets:

- **Tai Lake (DWTP-B)**: SHAP analysis consistently ranked **Temperature** as the single dominant predictor for all three DBP targets — THMs, DBCM, and BDCM — across all four model families (Random Forest, XGBoost, MLP, KAN).
- **Dataset 1 (multi-facility)**: Temperature is no longer the dominant feature. In Random Forest it drops to rank 4–5 across targets; in XGBoost it remains secondary only for DBCM, while the leading features shift to **Cl₂ dose** (for THMs) and **Bromide** (for DBCM/BDCM).

This shift could plausibly arise from at least three sources that are inherently confounded when comparing two real-world datasets:

1. **Feature set composition** — Dataset 1 includes direct process variables such as Cl₂ dose that are absent from the Tai Lake dataset. A newly introduced feature with high predictive power may absorb variance previously attributed to Temperature.
2. **Sample size** — Tai Lake contains 175 observations; Dataset 1 contains 514 (488 complete cases). Small-sample models are prone to latching onto spurious covariances — for instance, Temperature may act as a seasonal proxy for chlorination intensity in a single-plant dataset.
3. **Data source / data-distribution effect** — The two datasets originate from different geographic locations, water sources, and treatment configurations. The statistical relationship between Temperature and DBP formation may therefore differ between sites.

Because these three factors change simultaneously between the two datasets, any naive comparison of SHAP rankings is confounded. This study was designed to isolate each factor through a controlled factorial experiment.

---

## 2. Experimental Design

### 2.1 Control Logic

The study defines five experimental conditions, each varying at most one factor relative to a reference:

| Condition | Dataset | N | Features | Status | Role |
|-----------|---------|---|----------|--------|------|
| **A'** | Tai Lake | 175 | 5-common | New | External reference baseline |
| **B** | Dataset 1 | 488 | 5-common | Existing | Full-data, matched features |
| **C** | Dataset 1 | 488 | 6-feat (+Cl₂) | Existing | Full-data, expanded features |
| **D** | Dataset 1 | 175 × 5 | 5-common | New | Subsampled, matched features |
| **E** | Dataset 1 | 175 × 5 | 6-feat (+Cl₂) | New | Subsampled, expanded features |

The **5-common feature set** is the intersection of both datasets: `{pH, UV254, Temperature, TOC, Bromide}`. This is the only feature set that allows a direct, feature-aligned comparison across data sources.

The key comparisons and what each isolates:

| Comparison | Fixed | Varied | Question answered |
|------------|-------|--------|-------------------|
| D vs. B | Data source, features | **Sample size** (175 vs. 488) | Is the shift driven by how many samples were available? |
| E vs. C | Data source, features | **Sample size** (175 vs. 488) | Same question, in the presence of Cl₂ dose |
| D vs. E | Data source, N | **Feature set** (5 vs. 6) | Does adding Cl₂ dose change rankings at small N? |
| B vs. C | Data source, N | **Feature set** (5 vs. 6) | Does adding Cl₂ dose change rankings at full N? |
| D vs. A' | Features, N | **Data source** | Is the shift a property of the data itself, not the model? |

### 2.2 The 5-Common Feature Set

Tai Lake's original 6-feature ablation included COD, which does not exist in Dataset 1. To achieve strict feature alignment, a new Tai Lake model (condition A') was trained using only the 5 shared features. This required training a new model from scratch — condition A' did not previously exist.

### 2.3 Sub-sampling Protocol (Conditions D and E)

To simulate "what if Dataset 1 had only 175 samples?", the following procedure was applied:

- **Pool**: all 488 complete-case rows from Dataset 1 (for the 6-feature set).
- **K = 5 independent random seeds** (seeds 1–5).
- Each seed: draw 175 rows without replacement, then split into 141 train / 34 test (matching Tai Lake's exact split ratio).
- **Hyperparameters**: fixed at the per-target optimal values from conditions B and C respectively, with no re-tuning. This prevents Optuna sampling variation from contaminating the sample-size comparison.
- **Ensemble**: 5-fold cross-validation per seed, producing a 5-member ensemble. SHAP values are averaged across folds. Final SHAP rankings are summarised as mean ± std across 5 seeds.

### 2.4 Models and SHAP Method

Only **Random Forest** and **XGBoost** were used, for two reasons:
- Both support `shap.TreeExplainer`, which computes exact Shapley values in polynomial time — no approximation.
- Running MLP/KAN with `KernelExplainer` across 5 seeds × 5 folds would be computationally prohibitive.

For cross-dataset interpretation, Tai Lake's `T_THMs_ug_L` is aligned with Dataset 1's `thm4_in_avg` as the closest shared THM-family response. SHAP values are computed on the held-out test set, averaged across the 5 CV fold members, and reported in raw target units after undoing fold-level target scaling.

---

## 3. Results

### 3.1 Condition A': Tai Lake Baseline (5-Common Features)

Training A' — a Tai Lake model with only `{pH, UV254, Temperature, TOC, Bromide}` — confirms that removing COD does not change the dominant pattern: **Temperature ranks #1 across all three targets and both models**, with a large absolute margin.

**Random Forest — Condition A' (Tai Lake, N=175, 5-common features)**

| Rank | THMs | DBCM | BDCM |
|------|------|------|------|
| 1 | Temperature (5.962) | Temperature (1.643) | Temperature (1.055) |
| 2 | Bromide (1.608) | Bromide (1.172) | UV254 (0.162) |
| 3 | TOC (0.720) | TOC (0.454) | Bromide (0.147) |
| 4 | pH (0.593) | UV254 (0.226) | pH (0.122) |
| 5 | UV254 (0.380) | pH (0.224) | TOC (0.095) |

In raw target units, Temperature remains the largest SHAP term for all three targets after COD is removed. This is not an artifact of the 6-feature model having COD compete with Temperature — the dominance is intrinsic to the Tai Lake data.

---

### 3.2 Sample Size Effect: D vs. B and E vs. C

The central question: does reducing Dataset 1 from 488 to 175 samples cause Temperature to rise in the SHAP rankings?

**Random Forest — THM4 / THM-family: B (full) vs. D (subsampled)**

| Feature | B (N=488) Rank | D (N=175) Rank ± SD |
|---------|---------------|---------------------|
| TOC | **#1** | **#1.4 ± 0.55** |
| UV254 | #3 | #1.8 ± 0.84 |
| Bromide | #2 | #2.8 ± 0.45 |
| Temperature | #4 | #4.4 ± 0.55 |
| pH | #5 | #4.6 ± 0.55 |

**Random Forest — BDCM: B (full) vs. D (subsampled)**

| Feature | B (N=488) Rank | D (N=175) Rank ± SD |
|---------|---------------|---------------------|
| Bromide | **#1** | **#1.2 ± 0.45** |
| TOC | #3 | #2.4 ± 0.89 |
| UV254 | #2 | #2.4 ± 0.55 |
| Temperature | #4 | #4.2 ± 0.45 |
| pH | #5 | #4.8 ± 0.45 |

For Random Forest, the rankings are **highly stable** under sub-sampling: Temperature remains rank 4–5 whether 488 or 175 samples are used, with rank standard deviations of roughly 0.45–0.89. XGBoost shows the same qualitative result for THM4 and BDCM, while DBCM is more mixed: Temperature remains secondary rather than dominant, with mean rank around #3 under sub-sampling.

The same holds for conditions E vs. C (with Cl₂ dose included): Cl₂ dose remains the dominant THM feature and stays near the top for BDCM after sub-sampling; reducing N does not restore Temperature to the top of the ranking.

**Finding 1: Sample size is not the driver.** Halving the Dataset 1 sample count does not elevate Temperature in the rankings.

---

### 3.3 Feature Set Effect: D vs. E and B vs. C

Does introducing Cl₂ dose explain why Temperature was relegated to low ranks?

**Random Forest — THM4 / THM-family: D (5-common) vs. E (6-feat), both N=175**

| Feature | D (no Cl₂) Rank ± SD | E (+Cl₂) Rank ± SD |
|---------|----------------------|---------------------|
| **Cl₂ dose** | — | **#1.4 ± 0.89** |
| TOC | #1.2 ± 0.45 | #2.0 ± 0.71 |
| UV254 | #2.0 ± 0.71 | #3.2 ± 0.84 |
| Bromide | #2.8 ± 0.45 | #3.4 ± 0.89 |
| Temperature | #4.4 ± 0.55 | #5.0 ± 0.00 |
| pH | #4.6 ± 0.55 | #6.0 ± 0.00 |

Cl₂ dose enters at rank #1 when added, pushing other features down. However, **Temperature is already at rank #4 before Cl₂ is added** (condition D). Adding Cl₂ only moves Temperature from #4.4 to #5.0 on average — a modest change.

**Finding 2: The Cl₂ feature matters but is not the root cause.** Cl₂ dose captures real variance in THM formation and correctly rises to the top. However, it does not explain why Temperature was ever at rank #4 in Dataset 1 — that pattern was established without Cl₂.

---

### 3.4 Data Source Effect: D vs. A'

This is the decisive comparison: same feature set `{pH, UV254, Temperature, TOC, Bromide}`, same sample size (N=175), different dataset.

**Random Forest — All three targets: A' (Tai Lake) vs. D (Dataset 1)**

| Target | Feature | A' Rank | D Mean Rank ± SD |
|--------|---------|---------|-----------------|
| THM4 | **Temperature** | **#1** | #4.4 ± 0.55 |
| THM4 | TOC | #3 | #1.4 ± 0.55 |
| THM4 | Bromide | #2 | #2.8 ± 0.45 |
| THM4 | UV254 | #5 | #1.8 ± 0.84 |
| | | | |
| DBCM | **Temperature** | **#1** | #4.2 ± 0.84 |
| DBCM | Bromide | #2 | #1.0 ± 0.00 |
| DBCM | pH | #5 | #2.0 ± 0.00 |
| | | | |
| BDCM | **Temperature** | **#1** | #4.2 ± 0.45 |
| BDCM | Bromide | #3 | #1.2 ± 0.45 |
| BDCM | UV254 | #2 | #2.4 ± 0.55 |

The contrast is unambiguous: **Temperature ranks #1 in Tai Lake and #4–5 in Dataset 1, even when both datasets have exactly the same features and exactly the same number of samples.**

XGBoost supports the same broad conclusion for THM4 and BDCM, but DBCM is more nuanced: Temperature stays #1 in Tai Lake, yet in Dataset 1 it remains a secondary feature (rank about #2 in full-data runs and about #3 under sub-sampling) rather than disappearing completely.

**Finding 3: A data-source / data-distribution effect is the primary driver within this tree-model SHAP study.** The SHAP shift is strongest and cleanest for THM4 and BDCM, and remains directionally consistent for DBCM. Within the Random Forest and XGBoost protocol used here, the change cannot be explained by sample size or feature availability alone.

### 3.5 Within-Dataset1 subgroup check: low vs. high tsid temperature variability

To quantify the intuition that "Temperature barely changes inside many Dataset 1 series," Dataset 1 test rows were split by the within-`tsid` standard deviation of `temp_in_avg`:

- **Low-variability group**: within-`tsid` temperature std < 1.0 C
- **High-variability group**: within-`tsid` temperature std >= 1.0 C

After complete-case filtering for the formal tree-model checkpoints, the subgroup test set contained:

| Group | Test rows | Test tsid | Row-level temp std | Mean within-tsid temp std |
|-------|-----------|-----------|--------------------|---------------------------|
| Low variability | 50 | 10 | 2.33 C | 0.34 C |
| High variability | 55 | 7 | 3.68 C | 2.20 C |

Using **tree-model SHAP** on the held-out test rows, Temperature ranked higher in the **high-variability** group in **10 of 12** `run x model x target` comparisons, was unchanged in 2, and never ranked lower. The mean improvement was **0.92 rank positions**.

| Model | Target | B low | B high | C low | C high |
|-------|--------|-------|--------|-------|--------|
| RF | THM4 | #4 | #4 | #5 | #5 |
| RF | DBCM | #5 | #3 | #5 | #4 |
| RF | BDCM | #5 | #4 | #6 | #5 |
| XGB | THM4 | #5 | #4 | #6 | #5 |
| XGB | DBCM | #3 | #2 | #3 | #2 |
| XGB | BDCM | #5 | #4 | #6 | #5 |

This subgroup evidence sharpens the interpretation:

- **Temperature does recover somewhat when Dataset 1 contains more internal thermal movement.**
- The clearest lift appears for **DBCM**, where Temperature moves from rank **#5 -> #3** in B-RF and from **#3 -> #2** in both XGBoost settings.
- However, the recovery is still modest: Temperature does **not** become the dominant Dataset 1 feature. TOC, Bromide, and Cl₂ remain the main drivers depending on target and feature set.

So, **limited within-series temperature variation is a contributing explanation for weak Temperature attribution in Dataset 1, but not a complete explanation**. It helps explain part of the attenuation, yet it does not overturn the broader conclusion that the dominant shift is a dataset-level distribution effect.

---

## 4. Discussion

### 4.1 Why does Temperature dominate in Tai Lake?

A likely explanation is **seasonal co-variation between temperature and chlorination practice** at the Tai Lake treatment plant. In a single plant's operational history, water temperature tracks the seasons — and so does operator behaviour (e.g., higher Cl₂ doses in summer to control algae, or higher bromide inputs during low-flow periods). This means Temperature acts as a strong covariate proxy for multiple process variables that the Tai Lake dataset does not include directly.

When a model is trained on this data, it correctly assigns high importance to Temperature — not because temperature chemically drives DBP formation in isolation, but because it encodes information about the overall seasonal treatment regime. SHAP captures this statistical contribution faithfully.

### 4.2 Why does Temperature rank low in Dataset 1?

Dataset 1 aggregates observations from **multiple treatment facilities** across different geographies and operational schedules. The seasonal temperature–chlorination covariance that exists within a single plant is diluted or absent when the data spans many different operational contexts. Bromide and TOC — chemically more direct precursors to DBP formation — emerge as dominant predictors when the spurious seasonal proxy is removed.

Additionally, Cl₂ dose is measured directly in Dataset 1 and captures the operational signal that Temperature was proxying in the Tai Lake data, further reducing Temperature's apparent importance.

### 4.3 Implications for model generalisation

These findings have practical implications for transfer learning and multi-site DBP modelling:

- SHAP rankings from a single-plant model should not be assumed to generalise to other water systems. Feature importance is a property of the data distribution, not just the chemistry.
- When building models intended for multi-facility deployment, including direct process variables such as Cl₂ dose is important not only for prediction accuracy but also for interpretability — it prevents ambient correlates (Temperature, season) from absorbing disproportionate importance.
- Sub-sampling experiments of the type presented here are a low-cost diagnostic for detecting whether SHAP stability is a sample-size concern. In this case, they confirm it is not, which strengthens confidence in the data-source explanation.

---

## 5. Summary

| Hypothesised driver | Finding |
|---------------------|---------|
| Sample size (175 vs. 488) | **Not supported.** D ≈ B and E ≈ C preserve the same qualitative hierarchy: sub-sampling does not restore Temperature to dominant status. |
| Feature set (presence of Cl₂ dose) | **Partially supported.** Cl₂ dose correctly captures THM variance and enters at rank #1. However, Temperature is already at rank #4 in Dataset 1 without Cl₂, so this is not the root cause. |
| Within-tsid temperature variability | **Contributing, but not sufficient.** In higher-variability Dataset 1 subgroups, Temperature usually rises by about one rank position, especially for DBCM, but still does not recover to rank #1. |
| Data source / data-distribution effect | **Primary driver.** At matched N=175 and identical features, Temperature stays #1 in Tai Lake but drops sharply in Dataset 1 for THM4/BDCM and weakens materially for DBCM. Within this tree-model SHAP protocol, the difference is intrinsic to the datasets rather than to sample size or feature availability alone. |

The dominant SHAP feature shift from Temperature to Cl₂ dose / Bromide reflects a genuine difference in how temperature covaries with DBP formation across different water treatment systems, amplified by the inclusion of direct process variables in Dataset 1 that were absent in the Tai Lake study and by the relatively weak within-series temperature movement in many Dataset 1 `tsid` blocks. Chemistry is a plausible mechanism, but this study isolates a dataset-level contrast rather than a single causal physicochemical pathway.

---

## 6. Methods Summary

| Component | Detail |
|-----------|--------|
| Models | Random Forest, XGBoost (TreeExplainer — exact SHAP) |
| CV folds | 5-fold ensemble per condition |
| Sub-sampling K | 5 independent seeds |
| Sub-sample size | N = 175 (141 train / 34 test, matching Tai Lake split) |
| Hyperparameters | Fixed from full-data Optuna optima (B → D, C → E) |
| SHAP method | shap.TreeExplainer on scaled features, reported back in raw target units and averaged over CV members |
| Outputs | `results/shap_attribution/` and `results/shap_attribution/temp_variability_stratified/` — raw CSV, aggregated CSV, bar charts, ranking heatmaps, and subgroup importance summaries |

---

*Analysis conducted with the SHAP Attribution Study pipeline (`experiments/shap_attribution/`).*
