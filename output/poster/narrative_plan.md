# DBP Prediction Poster Narrative Plan

## Audience
IHE Delft lab/research audience with water-quality, treatment-process, and environmental modeling background.

## Objective
Present the project as a reproducible, interpretable machine-learning workflow for predicting drinking-water disinfection by-products (DBPs), with one clear takeaway: a unified 7-feature backbone works well, but the best model and dominant drivers are target-specific.

## Narrative Arc
1. Problem: DBP formation is hard to predict because it depends on precursor chemistry and disinfection operation.
2. Method: run per-target ML experiments for THM4, BDCM, and DBCM using a shared data/feature/tuning pipeline.
3. Result: the final 7-feature design improves formal performance and selects RF for THM4/BDCM and MLP for DBCM.
4. Interpretation: SHAP shows THM4 is led by Cl2 dose and TOC, while brominated DBPs are led by bromide.
5. Implication: the model is useful not only for prediction, but also for diagnosing which process variables carry actionable signal.

## One-Page Layout
- Header: project title, version/author line, green IHE-style title band.
- Top full-width summary strip: one-sentence project goal and key data footprint.
- Left upper large figure: editable workflow diagram showing data, feature backbone, model tuning, evaluation, and SHAP interpretation.
- Right upper text box: research question and dataset/method bullet points.
- Right middle table: final target-specific winning models and test metrics.
- Left middle text box: key findings and interpretation guardrails.
- Left lower figure: native performance chart with R2 across candidate model families and targets.
- Right lower figure: native SHAP chart using mean absolute SHAP values for the selected 7-feature models.
- Footer: IHE logo, contact box, and compact conclusion line.

## Source Plan
- README.md for project scope, dataset, targets, and model families.
- output/pdf/dataset1_7feat_formal_report.pdf for final performance metrics and 7-feature recommendation.
- output/pdf/dataset1_ablation_cl2_report.pdf for the Cl2 ablation story.
- output/pdf/shap_attribution_study_report.pdf plus results/input_structure summaries for the data-distribution interpretation.
- results/shap_analysis_dataset1_7feat_best/top_features.md and mean_abs_shap_summary.csv for SHAP rankings and values.

## Visual System
- Keep the template's IHE-like green title band, pale peach method panel, pale green finding panel, light gray figure/table panels, and blue footer contact bar.
- Use a restrained scientific palette: dark charcoal text, green accents for the 7-feature backbone, blue for model performance, and amber/red accents for chemical/process drivers.
- Use editable text, native tables, native charts, and simple vector workflow shapes.

## Asset Needs
- Reuse the IHE logo bitmap embedded in the template.
- No generated bitmap imagery is required; the poster is chart/process focused.

## Editability Plan
- All poster copy, tables, workflow labels, and charts remain editable PowerPoint objects.
- Bitmap usage is limited to the IHE logo.
- The source template screenshot is not used as a background image.
