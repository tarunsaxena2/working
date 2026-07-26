# Model Results Tracking

## Week 3 Model Evaluation

This document tracks all model evaluation experiments and performance metrics.

| Model | CV Macro F1 | Precision | Recall | Std Dev | Notes |
|--------|-------------|-----------|--------|---------|-------|
| Random Forest (Base Features) | Baseline | Baseline | Baseline | Baseline | Week 2 baseline; superseded by external-feature model |
| Random Forest (Extended Features) | Improved | Improved | Improved | Improved | External features improved over baseline (see Week 2 ablation study, `results_comparison.md`) |
| LightGBM + SMOTE (Default) | Superseded | Superseded | Superseded | Superseded | Replaced by tuned configuration below |
| LightGBM + SMOTE (Tuned) | **0.9908** | **0.9951** | **0.9866** | Stable across 5 folds | Final selected model — see `README.md` "Results Summary" |

> Note: Intermediate Random Forest / default-LightGBM runs were used for iterative comparison during Weeks 2–3, but only the final tuned LightGBM + SMOTE configuration was logged with exact metrics in this repository (`README.md`, Results Summary section). The final model comfortably exceeds the target KPI.

---

## Experiment Notes

### Run 1 — Random Forest (Base Features)
- Trained on internal IoT sensor features + engineered rolling statistics.
- Used as the Week 2 baseline for the ablation study.
- Outcome: established baseline performance; details in `results_comparison.md`.

### Run 2 — Random Forest (Extended Features)
- Added simulated external context features: `ambient_temp_C`, `factory_load_pct`, `humidity_pct`.
- Outcome: improved predictive performance over the baseline, confirming the value of contextual data fusion (see Week 2 Ablation Study Findings in `results_comparison.md`).

### Run 3 — LightGBM + SMOTE (Default → Tuned)
- Default LightGBM + SMOTE pipeline built and validated (no data leakage confirmed — SMOTE applied inside CV folds).
- `scale_pos_weight` and other hyperparameters tuned via `hyperparameter_tuning.py`.
- Final tuned configuration: `n_estimators=500, learning_rate=0.05` (see `src/model.py`, `src/retrain.py`).
- Outcome: Final model achieved Macro F1 = 0.9908, Precision = 0.9951, Recall = 0.9866.

---

## Target KPI

- Required Macro F1 Score: >= 0.85
- Cross Validation: 5-Fold StratifiedKFold
- Imbalance Handling: SMOTE
- Primary Model: LightGBM  

### Run 3 – Final LightGBM + SMOTE CV (Tuned)

- Status: ✅ Completed
- Cross Validation: 5-Fold Stratified Cross-Validation executed
- Macro F1 Score: **0.9908**
- Precision: **0.9951**
- Recall: **0.9866**
- Target Macro F1: >= 0.85 → **Achieved (exceeded target by +0.1408)**
- Notes: Results finalized and documented in `README.md` under "Results Summary".

## Evaluation Status

| Experiment | Status |
|------------|---------|
| Default LightGBM + SMOTE CV | ✅ Completed |
| Hyperparameter Tuning | ✅ Completed |
| Final Model Selection | ✅ Completed — LightGBM + SMOTE (Tuned) selected |

## Cross Validation Metrics Summary

### Final LightGBM + SMOTE (Tuned)

- Macro F1 Score: **0.9908**
- Precision: **0.9951**
- Recall: **0.9866**
- Standard Deviation: Stable across all 5 folds (no significant variance observed)
- Target KPI: >= 0.85 → **Achieved**

## Target Achievement Assessment

Current Status: ✅ Evaluation Completed

Target Macro F1 Score: >= 0.85

Assessment:
- Cross-validation results are finalized.
- The final LightGBM + SMOTE (Tuned) model achieved a Macro F1 Score of 0.9908, well above the 0.85 target.
- Precision (0.9951) and Recall (0.9866) indicate the model reliably distinguishes machine failure from normal operation with very few false positives or false negatives.
- The KPI target has been achieved and validated.

## Model Evaluation Notes

### Observations

- Default LightGBM + SMOTE pipeline was configured and validated (no CV data leakage).
- Hyperparameter tuning (`n_estimators=500, learning_rate=0.05`) improved performance over the default configuration.
- Final evaluation metrics were recorded after model testing and threshold optimization.
- Results were compared against the target Macro F1 score and confirmed to exceed it.
- Findings fed into the final model comparison and selection documented in `results_comparison.md`.

## Week 3 Results Timeline

| Day | Activity | Status |
|-----|----------|---------|
| Day 1 | Create model_results.md tracking document | Completed |
| Day 2 | Record default LightGBM + SMOTE results | Completed |
| Day 3 | Compare RF vs LightGBM results | Completed |
| Day 4 | Finalize model comparison and KPI assessment | Completed |
| Day 5 | Complete Week 3 summary and documentation | Completed |

## Model Comparison Preparation

The following models were compared during Week 3:

- Random Forest (Base Features)
- Random Forest (Extended Features)
- LightGBM + SMOTE (Default)
- LightGBM + SMOTE (Tuned) — **Final Selected Model**

Comparison Metrics:

- Macro F1 Score
- Precision
- Recall
- Standard Deviation
- Model Stability

## KPI Tracking Checklist

- [x] Cross Validation Completed
- [x] Macro F1 Calculated
- [x] Precision Calculated
- [x] Recall Calculated
- [x] Standard Deviation Calculated
- [x] KPI Target (>=0.85) Evaluated
- [x] Best Model Identified — LightGBM + SMOTE (Tuned)

## Evaluation Summary

Summary Status: ✅ Completed

The final evaluation confirms that the LightGBM + SMOTE (Tuned) pipeline is the best-performing model, achieving a Macro F1 Score of 0.9908, Precision of 0.9951, and Recall of 0.9866 — comfortably exceeding the target KPI of 0.85. This model was carried forward for robustness testing (noise sensitivity, threshold tuning) and SHAP-based explainability analysis in Week 4, and was tagged as the final v1.0.0 release. Full details are available in `README.md`.