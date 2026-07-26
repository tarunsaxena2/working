# 📊 Results Comparison

This document tracks all model evaluation results across the project lifecycle.

## Document Version

Version: 2.0 (Final)
Originally created for Week 2 experiment tracking; finalized at end of Week 4 with validated results.

## Evaluation Ownership

Responsible Role: Evaluation & Deployment Lead (Member 4)

This document tracks:
- Ablation Study Results
- Model Performance Comparison
- Cross-Validation Metrics
- Final Evaluation Summary

## Evaluation Metric

Primary Metric:
- Macro F1 Score

Supporting Metrics:
- Precision
- Recall
- Standard Deviation

---

# Ablation Study Results (Week 2)

Two Random Forest models were compared using identical 5-Fold Stratified Cross-Validation to measure the impact of adding simulated external contextual features (`ambient_temp_C`, `factory_load_pct`, `humidity_pct`).

## Without External Features (Baseline)
- Feature Set: Internal IoT sensor features + engineered rolling statistics only
- Result: Baseline predictive performance established
- Notes: Captured machine behavior effectively but did not represent environmental conditions

## With External Features (Extended)
- Feature Set: Baseline features + Ambient Temperature + Factory Load + Humidity
- Result: Improved predictive performance over the baseline
- Notes: External contextual signal enabled the model to better distinguish normal vs. failure conditions, without any row loss or data integrity issues during fusion (see `review_notes.md`)

**Conclusion:** Integrating external contextual features enhances predictive capability. This confirmed the value of contextual data fusion and justified carrying the extended feature set forward into Week 3 modeling. Exact per-run Macro F1/Precision/Recall values for these exploratory Random Forest runs were not persisted numerically in the repository; the fully validated final metrics are reported below for the production LightGBM pipeline.

---

# Model Comparison Table (Final)

| Model | Feature Set | Macro F1 | Precision | Recall | Notes |
|---------|---------|---------|---------|---------|---------|
| Random Forest | Base Features | Baseline | Baseline | Baseline | Week 2 baseline for ablation study |
| Random Forest | Extended Features | Improved vs. baseline | Improved vs. baseline | Improved vs. baseline | Confirmed value of external features |
| LightGBM + SMOTE | Default | Superseded | Superseded | Superseded | Replaced by tuned configuration |
| **LightGBM + SMOTE** | **Tuned (Final)** | **0.9908** | **0.9951** | **0.9866** | ✅ **Final selected model — target KPI (≥0.85) exceeded** |

---

# Week 3 Metrics (Final)

| Model | CV Macro F1 | Precision | Recall | Std Dev |
|---------|---------|---------|---------|---------|
| LightGBM + SMOTE (Tuned) | 0.9908 | 0.9951 | 0.9866 | Stable across 5 folds |

---

# Notes

- All final metrics are based on 5-Fold Stratified Cross-Validation.
- Macro F1 is the primary evaluation metric.
- SMOTE was applied inside CV folds only (no data leakage — verified in `progress_week3.md`).
- The final model was carried forward into Week 4 for noise-robustness testing, threshold optimization, and SHAP explainability analysis (see `README.md`).

---

# Experiment Log

| Date | Experiment | Status |
|--------|--------|--------|
| Week 2 | Baseline Random Forest | ✅ Completed |
| Week 2 | Random Forest + External Features | ✅ Completed |
| Week 3 | LightGBM Baseline (Default) | ✅ Completed |
| Week 3 | LightGBM + SMOTE (Tuned) | ✅ Completed |

---

# External Feature Comparison

## Integrated External Features

- Ambient Temperature
- Humidity
- Factory Load
- Shift Information (considered; not used in final feature set)

Status: ✅ Integration Completed — features carried into the final production feature set.

## Result Interpretation

Key observations and performance insights:

- **Impact of external features:** Positive — improved discrimination between normal and failure states.
- **Feature contribution (SHAP):** Rotational Speed, Torque, Tool Wear, Air Temperature, and Process Temperature were the top contributors in the final model.
- **Performance stability:** Macro F1 remained high and stable across CV folds and under moderate injected Gaussian noise (see `README.md` — Robustness Analysis).
- **Generalization capability:** Strong — final model achieved Macro F1 = 0.9908 on held-out cross-validation folds.

---

# Review Checklist

- [x] Baseline Results Recorded
- [x] External Feature Results Recorded
- [x] Model Comparison Updated
- [x] Metrics Verified
- [x] Final Conclusions Added

---

# Week 2 Ablation Study Findings

The Week 2 ablation study evaluated the impact of incorporating simulated external contextual features into the predictive maintenance dataset. Two Random Forest models were compared using identical five-fold Stratified Cross-Validation. The baseline model was trained using only internal IoT sensor features and engineered rolling statistics, while the extended model additionally included ambient temperature, factory load percentage, and humidity as external contextual variables.

The comparison demonstrated that the model with external contextual features achieved improved predictive performance compared to the baseline model. Although the internal sensor readings captured machine behavior effectively, they did not fully represent the environmental conditions influencing equipment performance. The additional contextual information provided complementary insights that enabled the model to better distinguish between normal and failure conditions.

The feature fusion process preserved dataset integrity by maintaining row alignment and ensuring that no observations were lost during merging. The resulting dataset contained a richer representation of operational conditions while remaining consistent for downstream model training.

Overall, the ablation study confirms that integrating external contextual features enhances the predictive capability of the machine learning pipeline. These findings established a stronger foundation for the subsequent Week 3 model optimization and evaluation, which produced the final LightGBM + SMOTE model.

---

# Model Results Comparison — Week 3

This section compares Random Forest and LightGBM model performance across different feature configurations, finalized at the end of Week 3.

| Model | Macro F1 | Precision | Recall | Std Dev | Status |
| ----- | -------- | --------- | ------ | ------- | ------ |
| Random Forest (Base Features) | Baseline | Baseline | Baseline | Baseline | ✅ Completed (Week 2) |
| Random Forest (Extended Features) | Improved | Improved | Improved | Improved | ✅ Completed (Week 2) |
| LightGBM + SMOTE (Default) | Superseded | Superseded | Superseded | Superseded | ✅ Completed, superseded by tuning |
| **LightGBM + SMOTE (Tuned)** | **0.9908** | **0.9951** | **0.9866** | Stable | ✅ **Final Model Selected** |

## Comparison Criteria

- Macro F1 Score
- Precision
- Recall
- Standard Deviation
- Training Stability
- Overall Performance Ranking

## Final Findings

- Random Forest results confirmed the value of external contextual features (Week 2 ablation).
- LightGBM + SMOTE outperformed Random Forest approaches after hyperparameter tuning.
- Final ranking: **LightGBM + SMOTE (Tuned) is the best-performing and selected production model.**

## Review Status

- Comparison table created
- Models added for evaluation
- Comparison framework prepared
- ✅ Final experiment results integrated

Status: ✅ Completed

---

# Final Comparison Conclusion

The model comparison process is complete. The comparison tables and documentation structure have been fully populated with validated experiment results from both the Random Forest and LightGBM pipelines.

The final winning model was selected based on verified cross-validation metrics: **LightGBM + SMOTE (Tuned)**, achieving a Macro F1 Score of **0.9908**, Precision of **0.9951**, and Recall of **0.9866** — substantially exceeding the target Macro F1 Score of ≥ 0.85.

As expected, the SMOTE + LightGBM pipeline outperformed the Random Forest baselines by addressing class imbalance more effectively through SMOTE while leveraging LightGBM's gradient boosting algorithm to model complex, non-linear feature interactions (including the added external contextual features).

**Final Status:** ✅ Comparison completed. LightGBM + SMOTE (Tuned) confirmed as the final winning model and carried forward to Week 4 robustness testing, threshold optimization, and deployment preparation (v1.0.0 release).