# Model Results — Final Confirmed Metrics

## Best Model Configuration
| Parameter | Value |
|---|---|
| Model | LightGBM + SMOTE |
| n_estimators | 500 |
| learning_rate | 0.1 |
| num_leaves | 15 |
| scale_pos_weight | Removed |
| CV Strategy | 5-Fold Stratified |

## Final Metrics (Holdout Test Set)
| Metric | Value | Status |
|---|---|---|
| Macro F1 | 0.8501 | ✅ KPI Met (≥0.85) |
| Precision | 0.8233 | ✅ |
| Recall | 0.8825 | ✅ |

## Notes
- SMOTE applied only inside training folds — no data leakage
- External context features included (ambient_temp_C, factory_load_pct, humidity_pct)
- scale_pos_weight removed after tuning — SMOTE alone handles imbalance sufficiently
- Metrics verified by both Tarun Saxena and Vaibhav Gautam — confirmed match ✅
