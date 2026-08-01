\# Demo Presentation Notes

\## Contextual Predictive Maintenance — IoT Edge AI

\## Vaibhav Gautam — Dashboard \& Integration



\---



\## Slide 1: Project Overview

\- Problem: Machines fail unexpectedly — costly downtime

\- Solution: AI-powered predictive maintenance using IoT sensors + external context

\- Team: Tarun Saxena (Model \& Backend) + Vaibhav Gautam (Dashboard \& Integration)

\- KPI: Macro F1 ≥ 0.85 — \*\*ACHIEVED: 0.8501\*\* ✅



\---



\## Slide 2: Architecture



ESP32/CSV → simulate\_stream.py → api.py (/predict) → Dashboard

&#x20;                          ↓

&#x20;                   LightGBM + SMOTE

&#x20;                          ↓

&#x20;                   SHAP Explanation



\---



\## Slide 3: Dashboard Demo Flow

1\. Open Overview page — show KPI cards (Macro F1 = 0.8501)

2\. Open Dataset Explorer — show class imbalance (28.5:1)

3\. Open Model Performance — show confusion matrix + PR curve

4\. Open SHAP Explainability — show feature importance

5\. Open Noise Robustness — show model holds up under noise

6\. Open Live Monitoring — demo real-time prediction



\---



\## Slide 4: Live Monitoring Demo

Steps to demo:

1\. Start API: `python -m uvicorn api:app --reload`

2\. Start simulator: `python simulate\_stream.py`

3\. Show API ONLINE badge in dashboard

4\. Send manual prediction — show verdict banner + gauge

5\. Show SHAP explanation per prediction

6\. Enable Auto Refresh — show live updates



\---



\## Slide 5: Results Summary

| Metric | Value | Status |

|---|---|---|

| Macro F1 | 0.8501 | ✅ KPI Met |

| Precision | 0.8233 | ✅ |

| Recall | 0.8825 | ✅ |

| API Response | < 100ms | ✅ |

| Noise Robustness | Maintained under low/medium noise | ✅ |



\---



\## Slide 6: Next Steps (Part 2)

\- Swap simulate\_stream.py for real ESP32 sensor feed

\- Deploy API to cloud (AWS/GCP)

\- Add real-time database logging

\- Mobile alert notifications

