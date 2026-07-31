# AGENTS.md

# Contextual Predictive Maintenance - Team Agent Definitions

## Project Overview

This project aims to build an IoT-based Predictive Maintenance System using Machine Learning and Contextual Data Fusion.

Target KPI:
- Macro F1 Score >= 0.85

Project Duration:
- 4 Weeks

Repository:
- predictive-maintenance-iot

---

# Member 1 - Data Engineer

Role:
Data Engineering and Data Preparation

Responsibilities:
- Download AI4I dataset
- Load dataset into Pandas
- Handle missing values
- Remove duplicate records
- Encode categorical features
- Create cleaned dataset
- Define feature sets
- Run model evaluation metrics

Assigned Issues:
- #2 Download and load AI4I dataset
- #3 Handle missing values and encode Type column
- #9 Ablation study – base vs extended features
- #15 Evaluate Macro F1, Precision, Recall per fold
- #18 Inject Gaussian noise into test dataset

Deliverables:
- data_loader.py
- cleaned dataset
- evaluation outputs

---

# Member 2 - ML Engineer

Role:
Machine Learning Model Development

Responsibilities:
- Perform exploratory data analysis
- Generate visualizations
- Build LightGBM pipeline
- Configure SMOTE
- Tune hyperparameters
- Generate Precision-Recall analysis
- Evaluate model performance

Assigned Issues:
- #4 EDA – class imbalance and correlation heatmap
- #8 Visualize external feature correlations
- #13 SMOTE + LightGBM + StratifiedKFold pipeline
- #14 Tune LightGBM hyperparameters
- #19 Plot Precision-Recall curves

Deliverables:
- week1_eda.ipynb
- model.py
- evaluation plots

---

# Member 3 - Context & Integration

Role:
Feature Engineering and Context Integration

Responsibilities:
- Create rolling statistical features
- Simulate external contextual features
- Merge contextual data
- Perform SHAP explainability
- Optimize classification threshold
- Validate data fusion process

Assigned Issues:
- #5 Rolling mean, std, variance for sensor columns
- #6 Simulate ambient temp, factory load, humidity
- #7 Merge external context with IoT telemetry
- #14 Generate SHAP feature importance plots
- #18 Tune decision threshold

Deliverables:
- feature_engineering.py
- fusion notebooks
- SHAP visualizations

---

# Member 4 - Evaluation & Deployment Lead

Role:
Project Management, Documentation and Deployment

Responsibilities:
- Manage GitHub repository
- Create and track issues
- Maintain Kanban board
- Review project progress
- Compare model results
- Document ablation findings
- Build final dashboard
- Prepare deployment documentation
- Finalize project release

Assigned Issues:
- #1 Initialize repo, .gitignore, Kanban, all 20 Issues
- #10 Document ablation findings
- #17 Compare RF baseline vs LightGBM results
- #21 Build final results dashboard
- #22 README, clean notebooks, tag v1.0.0 release

Deliverables:
- results_comparison.md
- model_results.md
- final_dashboard.ipynb
- final README.md

---

# Workflow

1. Create feature engineering pipeline
2. Train baseline model
3. Add contextual features
4. Perform ablation study
5. Train LightGBM + SMOTE model
6. Generate SHAP explainability
7. Perform robustness analysis
8. Build dashboard
9. Finalize deployment documentation

---

# GitHub Rules

- Reference issue number in every commit
- Use semantic commit messages
- Do not commit data/ directory
- Do not commit models/ directory
- Clear notebook outputs before commit
- Minimum 3–5 commits per active development day

  ## Project Goal

Build an IoT-based Predictive Maintenance System.

Target KPI:
Macro F1 Score >= 0.85


## Ownership Rules

- notebooks/week1_eda.ipynb is owned by Member 2.
- Only Member 2 should edit notebooks/week1_eda.ipynb.
- Other members should avoid editing this file to prevent merge conflicts.

