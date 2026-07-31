"""
=================================================================
 Contextual Predictive Maintenance (IoT Edge AI) — Dashboard
 Infotact Technical Internship Program 2026
 Repo: predictive-maintance-iot  (Tarun Saxena · Vaibhav Gautam)
=================================================================

This dashboard is wired directly to your real project files:

    data/ai4i2020.csv        -> raw AI4I 2020 sensor dataset
    src/retrain.py           -> the feature-engineering + SMOTE +
                                 LightGBM logic this dashboard mirrors
    models/lgbm_retrained.pkl -> the model retrain.py saves
    models/lgbm_pipeline.pkl  -> the model src/model.py saves

WHERE TO PUT THIS FILE
-----------------------
Save as `app.py` in the ROOT of the repo (same level as README.md):

    predictive-maintance-iot/
    ├── app.py              <-- HERE
    ├── data/ai4i2020.csv
    ├── models/
    ├── notebooks/
    ├── src/
    ├── README.md
    └── requirements.txt

HOW TO RUN THE FULL PROJECT
-----------------------------
1) Create/activate a virtual environment and install dependencies:

       cd predictive-maintance-iot
       python -m venv venv
       source venv/bin/activate          # Windows: venv\\Scripts\\activate
       pip install -r requirements.txt
       pip install streamlit plotly       # dashboard-only extras

   NOTE: this repo's requirements.txt was exported as UTF-16 by pip
   freeze on Windows. If `pip install -r requirements.txt` errors
   with a decode error, re-save that file as UTF-8 (open it in
   VS Code -> bottom-right encoding -> "Save with Encoding" -> UTF-8),
   or just run:
       pip install pandas numpy lightgbm imbalanced-learn shap
                   scikit-learn matplotlib seaborn joblib streamlit plotly

2) (Recommended) Train the real model once so the dashboard loads
   your actual trained pipeline instead of training on the fly:

       python src/retrain.py

   This reads data/ai4i2020.csv, engineers features, trains the
   SMOTE + LightGBM pipeline, and saves it to
   models/lgbm_retrained.pkl (models/ and data/ are gitignored,
   which is why they aren't already in the repo).

3) Launch the dashboard:

       streamlit run app.py

   Open the URL Streamlit prints (usually http://localhost:8501).

If you skip step 2, the dashboard still works — it trains an
in-memory copy of the same pipeline on first load (cached), and
shows a banner telling you it's running in that mode.
"""

import os
import re
import glob
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve, auc,
    f1_score, precision_score, recall_score,
)

warnings.filterwarnings("ignore")

try:
    import joblib
except Exception:
    joblib = None

try:
    import shap
except Exception:
    shap = None

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier


# =================================================================
# PROJECT-SPECIFIC CONSTANTS — mirrors src/retrain.py exactly
# =================================================================
DATA_PATH = "data/ai4i2020.csv"
FALLBACK_DATA_GLOBS = ["data/ai4i2020.csv", "**/ai4i2020.csv"]

MODEL_CANDIDATES = [
    "models/lgbm_retrained.pkl",   # saved by src/retrain.py
    "models/lgbm_pipeline.pkl",    # saved by src/model.py
]

RAW_FEATURES = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]
EXTERNAL_CONTEXT_FEATURES = ["ambient_temp_C", "factory_load_pct", "humidity_pct"]
TARGET_COL = "Machine failure"
FAILURE_SUBTYPES = ["TWF", "HDF", "PWF", "OSF", "RNF"]

NOISE_TARGET_F1 = 0.85


def clean_col(c):
    """Same regex src/retrain.py & src/evaluate.py use to clean LightGBM feature names."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", c)


# =================================================================
# PAGE CONFIG + STYLE
# =================================================================
st.set_page_config(
    page_title="Predictive Maintenance IoT | Dashboard",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root{
        --bg-0:#080B10; --bg-1:#0D1219; --panel:#11161E; --panel-alt:#151B24;
        --border:#212A36; --border-soft:#1A222C;
        --text-hi:#EAF0F7; --text-mid:#A9B6C9; --text-dim:#697788;
        --cyan:#2FD9CB; --amber:#FFB020; --danger:#FF5C6C; --success:#3ED598;
    }

    html, body, [class*="css"]{ font-family:'IBM Plex Sans', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header[data-testid="stHeader"]{background: transparent;}

    .stApp{
        background-color: var(--bg-0);
        background-image:
            radial-gradient(1100px 550px at 100% -8%, rgba(47,217,203,0.10), transparent 60%),
            radial-gradient(900px 500px at -5% 8%, rgba(255,176,32,0.07), transparent 55%),
            radial-gradient(1000px 600px at 50% 115%, rgba(47,217,203,0.05), transparent 60%),
            repeating-linear-gradient(0deg, rgba(255,255,255,0.025) 0px, rgba(255,255,255,0.025) 1px, transparent 1px, transparent 64px),
            repeating-linear-gradient(90deg, rgba(255,255,255,0.025) 0px, rgba(255,255,255,0.025) 1px, transparent 1px, transparent 64px),
            linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%);
        background-attachment: fixed;
    }
    .stApp::before{
        content:""; position: fixed; inset:0; pointer-events:none; z-index:0;
        background: radial-gradient(1600px 900px at 50% 0%, transparent 40%, rgba(4,6,9,0.55) 100%);
    }
    .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px; position: relative; z-index: 1;}

    /* ---------- Sidebar : control-panel look ---------- */
    section[data-testid="stSidebar"]{
        background: linear-gradient(180deg, rgba(10,13,19,0.92) 0%, rgba(12,16,23,0.92) 100%);
        backdrop-filter: blur(6px);
        border-right: 1px solid var(--border-soft);
    }
    section[data-testid="stSidebar"] .stRadio label{
        font-family:'JetBrains Mono', monospace; font-size: 0.86rem; color: var(--text-mid);
    }
    section[data-testid="stSidebar"] hr{ border-color: var(--border-soft); }

    /* ---------- Page header (console readout) ---------- */
    .console-header{
        display:flex; align-items:center; gap:14px; padding: 4px 0 14px 0;
        border-bottom: 1px solid var(--border-soft); margin-bottom: 22px;
    }
    .console-icon{
        width:46px; height:46px; min-width:46px; border-radius:10px;
        background: linear-gradient(135deg, rgba(47,217,203,0.16), rgba(47,217,203,0.02));
        border: 1px solid rgba(47,217,203,0.35);
        display:flex; align-items:center; justify-content:center; font-size:1.35rem;
    }
    .console-eyebrow{
        font-family:'JetBrains Mono', monospace; font-size:0.68rem; letter-spacing:0.16em;
        text-transform:uppercase; color: var(--cyan); margin-bottom: 2px;
    }
    .console-title{ font-size:1.55rem; font-weight:700; color: var(--text-hi); line-height:1.25; }
    .console-sub{ font-size:0.86rem; color: var(--text-dim); margin-top:2px; }

    /* ---------- KPI / gauge cards ---------- */
    .kpi-card {
        position: relative; background: linear-gradient(160deg, rgba(17,22,30,0.92) 0%, rgba(21,27,36,0.92) 100%);
        backdrop-filter: blur(4px);
        border: 1px solid var(--border); border-left: 3px solid var(--cyan);
        border-radius: 10px; padding: 16px 18px; box-shadow: 0 6px 18px rgba(0,0,0,0.28);
        transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
    }
    .kpi-card:hover{ transform: translateY(-2px); border-color: rgba(47,217,203,0.55); box-shadow: 0 10px 24px rgba(0,0,0,0.35); }
    .kpi-label {font-family:'JetBrains Mono', monospace; font-size: 0.68rem; text-transform: uppercase;
        letter-spacing: 0.1em; color: var(--text-dim); margin-bottom: 8px;}
    .kpi-value {font-family:'JetBrains Mono', monospace; font-size: 1.9rem; font-weight: 700; color: var(--text-hi);}
    .kpi-sub {font-size: 0.75rem; color: var(--success); margin-top: 5px;}

    /* ---------- Section titles ---------- */
    .section-title {
        font-size: 1.05rem; font-weight: 600; margin-top: 0.6rem; margin-bottom: 0.7rem;
        color: var(--text-hi); display:flex; align-items:center; gap:8px;
    }
    .section-title::before{
        content:""; width:3px; height:16px; background: var(--amber); border-radius:2px; display:inline-block;
    }

    /* ---------- Status badge with LED ---------- */
    .badge {display: inline-flex; align-items:center; gap:6px; padding: 4px 12px; border-radius: 999px;
        font-family:'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.04em;}
    .badge-dot{ width:7px; height:7px; border-radius:50%; }
    .badge-live {background: rgba(62,213,152,0.10); color: #7CF0BE; border: 1px solid rgba(62,213,152,0.35);}
    .badge-live .badge-dot{ background:#3ED598; box-shadow:0 0 8px #3ED598; animation: pulse 1.8s infinite; }
    .badge-demo {background: rgba(255,176,32,0.10); color: #FFCB6B; border: 1px solid rgba(255,176,32,0.35);}
    .badge-demo .badge-dot{ background:#FFB020; box-shadow:0 0 8px #FFB020; }
    @keyframes pulse{ 0%{opacity:1;} 50%{opacity:0.35;} 100%{opacity:1;} }

    /* ---------- Buttons / inputs ---------- */
    .stButton>button, .stFormSubmitButton>button{
        background: linear-gradient(135deg, #1A9E93 0%, #17847B 100%);
        color:#04140F; font-weight:700; border:none; border-radius:8px; letter-spacing:0.02em;
        transition: filter .15s ease;
    }
    .stButton>button:hover, .stFormSubmitButton>button:hover{ filter: brightness(1.12); }
    div[data-baseweb="tab-list"]{ gap: 4px; }
    button[data-baseweb="tab"]{ font-family:'JetBrains Mono', monospace; font-size:0.82rem; }

    ::-webkit-scrollbar{ width:10px; height:10px; }
    ::-webkit-scrollbar-track{ background: var(--bg-0); }
    ::-webkit-scrollbar-thumb{ background: #232C38; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


def console_header(icon: str, title: str, subtitle: str = "", eyebrow: str = "SYSTEM MODULE"):
    """Renders a consistent, styled page header (icon chip + eyebrow + title + subtitle)."""
    sub_html = f'<div class="console-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="console-header">
        <div class="console-icon">{icon}</div>
        <div>
            <div class="console-eyebrow">{eyebrow}</div>
            <div class="console-title">{title}</div>
            {sub_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =================================================================
# DATA LOADING + FEATURE ENGINEERING (mirrors src/retrain.py)
# =================================================================
def find_data_path():
    for p in FALLBACK_DATA_GLOBS:
        matches = glob.glob(p, recursive=True)
        if matches:
            return matches[0]
    return None


@st.cache_data(show_spinner=False)
def load_raw_data():
    path = find_data_path()
    if path is None:
        raise FileNotFoundError(
            "Couldn't find data/ai4i2020.csv. Place app.py in the repo root, "
            "next to the data/ folder."
        )
    df = pd.read_csv(path)
    return df, path


@st.cache_data(show_spinner=False)
def engineer_features(df):
    """Exact logic from src/retrain.py::apply_feature_engineering()."""
    df = df.copy()
    le = LabelEncoder()
    df["Type_enc"] = le.fit_transform(df["Type"])

    np.random.seed(42)
    df["ambient_temp_C"] = np.random.normal(loc=28, scale=5, size=len(df))
    df["factory_load_pct"] = np.random.uniform(50, 100, size=len(df))
    df["humidity_pct"] = np.random.normal(loc=60, scale=10, size=len(df))

    ext_features = RAW_FEATURES + ["Type_enc"] + EXTERNAL_CONTEXT_FEATURES
    X = df[ext_features].copy()
    y = df[TARGET_COL]
    X.columns = [clean_col(c) for c in X.columns]
    return df, X, y, le


@st.cache_resource(show_spinner=False)
def load_or_train_pipeline(_X, _y):
    """Try the real saved model first; otherwise train one live (same recipe as retrain.py)."""
    if joblib is not None:
        for path in MODEL_CANDIDATES:
            if os.path.exists(path):
                try:
                    pipeline = joblib.load(path)
                    return pipeline, path, True
                except Exception:
                    continue

    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, test_size=0.2, random_state=42, stratify=_y
    )
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("lgbm", LGBMClassifier(
            random_state=42, n_jobs=-1, verbose=-1,
            n_estimators=500, learning_rate=0.05,
            num_leaves=31, scale_pos_weight=20,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline, None, False


# =================================================================
# LOAD EVERYTHING
# =================================================================
load_error = None
try:
    raw_df, data_path = load_raw_data()
    full_df, X, y, type_encoder = engineer_features(raw_df)
    pipeline, model_path, model_is_real = load_or_train_pipeline(X, y)
    y_proba = pipeline.predict_proba(X)[:, 1]
except Exception as e:
    load_error = str(e)

if load_error:
    st.error(
        f"**Couldn't load the project data/model.**\n\n{load_error}\n\n"
        "Make sure `app.py` sits in the repo root, next to the `data/` folder, "
        "and that `data/ai4i2020.csv` exists."
    )
    st.stop()


# =================================================================
# SIDEBAR
# =================================================================
st.sidebar.markdown("""
<div style="padding: 6px 0 14px 0; border-bottom: 1px solid #1A222C; margin-bottom: 14px;">
    <div style="font-size: 1.15rem; font-weight: 700; color: #EAF0F7;">🛠️ Predictive Maintenance</div>
    <div style="font-family:'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.08em;
        text-transform: uppercase; color: #697788; margin-top: 4px;">
        Contextual IoT Edge AI · Infotact 2026
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Overview", "🔍 Dataset Explorer", "🎯 Model Performance",
        "🧠 Explainability (SHAP)", "🌊 Noise Robustness",
        "⚡ Live Prediction", "ℹ️ About the Project",
    ],
)

st.sidebar.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
badge_cls = "badge-live" if model_is_real else "badge-demo"
badge_txt = "TRAINED MODEL LOADED" if model_is_real else "TRAINED LIVE THIS SESSION"
st.sidebar.markdown(
    f'<span class="badge {badge_cls}"><span class="badge-dot"></span>{badge_txt}</span>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(f"""
<div style="font-family:'JetBrains Mono', monospace; font-size:0.72rem; color:#7C8A9E;
    margin-top:10px; line-height:1.7;">
    DATA &nbsp;<span style="color:#A9B6C9;">{data_path}</span><br>
    MODEL &nbsp;<span style="color:#A9B6C9;">{model_path or 'in-memory (run src/retrain.py to persist)'}</span>
</div>
""", unsafe_allow_html=True)
if not model_is_real:
    st.sidebar.info("Run `python src/retrain.py` once to save a real model to `models/` "
                     "so the dashboard loads instantly next time.", icon="💡")


# =================================================================
# PAGE: OVERVIEW
# =================================================================
if page == "📊 Overview":
    console_header("📊", "Contextual Predictive Maintenance", eyebrow="OVERVIEW",
                    subtitle="AI-powered predictive maintenance using contextual data fusion & explainable machine learning")

    y_pred_default = (y_proba >= 0.5).astype(int)
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (c1, "Macro F1 Score", f"{f1_score(y, y_pred_default, average='macro'):.4f}", "Target ≥ 0.85"),
        (c2, "Precision", f"{precision_score(y, y_pred_default, zero_division=0):.4f}", "Positive-class"),
        (c3, "Recall", f"{recall_score(y, y_pred_default, zero_division=0):.4f}", "Positive-class"),
        (c4, "Records", f"{len(full_df):,}", f"Failure rate {y.mean()*100:.2f}%"),
    ]
    for col, label, value, sub in kpis:
        with col:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>""",
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown('<div class="section-title">Pipeline</div>', unsafe_allow_html=True)
        stages = ["Raw IoT Data", "Type Encoding", "External Context Fusion",
                   "SMOTE", "LightGBM", "SHAP", "Threshold Tuning", "Prediction"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(stages))), y=[0]*len(stages), mode="lines",
                                  line=dict(color="#2c3a55", width=3), showlegend=False))
        fig.add_trace(go.Scatter(x=list(range(len(stages))), y=[0]*len(stages), mode="markers+text",
                                  text=stages, textposition="top center",
                                  marker=dict(size=20, color="#2FD9CB"), showlegend=False))
        fig.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10),
                           xaxis=dict(visible=False), yaxis=dict(visible=False, range=[-1, 1]),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Class Balance</div>', unsafe_allow_html=True)
        counts = y.value_counts().rename({0: "Healthy", 1: "Failure"})
        fig = px.pie(values=counts.values, names=counts.index, hole=0.55,
                     color=counts.index, color_discrete_map={"Healthy": "#3ED598", "Failure": "#FF5C6C"})
        fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Failure Subtypes (raw dataset flags)</div>', unsafe_allow_html=True)
    present_subtypes = [c for c in FAILURE_SUBTYPES if c in full_df.columns]
    if present_subtypes:
        sub_counts = full_df[present_subtypes].sum().sort_values(ascending=True)
        fig = px.bar(sub_counts, orientation="h", labels={"value": "Count", "index": "Failure type"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=280, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Technology Stack</div>', unsafe_allow_html=True)
    st.write("`Python` · `Pandas`/`NumPy` · `LightGBM` · `Scikit-Learn` · "
             "`Imbalanced-Learn (SMOTE)` · `SHAP` · `Streamlit` · `Plotly`")


# =================================================================
# PAGE: DATASET EXPLORER
# =================================================================
elif page == "🔍 Dataset Explorer":
    console_header("🔍", "Dataset Explorer", eyebrow="DATA",
                    subtitle=f"Source: {data_path} · Raw shape: {raw_df.shape[0]:,} rows × {raw_df.shape[1]} cols "
                             f"· Engineered features: {X.shape[1]}")

    tab1, tab2, tab3, tab4 = st.tabs(["Raw Data", "Engineered Features", "Sensor Distributions", "Correlation"])

    with tab1:
        st.dataframe(raw_df.head(200), use_container_width=True)
        st.download_button("Download raw CSV", raw_df.to_csv(index=False).encode(), "ai4i2020_preview.csv")

    with tab2:
        st.dataframe(pd.concat([X, y], axis=1).head(200), use_container_width=True)
        st.caption("Type_enc + external context columns simulated with `np.random.seed(42)`, "
                   "exactly as in `src/retrain.py`.")

    with tab3:
        chosen = st.selectbox("Feature", RAW_FEATURES + EXTERNAL_CONTEXT_FEATURES, index=0)
        fig = px.histogram(full_df, x=chosen, color=full_df[TARGET_COL].map({0: "Healthy", 1: "Failure"}),
                            barmode="overlay", opacity=0.7, nbins=40,
                            color_discrete_map={"Healthy": "#3ED598", "Failure": "#FF5C6C"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", legend_title="Status")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        corr_df = pd.concat([X, y.rename(TARGET_COL)], axis=1)
        corr = corr_df.corr()
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=550)
        st.plotly_chart(fig, use_container_width=True)


# =================================================================
# PAGE: MODEL PERFORMANCE
# =================================================================
elif page == "🎯 Model Performance":
    console_header("🎯", "Model Performance", eyebrow="EVALUATION",
                    subtitle="SMOTE + LightGBM pipeline, evaluated on the full engineered dataset")

    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
    y_pred = (y_proba >= threshold).astype(int)

    c1, c2, c3 = st.columns(3)
    c1.metric("Macro F1", f"{f1_score(y, y_pred, average='macro'):.4f}")
    c2.metric("Precision", f"{precision_score(y, y_pred, zero_division=0):.4f}")
    c3.metric("Recall", f"{recall_score(y, y_pred, zero_division=0):.4f}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y, y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                         labels=dict(x="Predicted", y="Actual"),
                         x=["Healthy", "Failure"], y=["Healthy", "Failure"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.markdown('<div class="section-title">Precision–Recall Curve</div>', unsafe_allow_html=True)
        prec, rec, _ = precision_recall_curve(y, y_proba)
        fig = px.area(x=rec, y=prec, labels={"x": "Recall", "y": "Precision"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">ROC Curve</div>', unsafe_allow_html=True)
    fpr, tpr, _ = roc_curve(y, y_proba)
    fig = px.area(x=fpr, y=tpr, labels={"x": "False Positive Rate", "y": "True Positive Rate"},
                  title=f"AUC = {auc(fpr, tpr):.4f}")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    lgbm_model = pipeline.named_steps.get("lgbm", None)
    if lgbm_model is not None and hasattr(lgbm_model, "feature_importances_"):
        st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
        imp = pd.Series(lgbm_model.feature_importances_, index=X.columns).sort_values(ascending=True)
        fig = px.bar(imp, orientation="h")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# =================================================================
# PAGE: SHAP EXPLAINABILITY
# =================================================================
elif page == "🧠 Explainability (SHAP)":
    console_header("🧠", "Explainability — SHAP", eyebrow="INTERPRETABILITY",
                    subtitle="Which sensors and contextual features drive the model's failure predictions")

    lgbm_model = pipeline.named_steps.get("lgbm", pipeline)
    if shap is None:
        st.warning("Install `shap` (`pip install shap`) to see live SHAP plots here.")
    else:
        with st.spinner("Computing SHAP values (sampled for speed)..."):
            sample = X.sample(min(800, len(X)), random_state=42)
            try:
                explainer = shap.TreeExplainer(lgbm_model)
                sv = explainer.shap_values(sample)
                sv = sv[1] if isinstance(sv, list) else sv
                imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=True)

                st.markdown('<div class="section-title">Global Feature Importance (mean |SHAP|)</div>', unsafe_allow_html=True)
                fig = px.bar(imp, orientation="h")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=450, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown('<div class="section-title">Dependence Plot</div>', unsafe_allow_html=True)
                feat = st.selectbox("Feature", list(X.columns), index=list(X.columns).index(imp.index[-1]))
                fidx = list(X.columns).index(feat)
                fig2 = px.scatter(x=sample[feat], y=sv[:, fidx], labels={"x": feat, "y": "SHAP value"},
                                   color=sample[feat], color_continuous_scale="RdBu_r")
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=380)
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Couldn't compute SHAP values: {e}")


# =================================================================
# PAGE: NOISE ROBUSTNESS
# =================================================================
elif page == "🌊 Noise Robustness":
    console_header("🌊", "Noise Sensitivity Analysis", eyebrow="ROBUSTNESS",
                    subtitle="Gaussian noise (mean 0, std = noise level × feature std) injected into sensor "
                             "readings to simulate real-world signal drift — same idea as the Week 4 robustness study")

    noise_levels = [0.0, 0.05, 0.15, 0.30]
    rows = []
    rng = np.random.default_rng(0)
    stds = X.std()
    for nl in noise_levels:
        Xn = X.copy()
        if nl > 0:
            Xn = Xn + rng.normal(0, nl, Xn.shape) * stds.values
        proba_n = pipeline.predict_proba(Xn)[:, 1]
        f1_n = f1_score(y, (proba_n >= 0.5).astype(int), average="macro")
        rows.append({"Noise Level (σ)": nl, "Macro F1": f1_n})
    noise_df = pd.DataFrame(rows)
    noise_df["% Drop vs Clean"] = (noise_df["Macro F1"].iloc[0] - noise_df["Macro F1"]) / noise_df["Macro F1"].iloc[0] * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(x=noise_df["Noise Level (σ)"].astype(str), y=noise_df["Macro F1"], marker_color="#2FD9CB"))
    fig.add_hline(y=NOISE_TARGET_F1, line_dash="dash", line_color="#FF5C6C", annotation_text="Target F1 = 0.85")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=420,
                       xaxis_title="Noise Level (σ)", yaxis_title="Macro F1 Score")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(noise_df.style.format({"Macro F1": "{:.4f}", "% Drop vs Clean": "{:.2f}%"}), use_container_width=True)

    st.markdown('<div class="section-title">Threshold Sweep (0.10 – 0.90)</div>', unsafe_allow_html=True)
    thresholds = np.arange(0.10, 0.95, 0.05)
    sweep = [{
        "Threshold": t,
        "Precision": precision_score(y, (y_proba >= t).astype(int), zero_division=0),
        "Recall": recall_score(y, (y_proba >= t).astype(int), zero_division=0),
        "F1": f1_score(y, (y_proba >= t).astype(int), average="macro"),
    } for t in thresholds]
    sweep_df = pd.DataFrame(sweep)
    fig2 = px.line(sweep_df, x="Threshold", y=["Precision", "Recall", "F1"])
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=380)
    st.plotly_chart(fig2, use_container_width=True)
    best_row = sweep_df.loc[sweep_df["F1"].idxmax()]
    st.success(f"Best F1 at threshold ≈ **{best_row['Threshold']:.2f}** (F1 = {best_row['F1']:.4f})")


# =================================================================
# PAGE: LIVE PREDICTION
# =================================================================
elif page == "⚡ Live Prediction":
    console_header("⚡", "Live Failure Prediction", eyebrow="INFERENCE",
                    subtitle="Enter current sensor + contextual readings to get a failure-risk estimate from the real pipeline")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            machine_type = st.selectbox("Machine Type", sorted(raw_df["Type"].unique().tolist()))
            air_temp = st.slider("Air temperature [K]", float(raw_df["Air temperature [K]"].min()),
                                  float(raw_df["Air temperature [K]"].max()), float(raw_df["Air temperature [K]"].median()))
            process_temp = st.slider("Process temperature [K]", float(raw_df["Process temperature [K]"].min()),
                                      float(raw_df["Process temperature [K]"].max()), float(raw_df["Process temperature [K]"].median()))
        with c2:
            rot_speed = st.slider("Rotational speed [rpm]", float(raw_df["Rotational speed [rpm]"].min()),
                                   float(raw_df["Rotational speed [rpm]"].max()), float(raw_df["Rotational speed [rpm]"].median()))
            torque = st.slider("Torque [Nm]", float(raw_df["Torque [Nm]"].min()),
                                float(raw_df["Torque [Nm]"].max()), float(raw_df["Torque [Nm]"].median()))
            tool_wear = st.slider("Tool wear [min]", float(raw_df["Tool wear [min]"].min()),
                                   float(raw_df["Tool wear [min]"].max()), float(raw_df["Tool wear [min]"].median()))
        with c3:
            ambient_temp = st.slider("Ambient temperature [°C]", 10.0, 45.0, 28.0)
            factory_load = st.slider("Factory load [%]", 50.0, 100.0, 75.0)
            humidity = st.slider("Humidity [%]", 20.0, 90.0, 60.0)

        submitted = st.form_submit_button("Predict Failure Risk", use_container_width=True)

    if submitted:
        type_enc_val = int(type_encoder.transform([machine_type])[0])
        row = {
            "Air temperature [K]": air_temp, "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rot_speed, "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear, "Type_enc": type_enc_val,
            "ambient_temp_C": ambient_temp, "factory_load_pct": factory_load,
            "humidity_pct": humidity,
        }
        x_new = pd.DataFrame([row])
        x_new.columns = [clean_col(c) for c in x_new.columns]
        x_new = x_new[X.columns]  # enforce training column order

        proba = float(pipeline.predict_proba(x_new)[0, 1])

        c1, c2 = st.columns([1, 1.4])
        with c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=proba * 100, number={"suffix": "%"},
                title={"text": "Failure Probability"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#FF5C6C" if proba > 0.5 else "#3ED598"},
                       "steps": [{"range": [0, 50], "color": "#123d2a"}, {"range": [50, 100], "color": "#3d1212"}]},
            ))
            fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", font_color="#f2f5fb")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if proba > 0.5:
                verdict, vcolor, vbg = "⚠️ High Risk — schedule maintenance", "#FF5C6C", "rgba(255,92,108,0.10)"
            else:
                verdict, vcolor, vbg = "✅ Healthy — no action needed", "#3ED598", "rgba(62,213,152,0.10)"
            st.markdown(f"""<div style="background:{vbg}; border:1px solid {vcolor}44;
                border-left:3px solid {vcolor}; border-radius:10px; padding:14px 18px;
                font-weight:700; color:{vcolor}; font-size:1.05rem; margin-bottom:14px;">{verdict}</div>""",
                unsafe_allow_html=True)
            if shap is not None:
                try:
                    lgbm_model = pipeline.named_steps.get("lgbm", pipeline)
                    explainer = shap.TreeExplainer(lgbm_model)
                    sv = explainer.shap_values(x_new)
                    sv = sv[1] if isinstance(sv, list) else sv
                    contrib = pd.Series(sv[0], index=X.columns).sort_values()
                    fig2 = px.bar(contrib, orientation="h", title="Feature contribution to this prediction")
                    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350, showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception:
                    st.caption("SHAP explanation unavailable for this input.")


# =================================================================
# PAGE: ABOUT
# =================================================================
else:
    console_header("ℹ️", "About This Project", eyebrow="REFERENCE",
                    subtitle="Infotact Technical Internship Program — Advanced Data Science & Machine Learning (2026)")

    st.markdown("""
This system predicts industrial equipment failures before they occur by fusing
internal IoT sensor telemetry (air/process temperature, rotational speed, torque,
tool wear) with simulated external contextual signals (ambient temperature,
factory load, humidity).
""")

    st.markdown('<div class="section-title">Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
`data/ai4i2020.csv` → encode `Type` → simulate external context → SMOTE (train fold only)
→ LightGBM classifier → evaluate on a held-out test split → save to `models/`
""")

    st.markdown('<div class="section-title">This Session</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    for col, label, value, sub in [
        (sc1, "Dataset", f"{len(full_df):,} rows", data_path),
        (sc2, "Model Source", "Trained model" if model_is_real else "Trained live", model_path or "in-memory session"),
        (sc3, "Failure Rate", f"{y.mean()*100:.2f}%", "of records flagged"),
    ]:
        with col:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
                <div class="kpi-value" style="font-size:1.3rem;">{value}</div>
                <div class="kpi-sub" style="color:#7C8A9E;">{sub}</div></div>""",
                unsafe_allow_html=True)

    st.markdown('<div class="section-title">Team</div>', unsafe_allow_html=True)
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("""<div class="kpi-card"><div class="kpi-label">Data Engineer & Evaluation/Deployment Lead</div>
            <div class="kpi-value" style="font-size:1.2rem;">Tarun Saxena</div></div>""", unsafe_allow_html=True)
    with tc2:
        st.markdown("""<div class="kpi-card"><div class="kpi-label">ML Engineer & Context Integration Lead</div>
            <div class="kpi-value" style="font-size:1.2rem;">Vaibhav Gautam</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Tech Stack</div>', unsafe_allow_html=True)
    st.write("`Python` · `Pandas`/`NumPy` · `LightGBM` · `Scikit-Learn` · "
             "`Imbalanced-Learn (SMOTE)` · `SHAP` · `Streamlit` · `Plotly`")

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"Repository: github.com/tarunsaxena2/predictive-maintance-iot · "
               f"Dashboard rendered {datetime.now().strftime('%Y-%m-%d %H:%M')}")