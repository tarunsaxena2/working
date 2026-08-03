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
        --bg-0:#F4F6FA; --bg-1:#EAEFF7; --panel:#FFFFFF; --panel-alt:#F0F4FB;
        --border:#D1D9E6; --border-soft:#E2E8F0;
        --text-hi:#1A202C; --text-mid:#4A5568; --text-dim:#718096;
        --cyan:#0694A2; --amber:#D97706; --danger:#DC2626; --success:#059669;
    }

    html, body, [class*="css"]{ font-family:'IBM Plex Sans', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header[data-testid="stHeader"]{background: transparent;}

    .stApp{
        background-color: var(--bg-0);
        background-image:
            radial-gradient(1100px 550px at 100% -8%, rgba(6,148,162,0.06), transparent 60%),
            radial-gradient(900px 500px at -5% 8%, rgba(217,119,6,0.04), transparent 55%),
            linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%);
        background-attachment: fixed;
    }

    .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px; position: relative; z-index: 1;}

    /* Streamlit default text fix */
    p, li, span, label, div { color: var(--text-hi) !important; }
    h1, h2, h3, h4 { color: var(--text-hi) !important; }
    .stMarkdown { color: var(--text-hi) !important; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"]{
        background: linear-gradient(180deg, #FFFFFF 0%, #F4F6FA 100%);
        border-right: 1px solid var(--border-soft);
    }
    section[data-testid="stSidebar"] .stRadio label{
        font-family:'JetBrains Mono', monospace; font-size: 0.86rem; color: var(--text-mid);
    }
    section[data-testid="stSidebar"] hr{ border-color: var(--border-soft); }

    /* ---------- Page header ---------- */
    .console-header{
        display:flex; align-items:center; gap:14px; padding: 4px 0 14px 0;
        border-bottom: 1px solid var(--border-soft); margin-bottom: 22px;
    }
    .console-icon{
        width:46px; height:46px; min-width:46px; border-radius:10px;
        background: linear-gradient(135deg, rgba(6,148,162,0.12), rgba(6,148,162,0.04));
        border: 1px solid rgba(6,148,162,0.30);
        display:flex; align-items:center; justify-content:center; font-size:1.35rem;
    }
    .console-eyebrow{
        font-family:'JetBrains Mono', monospace; font-size:0.68rem; letter-spacing:0.16em;
        text-transform:uppercase; color: var(--cyan); margin-bottom: 2px;
    }
    .console-title{ font-size:1.55rem; font-weight:700; color: var(--text-hi) !important; line-height:1.25; }
    .console-sub{ font-size:0.86rem; color: var(--text-dim) !important; margin-top:2px; }

    /* ---------- KPI cards ---------- */
    .kpi-card {
        position: relative;
        background: #FFFFFF;
        border: 1px solid var(--border); border-left: 3px solid var(--cyan);
        border-radius: 10px; padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
    }
    .kpi-card:hover{ transform: translateY(-2px); border-color: var(--cyan); box-shadow: 0 6px 16px rgba(0,0,0,0.10); }
    .kpi-label {font-family:'JetBrains Mono', monospace; font-size: 0.68rem; text-transform: uppercase;
        letter-spacing: 0.1em; color: var(--text-dim) !important; margin-bottom: 8px;}
    .kpi-value {font-family:'JetBrains Mono', monospace; font-size: 1.9rem; font-weight: 700; color: var(--text-hi) !important;}
    .kpi-sub {font-size: 0.75rem; color: var(--success) !important; margin-top: 5px;}

    /* ---------- Section titles ---------- */
    .section-title {
        font-size: 1.05rem; font-weight: 600; margin-top: 0.6rem; margin-bottom: 0.7rem;
        color: var(--text-hi) !important; display:flex; align-items:center; gap:8px;
    }
    .section-title::before{
        content:""; width:3px; height:16px; background: var(--amber); border-radius:2px; display:inline-block;
    }

    /* ---------- Badges ---------- */
    .badge {display: inline-flex; align-items:center; gap:6px; padding: 4px 12px; border-radius: 999px;
        font-family:'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.04em;}
    .badge-dot{ width:7px; height:7px; border-radius:50%; }
    .badge-live {background: rgba(5,150,105,0.10); color: #065F46 !important; border: 1px solid rgba(5,150,105,0.35);}
    .badge-live .badge-dot{ background:#059669; box-shadow:0 0 8px #059669; animation: pulse 1.8s infinite; }
    .badge-demo {background: rgba(217,119,6,0.10); color: #92400E !important; border: 1px solid rgba(217,119,6,0.35);}
    .badge-demo .badge-dot{ background:#D97706; }
    .badge-blue {background: rgba(6,148,162,0.10); color: #0E7490 !important; border: 1px solid rgba(6,148,162,0.35);}
    @keyframes pulse{ 0%{opacity:1;} 50%{opacity:0.35;} 100%{opacity:1;} }

    /* ---------- Verdict banners ---------- */
    .verdict-bad{
        background: rgba(220,38,38,0.08); border: 1px solid rgba(220,38,38,0.35);
        border-left: 3px solid #DC2626; border-radius: 10px; padding: 14px 18px;
        font-weight: 700; color: #DC2626 !important; font-size: 1.05rem; margin-bottom: 14px;
    }
    .verdict-warn{
        background: rgba(217,119,6,0.08); border: 1px solid rgba(217,119,6,0.35);
        border-left: 3px solid #D97706; border-radius: 10px; padding: 14px 18px;
        font-weight: 700; color: #92400E !important; font-size: 1.05rem; margin-bottom: 14px;
    }
    .verdict-ok{
        background: rgba(5,150,105,0.08); border: 1px solid rgba(5,150,105,0.35);
        border-left: 3px solid #059669; border-radius: 10px; padding: 14px 18px;
        font-weight: 700; color: #065F46 !important; font-size: 1.05rem; margin-bottom: 14px;
    }

    /* ---------- Info box ---------- */
    .info-box {
        background: rgba(6,148,162,0.06); border: 1px solid rgba(6,148,162,0.25);
        border-left: 3px solid var(--cyan); border-radius: 10px; padding: 14px 18px;
        font-size: .88rem; color: var(--text-mid) !important; margin: 8px 0; line-height: 1.7;
    }

    /* ---------- Buttons ---------- */
    .stButton>button, .stFormSubmitButton>button{
        background: linear-gradient(135deg, #0694A2 0%, #047481 100%);
        color:#FFFFFF !important; font-weight:700; border:none; border-radius:8px;
        letter-spacing:0.02em; transition: filter .15s ease;
    }
    .stButton>button:hover, .stFormSubmitButton>button:hover{ filter: brightness(1.1); }

    /* ---------- Tabs ---------- */
    div[data-baseweb="tab-list"]{ gap: 4px; background: #F0F4FB; border-radius: 8px; padding: 4px; }
    button[data-baseweb="tab"]{ font-family:'JetBrains Mono', monospace; font-size:0.82rem; color: var(--text-mid) !important; }
    button[data-baseweb="tab"][aria-selected="true"]{ background: #FFFFFF; color: var(--cyan) !important; border-radius: 6px; }

    /* ---------- Dataframe ---------- */
    .stDataFrame { border: 1px solid var(--border); border-radius: 8px; }

    /* ---------- Metrics ---------- */
    [data-testid="stMetricValue"] { color: var(--text-hi) !important; }
    [data-testid="stMetricLabel"] { color: var(--text-dim) !important; }

    /* ---------- Scrollbar ---------- */
    ::-webkit-scrollbar{ width:8px; height:8px; }
    ::-webkit-scrollbar-track{ background: var(--bg-0); }
    ::-webkit-scrollbar-thumb{ background: #CBD5E0; border-radius: 6px; }
    ::-webkit-scrollbar-thumb:hover{ background: #A0AEC0; }

    /* ---------- Dark mode override (must stay LAST to win the cascade) ---------- */
    @media (prefers-color-scheme: dark) {
        :root{
            --bg-0:#0D1117; --bg-1:#161B22; --panel:#161B22; --panel-alt:#1C2128;
            --border:#30363D; --border-soft:#21262D;
            --text-hi:#E6EDF3; --text-mid:#B0B8C4; --text-dim:#7D8590;
            --cyan:#39C5CF; --amber:#D29922; --danger:#F85149; --success:#3FB950;
        }
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        .main {
            background-color: var(--bg-0) !important;
            background-image:
                radial-gradient(1100px 550px at 100% -8%, rgba(57,197,207,0.08), transparent 60%),
                radial-gradient(900px 500px at -5% 8%, rgba(210,153,34,0.05), transparent 55%),
                linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%) !important;
        }
        .kpi-card, section[data-testid="stSidebar"], div[data-baseweb="tab-list"],
        button[data-baseweb="tab"][aria-selected="true"] {
            background: var(--panel) !important;
        }
        .stDataFrame { border-color: var(--border) !important; }
    }
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

def kpi_card(col, label, value, sub=""):
    """Renders a small KPI card inside the given column."""
    with col:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
            <div class="kpi-value" style="font-size:1.3rem;">{value}</div>
            <div class="kpi-sub">{sub}</div></div>""", unsafe_allow_html=True)


def section(title: str):
    """Renders a section title with the amber accent bar."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def dark_fig(fig, height=350):
    """Applies consistent dark theme styling to a Plotly figure."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#EAF0F7",
    )
    return fig


def info_box(html_content: str):
    """Renders an info box with consistent styling."""
    st.markdown(f"""<div style="background: rgba(47,217,203,0.08);
        border: 1px solid rgba(47,217,203,0.3); border-left: 3px solid #2FD9CB;
        border-radius: 10px; padding: 14px 18px; font-size: 0.9rem;
        color: #A9B6C9; line-height: 1.7;">{html_content}</div>""",
        unsafe_allow_html=True)

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
        "🏠 Landing",
        "📊 Overview",
        "🔍 Dataset Explorer",
        "🎯 Model Performance",
        "📈 Model Comparison",
        "🧠 Explainability (SHAP)",
        "🌊 Noise Robustness",
        "⚡ Live Prediction",
        "💰 ROI Calculator",
        "📡 Live Monitoring",
        "📜 Prediction History",
        "🖼️ Output Gallery",
        "ℹ️ About the Project",
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
# PAGE: LANDING
# =================================================================
if page == "🏠 Landing":
    # ── Hero Banner ──────────────────────────────────────────────
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #FFFFFF 0%, #EBF8FF 50%, #F0FFF4 100%);
        border: 1px solid #BEE3F8; border-radius: 16px;
        padding: 52px 48px; margin-bottom: 32px; text-align: center;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06);
    ">
        <div style="font-size:3.5rem; margin-bottom:12px;">⚙️</div>
        <div style="font-size:2.4rem; font-weight:800; color:#1A202C; line-height:1.2; margin-bottom:16px;">
            Contextual Predictive Maintenance
        </div>
        <div style="font-size:1.15rem; color:#4A5568; max-width:620px; margin:0 auto 24px auto; line-height:1.7;">
            An AI-powered IoT system that predicts machine failures before they happen —
            fusing internal sensor data with real-world environmental context.
        </div>
        <div style="display:flex; gap:10px; justify-content:center; flex-wrap:wrap; margin-bottom:28px;">
            <span style="background:#EBF8FF;color:#2B6CB0;border:1px solid #BEE3F8;
                padding:5px 14px;border-radius:999px;font-size:.8rem;font-weight:600;">
                LightGBM + SMOTE
            </span>
            <span style="background:#F0FFF4;color:#276749;border:1px solid #9AE6B4;
                padding:5px 14px;border-radius:999px;font-size:.8rem;font-weight:600;">
                Macro F1 = 0.8501 ✅
            </span>
            <span style="background:#FFF5F5;color:#9B2C2C;border:1px solid #FEB2B2;
                padding:5px 14px;border-radius:999px;font-size:.8rem;font-weight:600;">
                IoT Edge AI
            </span>
            <span style="background:#FAF5FF;color:#553C9A;border:1px solid #D6BCFA;
                padding:5px 14px;border-radius:999px;font-size:.8rem;font-weight:600;">
                SHAP Explainability
            </span>
            <span style="background:#FFFAF0;color:#7B341E;border:1px solid #FBD38D;
                padding:5px 14px;border-radius:999px;font-size:.8rem;font-weight:600;">
                Real-time API
            </span>
        </div>
        <div style="font-size:.9rem; color:#718096;">
            Infotact Solutions & Co. · Bengaluru · Internship 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick Stats Strip ────────────────────────────────────────
    st.markdown('<div class="section-title">📊 At a Glance</div>', unsafe_allow_html=True)
    s1, s2, s3, s4, s5 = st.columns(5)
    for col, icon, label, value in [
        (s1, "🎯", "Macro F1",    "0.8501"),
        (s2, "📐", "Precision",   "0.8233"),
        (s3, "🔁", "Recall",      "0.8825"),
        (s4, "⚖️", "Imbalance",   "28.5:1"),
        (s5, "📦", "Dataset",     "10,000 rows"),
    ]:
        with col:
            st.markdown(
                f'<div class="kpi-card" style="text-align:center;">'
                f'<div style="font-size:1.6rem;margin-bottom:6px;">{icon}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value" style="font-size:1.3rem;">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Problem vs Solution ──────────────────────────────────────
    st.markdown('<div class="section-title">🔍 Problem & Solution</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div style="background:#FFF5F5; border:1px solid #FEB2B2; border-left:4px solid #FC8181;
            border-radius:10px; padding:20px 22px;">
            <div style="font-size:1.1rem; font-weight:700; color:#9B2C2C; margin-bottom:10px;">
                ❌ The Problem
            </div>
            <ul style="color:#742A2A; font-size:.9rem; line-height:2; margin:0; padding-left:18px;">
                <li>Machines fail unexpectedly — costly downtime</li>
                <li>Existing ML systems ignore external context</li>
                <li>Rare failure events = highly imbalanced data</li>
                <li>Black-box models — engineers can't trust them</li>
                <li>No real-time prediction capability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div style="background:#F0FFF4; border:1px solid #9AE6B4; border-left:4px solid #48BB78;
            border-radius:10px; padding:20px 22px;">
            <div style="font-size:1.1rem; font-weight:700; color:#276749; margin-bottom:10px;">
                ✅ Our Solution
            </div>
            <ul style="color:#22543D; font-size:.9rem; line-height:2; margin:0; padding-left:18px;">
                <li>Contextual data fusion — sensors + environment</li>
                <li>SMOTE handles class imbalance correctly</li>
                <li>LightGBM captures complex non-linear patterns</li>
                <li>SHAP explains every prediction to engineers</li>
                <li>FastAPI + Streamlit = real-time live dashboard</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── How it works ─────────────────────────────────────────────
    st.markdown('<div class="section-title">⚙️ How It Works</div>', unsafe_allow_html=True)
    steps = [
        ("1", "📥", "IoT Sensors",        "Air temp, process temp, rotational speed, torque, tool wear"),
        ("2", "🌍", "External Context",   "Ambient temperature, factory load, humidity"),
        ("3", "⚖️", "SMOTE Balancing",    "Applied inside CV folds only — no data leakage"),
        ("4", "🤖", "LightGBM Model",     "Gradient boosting — Macro F1 = 0.8501 ✅"),
        ("5", "🔍", "SHAP Explanation",   "Tells engineers WHY a failure is predicted"),
        ("6", "📡", "Live API",           "FastAPI /predict — responds in < 100ms"),
        ("7", "📊", "Dashboard",          "Streamlit — 9 pages, real-time monitoring"),
    ]
    for step in steps:
        num, icon, title, desc = step
        st.markdown(
            f'<div style="display:flex; gap:14px; align-items:flex-start; '
            f'padding:12px 16px; background:#FFFFFF; border:1px solid #E2E8F0; '
            f'border-radius:10px; margin:6px 0; box-shadow:0 1px 4px rgba(0,0,0,0.04);">'
            f'<div style="min-width:32px; height:32px; background:#EBF8FF; border:1px solid #BEE3F8; '
            f'border-radius:50%; display:flex; align-items:center; justify-content:center; '
            f'font-weight:700; color:#2B6CB0; font-size:.85rem;">{num}</div>'
            f'<div style="font-size:1.3rem;">{icon}</div>'
            f'<div>'
            f'<div style="font-weight:600; color:#1A202C; font-size:.95rem;">{title}</div>'
            f'<div style="color:#718096; font-size:.82rem; margin-top:2px;">{desc}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Navigate CTA ─────────────────────────────────────────────
    st.markdown("""
    <div style="background:#EBF8FF; border:1px solid #BEE3F8; border-radius:12px;
        padding:20px 24px; text-align:center;">
        <div style="font-size:1rem; font-weight:600; color:#2C5282; margin-bottom:6px;">
            👈 Use the sidebar to explore the dashboard
        </div>
        <div style="font-size:.85rem; color:#4A5568;">
            Overview · Dataset Explorer · Model Performance · SHAP · Noise Robustness ·
            Live Prediction · Live Monitoring · Prediction History · Output Gallery · About
        </div>
    </div>
    """, unsafe_allow_html=True)

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
# PAGE: MODEL COMPARISON
# =================================================================
elif page == "📈 Model Comparison":
    console_header("📈", "Model Comparison", eyebrow="RESEARCH",
                    subtitle="Does contextual data fusion actually help? Comparing across models and feature sets")

    with st.spinner("Running fresh ablation study (Random Forest, ~10-20s)..."):
        from src.model_comparison_data import get_model_comparison_data
        comparison_df = get_model_comparison_data()

    st.markdown('<div class="section-title">Macro F1 Across Model / Feature-Set Variants</div>', unsafe_allow_html=True)

    comparison_df["label"] = comparison_df["model"] + " — " + comparison_df["feature_set"]

    fig = px.bar(
        comparison_df, x="label", y="macro_f1",
        color="model",
        text=comparison_df["macro_f1"].apply(lambda v: f"{v:.4f}"),
        labels={"label": "Model / Feature Set", "macro_f1": "Macro F1 Score"},
    )
    fig.add_hline(y=0.85, line_dash="dash", line_color="#FF5C6C", annotation_text="Target F1 = 0.85")
    fig.update_traces(textposition="outside")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", height=450,
        xaxis_tickangle=-15, showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Raw Comparison Table</div>', unsafe_allow_html=True)
    st.dataframe(
        comparison_df[["model", "feature_set", "macro_f1", "precision", "recall"]],
        use_container_width=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Feature Importance — Why One Model Wins</div>', unsafe_allow_html=True)

    with st.spinner("Computing feature importances..."):
        from src.model_comparison_data import get_feature_importance_comparison
        importance_dict = get_feature_importance_comparison()

    imp_tabs = st.tabs(list(importance_dict.keys()))
    for tab, (label, importances) in zip(imp_tabs, importance_dict.items()):
        with tab:
            top_n = importances.head(10)
            fig_imp = px.bar(
                top_n[::-1], orientation="h",
                labels={"value": "Feature Importance", "index": "Feature"},
            )
            fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=380, showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    info_box(
        "<b>Honest finding:</b> External context features (ambient temperature, factory load, "
        "humidity) are simulated/random in this dataset, so they slightly <i>hurt</i> a basic "
        "Random Forest's Macro F1. However, the production <b>LightGBM + SMOTE</b> pipeline "
        "— which handles class imbalance properly — achieves strong performance "
        "(<b>Macro F1 = 0.8501</b>) using the same full feature set, comfortably exceeding "
        "the target of 0.85. This shows the value of proper class-imbalance handling combined "
        "with a stronger model, rather than context features alone."
    )

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
# PAGE: ROI CALCULATOR
# =================================================================
elif page == "💰 ROI Calculator":
    console_header("💰", "ROI Calculator", eyebrow="BUSINESS IMPACT",
                    subtitle="Estimate the rupee value of catching a failure before it happens")

    from src.roi_calculator import estimate_savings

    with st.form("roi_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            downtime_cost = st.number_input(
                "Downtime cost (Rs. per hour)", min_value=0.0, value=5000.0, step=500.0
            )
        with c2:
            hours_avoided = st.number_input(
                "Estimated hours of downtime avoided", min_value=0.0, value=8.0, step=1.0
            )
        with c3:
            machine_type_roi = st.selectbox("Machine Type", sorted(raw_df["Type"].unique().tolist()), key="roi_type")

        st.markdown("**Sensor Reading (for live failure probability)**")
        r1, r2, r3 = st.columns(3)
        with r1:
            roi_air_temp = st.slider("Air temperature [K]", 295.0, 305.0, 300.0, key="roi_air")
            roi_proc_temp = st.slider("Process temp [K]", 305.0, 315.0, 310.0, key="roi_proc")
        with r2:
            roi_rot_speed = st.slider("Rotational speed [rpm]", 1000.0, 2500.0, 1500.0, key="roi_rot")
            roi_torque = st.slider("Torque [Nm]", 3.0, 80.0, 40.0, key="roi_torq")
        with r3:
            roi_tool_wear = st.slider("Tool wear [min]", 0.0, 250.0, 100.0, key="roi_wear")

        roi_submit = st.form_submit_button("Calculate Estimated Savings", use_container_width=True)

    if roi_submit:
        type_enc_roi = int(type_encoder.transform([machine_type_roi])[0])
        row = {
            "Air temperature [K]": roi_air_temp, "Process temperature [K]": roi_proc_temp,
            "Rotational speed [rpm]": roi_rot_speed, "Torque [Nm]": roi_torque,
            "Tool wear [min]": roi_tool_wear, "Type_enc": type_enc_roi,
            "ambient_temp_C": 28.0, "factory_load_pct": 75.0, "humidity_pct": 60.0,
        }
        x_roi = pd.DataFrame([row])
        x_roi.columns = [clean_col(c) for c in x_roi.columns]
        x_roi = x_roi[X.columns]

        proba = float(pipeline.predict_proba(x_roi)[0, 1])

        result = estimate_savings(
            downtime_cost_per_hour=downtime_cost,
            hours_downtime_avoided=hours_avoided,
            failure_probability=proba
        )

        c1, c2, c3 = st.columns(3)
        kpi_card(c1, "Failure Probability", f"{proba*100:.1f}%", "from live model")
        kpi_card(c2, "Downtime Avoided", f"{hours_avoided:.0f} hrs", f"@ Rs. {downtime_cost:.0f}/hr")
        kpi_card(c3, "Estimated Savings", f"Rs. {result['estimated_savings']:,.2f}", "expected value")

        st.markdown("<br>", unsafe_allow_html=True)
        info_box(
            f"<b>How this is calculated:</b><br>"
            f"Estimated Savings = Downtime Cost/Hour (Rs. {downtime_cost:,.0f}) × "
            f"Hours Avoided ({hours_avoided:.0f}) × Failure Probability ({proba*100:.1f}%) "
            f"= <b>Rs. {result['estimated_savings']:,.2f}</b>"
        )
        st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Cumulative Savings (Live Demo)</div>', unsafe_allow_html=True)

    log_path = "logs/predictions_log.csv"
    if os.path.exists(log_path):
        try:
            log_df = pd.read_csv(log_path)
            if "probability" in log_df.columns and len(log_df) > 0:
                # Use the same downtime cost / hours-avoided assumptions as the form above
                per_prediction_savings = log_df["probability"] * downtime_cost * hours_avoided
                cumulative_savings = per_prediction_savings.cumsum()

                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    y=cumulative_savings,
                    mode="lines+markers",
                    line=dict(color="#059669", width=2),
                    marker=dict(size=5, color="#059669"),
                    fill="tozeroy",
                    fillcolor="rgba(5,150,105,0.10)",
                    name="Cumulative Savings"
                ))
                fig_cum.update_layout(
                    height=350,
                    xaxis_title="Prediction #",
                    yaxis_title="Cumulative Estimated Savings (Rs.)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#1A202C",
                )
                st.plotly_chart(fig_cum, use_container_width=True)

                total_savings = cumulative_savings.iloc[-1]
                st.markdown(
                    f'<div class="kpi-card" style="border-left-color:#059669;">'
                    f'<div class="kpi-label">Total Estimated Savings So Far</div>'
                    f'<div class="kpi-value">Rs. {total_savings:,.2f}</div>'
                    f'<div class="kpi-sub">across {len(log_df)} logged predictions</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("No predictions logged yet — send some predictions via Live Monitoring to see this chart grow.")
        except Exception as e:
            st.warning(f"Could not load prediction log: {e}")
    else:
        st.info(
            "No prediction log found yet. Start the API and send predictions "
            "(via Live Monitoring or the simulator) to populate this chart."
        )
# =================================================================
# PAGE: LIVE MONITORING
# =================================================================
elif page == "📡 Live Monitoring":
    console_header("📡", "Live Monitoring", eyebrow="REAL-TIME",
                   subtitle="Live sensor stream → api.py → dashboard · Start API first, then simulate_stream.py")

    API_URL = "http://127.0.0.1:8000"

    # ── API Status + Controls ────────────────────────────────────
    col_status, col_refresh, col_threshold = st.columns([2, 1, 1])
    with col_status:
        try:
            import requests as req
            health = req.get(f"{API_URL}/health", timeout=2)
            if health.status_code == 200:
                hdata = health.json()
                st.markdown(
                    f'<span class="badge badge-live">'
                    f'<span class="badge-dot"></span>'
                    f'API ONLINE &nbsp;·&nbsp; Model: {"✅ Loaded" if hdata.get("model_loaded") else "⚠️ Not loaded"}'
                    f'</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="badge badge-demo"><span class="badge-dot"></span>API ERROR</span>',
                    unsafe_allow_html=True,
                )
        except Exception:
            st.markdown(
                '<span class="badge badge-demo">'
                '<span class="badge-dot"></span>'
                'API OFFLINE — run: python -m uvicorn api:app --reload'
                '</span>',
                unsafe_allow_html=True,
            )
    with col_refresh:
        auto_refresh = st.toggle("🔄 Auto Refresh", value=False,
                                  help="Refreshes every 3 seconds")
    with col_threshold:
        live_threshold = st.slider("Threshold", 0.1, 0.9, 0.5, 0.05,
                                    help="Decision threshold for failure prediction",
                                    label_visibility="visible")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick stats bar ──────────────────────────────────────────
    qs1, qs2, qs3, qs4 = st.columns(4)
    for col, label, value, sub in [
        (qs1, "API Endpoint",  "localhost:8000",    "/predict + /health"),
        (qs2, "Model",         "LightGBM + SMOTE",  "Macro F1 = 0.8501 ✅"),
        (qs3, "Features",      "9",                  "5 internal + 4 context"),
        (qs4, "Threshold",     f"{live_threshold}",  "Adjustable above"),
    ]:
        with col:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value" style="font-size:1.3rem;">{value}</div>'
                f'<div class="kpi-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk level guide ─────────────────────────────────────────
    st.markdown('<div class="section-title">🚦 Risk Level Guide</div>', unsafe_allow_html=True)
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown(
            '<div class="kpi-card" style="border-left-color:#3ED598;text-align:center;">'
            '<div style="font-size:2rem;margin-bottom:6px;">✅</div>'
            '<div class="kpi-label">HEALTHY</div>'
            '<div style="font-size:.85rem;color:#3ED598;margin-top:4px;">Probability &lt; 30%</div>'
            '<div style="font-size:.75rem;color:#697788;margin-top:4px;">No action needed</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with rc2:
        st.markdown(
            '<div class="kpi-card" style="border-left-color:#FFB020;text-align:center;">'
            '<div style="font-size:2rem;margin-bottom:6px;">⚠️</div>'
            '<div class="kpi-label">ELEVATED RISK</div>'
            '<div style="font-size:.85rem;color:#FFB020;margin-top:4px;">Probability 30–50%</div>'
            '<div style="font-size:.75rem;color:#697788;margin-top:4px;">Monitor closely</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with rc3:
        st.markdown(
            '<div class="kpi-card" style="border-left-color:#FF5C6C;text-align:center;">'
            '<div style="font-size:2rem;margin-bottom:6px;">🚨</div>'
            '<div class="kpi-label">CRITICAL</div>'
            '<div style="font-size:.85rem;color:#FF5C6C;margin-top:4px;">Probability &gt; 50%</div>'
            '<div style="font-size:.75rem;color:#697788;margin-top:4px;">Schedule maintenance now</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Manual prediction form ───────────────────────────────────
    st.markdown('<div class="section-title">📥 Send Single Sensor Reading to API</div>', unsafe_allow_html=True)
    with st.form("live_predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**⚙️ Internal Sensors**")
            lm_type      = st.selectbox("Machine Type", ["L", "M", "H"], key="lm_type")
            lm_air_temp  = st.slider("Air temperature [K]",  295.0, 305.0, 300.0, 0.1, key="lm_air")
            lm_proc_temp = st.slider("Process temp [K]",     305.0, 315.0, 310.0, 0.1, key="lm_proc")
        with c2:
            st.markdown("**⚙️ More Sensors**")
            lm_rot_speed = st.slider("Rotational speed [rpm]", 1000.0, 2500.0, 1500.0, 10.0, key="lm_rot")
            lm_torque    = st.slider("Torque [Nm]",              3.0,   80.0,   40.0,  0.5,  key="lm_torq")
            lm_tool_wear = st.slider("Tool wear [min]",           0.0,  250.0,  100.0, 1.0,  key="lm_wear")
        with c3:
            st.markdown("**🌍 External Context**")
            lm_ambient  = st.slider("Ambient temp [°C]", 10.0, 45.0,  28.0, 0.5, key="lm_amb")
            lm_load     = st.slider("Factory load [%]",  50.0, 100.0, 75.0, 0.5, key="lm_load")
            lm_humidity = st.slider("Humidity [%]",      20.0, 90.0,  60.0, 0.5, key="lm_hum")

        send_btn = st.form_submit_button("📡  Send to API & Predict", use_container_width=True)

    if send_btn:
        from sklearn.preprocessing import LabelEncoder as _LE
        _le = _LE(); _le.fit(["H", "L", "M"])
        type_enc_live = int(_le.transform([lm_type])[0])

        payload = {
            "Air_temperature_K":     lm_air_temp,
            "Process_temperature_K": lm_proc_temp,
            "Rotational_speed_rpm":  lm_rot_speed,
            "Torque_Nm":             lm_torque,
            "Tool_wear_min":         lm_tool_wear,
            "Type_enc":              type_enc_live,
            "ambient_temp_C":        lm_ambient,
            "factory_load_pct":      lm_load,
            "humidity_pct":          lm_humidity,
        }

        try:
            import requests as req
            response = req.post(f"{API_URL}/predict", json=payload, timeout=5)

            if response.status_code == 200:
                result = response.json()
                prob   = result.get("probability", 0)
                pred   = 1 if prob >= live_threshold else 0

                # Verdict banner
                if prob >= 0.5:
                    v_cls, v_txt = "verdict-bad",  f"🚨  FAILURE PREDICTED  ({prob*100:.1f}%)"
                elif prob >= 0.3:
                    v_cls, v_txt = "verdict-warn", f"⚠️  ELEVATED RISK  ({prob*100:.1f}%)"
                else:
                    v_cls, v_txt = "verdict-ok",   f"✅  HEALTHY — No action needed  ({prob*100:.1f}%)"

                st.markdown(f'<div class="{v_cls}">{v_txt}</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                col_g, col_d = st.columns([1, 1.5])

                with col_g:
                    # Gauge with smooth color transition
                    if prob >= 0.5:
                        gauge_color = "#FF5C6C"
                    elif prob >= 0.3:
                        gauge_color = "#FFB020"
                    else:
                        gauge_color = "#3ED598"

                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prob * 100,
                        delta={"reference": 50, "valueformat": ".1f",
                               "increasing": {"color": "#FF5C6C"},
                               "decreasing": {"color": "#3ED598"}},
                        number={"suffix": "%", "font": {"size": 38, "color": gauge_color}},
                        title={"text": "Failure Probability",
                               "font": {"color": "#EAF0F7", "size": 14}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#697788",
                                     "tickfont": {"color": "#697788"}},
                            "bar":  {"color": gauge_color, "thickness": 0.28},
                            "bgcolor": "#0D1219",
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0,  30], "color": "rgba(62,213,152,0.10)"},
                                {"range": [30, 50], "color": "rgba(255,176,32,0.10)"},
                                {"range": [50,100], "color": "rgba(255,92,108,0.10)"},
                            ],
                            "threshold": {
                                "line":  {"color": "#FF5C6C", "width": 2},
                                "thickness": 0.75,
                                "value": live_threshold * 100,
                            },
                        },
                    ))
                    fig_g.update_layout(
                        height=310,
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#EAF0F7",
                        margin=dict(l=20, r=20, t=30, b=10),
                    )
                    st.plotly_chart(fig_g, use_container_width=True)

                with col_d:
                    st.markdown("**📋 API Response**")
                    st.json(result)

                    # SHAP per-prediction
                    if shap is not None:
                        try:
                            lgbm_live = pipeline.named_steps.get("lgbm", pipeline)
                            exp_live  = shap.TreeExplainer(lgbm_live)
                            x_live    = pd.DataFrame([payload])
                            x_live.columns = X.columns
                            sv_live   = exp_live.shap_values(x_live)
                            sv_live   = sv_live[1] if isinstance(sv_live, list) else sv_live
                            contrib   = pd.Series(sv_live[0], index=X.columns).sort_values()
                            bar_clrs  = ["#FF5C6C" if v > 0 else "#3ED598" for v in contrib.values]

                            fig_shap = go.Figure(go.Bar(
                                x=contrib.values, y=contrib.index,
                                orientation="h", marker_color=bar_clrs,
                                text=[f"{v:+.3f}" for v in contrib.values],
                                textposition="outside",
                            ))
                            fig_shap.update_layout(
                                title="SHAP — Feature Contribution to This Prediction",
                                xaxis_title="SHAP value",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#EAF0F7",
                                height=320,
                                showlegend=False,
                                margin=dict(l=10, r=60, t=40, b=10),
                            )
                            st.plotly_chart(fig_shap, use_container_width=True)

                            st.markdown(
                                '<div style="font-size:.8rem;color:#697788;margin-top:4px;">'
                                '<span style="color:#FF5C6C;">■</span> Red = pushes toward failure &nbsp;|&nbsp;'
                                '<span style="color:#3ED598;">■</span> Green = pushes toward healthy'
                                '</div>',
                                unsafe_allow_html=True,
                            )
                        except Exception:
                            st.caption("SHAP unavailable for this input.")
                    else:
                        st.info("Install `shap` to see per-feature contributions.")

            else:
                st.error(f"API Error {response.status_code}: {response.text}")

        except Exception as exc:
            st.error(
                f"❌ Cannot connect to API at `{API_URL}`\n\n"
                f"Start it with: `python -m uvicorn api:app --reload`\n\n"
                f"Error: {exc}"
            )

    # ── Auto refresh ─────────────────────────────────────────────
    if auto_refresh:
        import time
        st.markdown(
            '<div style="font-size:.8rem;color:#697788;margin-top:8px;">'
            '🔄 Auto-refreshing every 3 seconds...</div>',
            unsafe_allow_html=True,
        )
        time.sleep(3)
        st.rerun()

    # ── How to use ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="background:rgba(47,217,203,0.06);border:1px solid rgba(47,217,203,0.22);'
        'border-left:3px solid #2FD9CB;border-radius:10px;padding:14px 18px;'
        'font-size:.88rem;color:#A9B6C9;line-height:1.8;">'
        '<b style="color:#EAF0F7;">How to use Live Monitoring:</b><br>'
        '1️⃣ Start API: <code>python -m uvicorn api:app --reload</code><br>'
        '2️⃣ Start simulator: <code>python simulate_stream.py</code><br>'
        '3️⃣ Use the form above to send manual predictions<br>'
        '4️⃣ Enable <b>Auto Refresh</b> toggle for continuous updates<br>'
        '5️⃣ Adjust <b>Threshold</b> slider to change sensitivity<br>'
        '6️⃣ Risk gauge turns 🔴 red when failure probability &gt; 50%'
        '</div>',
        unsafe_allow_html=True,
    )

# =================================================================
# PAGE: PREDICTION HISTORY
# =================================================================
elif page == "📜 Prediction History":
    console_header("📜", "Prediction History", eyebrow="LOGS",
                   subtitle="All predictions logged by the API — timestamp, sensor values, prediction, probability")

    log_paths = ["logs/predictions_log.csv", "predictions_log.csv"]
    log_df = None
    log_path_found = None

    for lp in log_paths:
        if os.path.exists(lp):
            try:
                log_df = pd.read_csv(lp)
                log_path_found = lp
                break
            except Exception:
                continue

    if log_df is None or log_df.empty:
        st.warning(
            "No prediction logs found yet. "
            "Start the API (`python -m uvicorn api:app --reload`) "
            "and send some predictions to generate logs."
        )
    else:
        # ── Summary KPIs ─────────────────────────────────────────
        total = len(log_df)
        if "prediction" in log_df.columns:
            failures = int(log_df["prediction"].sum())
            healthy  = total - failures
        else:
            failures = 0
            healthy  = total

        if "probability" in log_df.columns:
            avg_prob = log_df["probability"].mean()
            max_prob = log_df["probability"].max()
        else:
            avg_prob = 0.0
            max_prob = 0.0

        c1, c2, c3, c4 = st.columns(4)
        for col, label, value, sub in [
            (c1, "Total Predictions", f"{total:,}",       "All logged predictions"),
            (c2, "Failures Flagged",  f"{failures:,}",    f"{failures/total*100:.1f}% of total"),
            (c3, "Avg Probability",   f"{avg_prob:.4f}",  "Mean failure probability"),
            (c4, "Max Probability",   f"{max_prob:.4f}",  "Highest risk seen"),
        ]:
            with col:
                st.markdown(
                    f'<div class="kpi-card">'
                    f'<div class="kpi-label">{label}</div>'
                    f'<div class="kpi-value" style="font-size:1.3rem;">{value}</div>'
                    f'<div class="kpi-sub">{sub}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Probability over time chart ───────────────────────────
        if "probability" in log_df.columns:
            st.markdown('<div class="section-title">📈 Failure Probability Over Time</div>',
                        unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=log_df["probability"],
                mode="lines+markers",
                line=dict(color="#2FD9CB", width=1.5),
                marker=dict(
                    color=["#FF5C6C" if p >= 0.5 else "#FFB020" if p >= 0.3 else "#3ED598"
                           for p in log_df["probability"]],
                    size=6,
                ),
                name="Failure Probability",
            ))
            fig.add_hline(y=0.5, line_dash="dash", line_color="#FF5C6C",
                          annotation_text="Critical (0.50)")
            fig.add_hline(y=0.3, line_dash="dash", line_color="#FFB020",
                          annotation_text="Warning (0.30)")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#EAF0F7",
                height=380,
                xaxis_title="Prediction #",
                yaxis_title="Failure Probability",
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Prediction distribution ───────────────────────────────
        if "prediction" in log_df.columns:
            st.markdown('<div class="section-title">📊 Prediction Distribution</div>',
                        unsafe_allow_html=True)
            col_a, col_b = st.columns(2)

            with col_a:
                counts = log_df["prediction"].value_counts().rename({0: "Healthy", 1: "Failure"})
                fig2 = px.pie(
                    values=counts.values, names=counts.index, hole=0.55,
                    color=counts.index,
                    color_discrete_map={"Healthy": "#3ED598", "Failure": "#FF5C6C"},
                )
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300)
                st.plotly_chart(fig2, use_container_width=True)

            with col_b:
                if "probability" in log_df.columns:
                    fig3 = px.histogram(
                        log_df, x="probability", nbins=30,
                        color_discrete_sequence=["#2FD9CB"],
                        labels={"probability": "Failure Probability"},
                    )
                    fig3.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#EAF0F7",
                        height=300,
                    )
                    st.plotly_chart(fig3, use_container_width=True)

        # ── Raw log table ─────────────────────────────────────────
        st.markdown('<div class="section-title">📋 Raw Prediction Log</div>',
                    unsafe_allow_html=True)
        st.dataframe(log_df.tail(100), use_container_width=True, height=400)

        st.download_button(
            "⬇️ Download Full Log CSV",
            log_df.to_csv(index=False).encode(),
            "predictions_log.csv",
            mime="text/csv",
        )

        st.caption(f"Log source: `{log_path_found}` · {total:,} total predictions logged")

# =================================================================
# PAGE: OUTPUT GALLERY
# =================================================================
elif page == "🖼️ Output Gallery":
    console_header("🖼️", "Output Gallery", eyebrow="ARTEFACTS",
                   subtitle="All PNG outputs generated during the 4-week sprint")

    output_dirs = ["outputs", "notebooks/outputs", "../outputs"]
    png_files = []
    for d in output_dirs:
        png_files.extend(sorted(glob.glob(os.path.join(d, "*.png"))))

    if not png_files:
        st.warning(
            "No PNG files found in outputs/ folder. "
            "Run the Week 3-4 notebooks to generate SHAP and PR-curve plots first."
        )
    else:
        st.markdown(
            f'<span class="badge badge-live">{len(png_files)} images found</span>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        groups = {
            "🔍 SHAP Analysis":        [f for f in png_files if "shap" in f.lower()],
            "📈 Precision-Recall":      [f for f in png_files if "pr_" in f.lower() or "precision" in f.lower()],
            "🌊 Noise Robustness":      [f for f in png_files if "noise" in f.lower() or "robustness" in f.lower()],
            "🎯 Threshold & Confusion": [f for f in png_files if "threshold" in f.lower() or "confusion" in f.lower()],
            "📊 EDA & Features":        [f for f in png_files if not any(
                k in f.lower() for k in ["shap","pr_","noise","threshold","confusion","robustness","precision"])],
        }

        for group_name, files in groups.items():
            if not files:
                continue
            st.markdown(f'<div class="section-title">{group_name}</div>', unsafe_allow_html=True)
            cols = st.columns(min(3, len(files)))
            for i, fpath in enumerate(files):
                with cols[i % 3]:
                    st.image(fpath, caption=os.path.basename(fpath), use_container_width=True)

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
               f"Dashboard rendered {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
               f"<a href='https://github.com/tarunsaxena2/working' target='_blank' "
               f"style='color:#2FD9CB;'>GitHub ↗</a>",
               unsafe_allow_html=True)
