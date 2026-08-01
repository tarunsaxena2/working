"""
alert_system.py — Alert System for High-Risk Predictions
Vaibhav Gautam — Dashboard & Integration
Week 4 Day 1 Task

Triggers visual banner + sound when failure probability exceeds threshold.
Used by the Live Monitoring tab in app.py.
"""

import streamlit as st


# =================================================================
# ALERT THRESHOLDS
# =================================================================
CRITICAL_THRESHOLD = 0.50   # Red alert — schedule maintenance
WARNING_THRESHOLD  = 0.30   # Yellow alert — monitor closely


# =================================================================
# SOUND ALERT (base64 encoded beep)
# =================================================================
ALERT_SOUND_HTML = """
<audio autoplay>
  <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAA
  EAAQAIWwAAIFsAAAEACABkYXRhSAYAAAAAAAAAAAAAAAAAAAAAAAAA" type="audio/wav">
</audio>
"""


def trigger_sound_alert():
    """Trigger a sound alert in the browser."""
    st.markdown(ALERT_SOUND_HTML, unsafe_allow_html=True)


def show_critical_alert(prob: float, feature_name: str = None):
    """
    Show a critical (red) alert banner for high failure probability.

    Parameters:
        prob (float): Failure probability (0-1)
        feature_name (str): Optional top contributing feature name
    """
    feature_note = f" — Top driver: {feature_name}" if feature_name else ""
    st.markdown(
        f'<div style="'
        f'background: rgba(255,92,108,0.15);'
        f'border: 2px solid #FF5C6C;'
        f'border-radius: 12px;'
        f'padding: 18px 22px;'
        f'margin: 10px 0;'
        f'animation: flashAlert 1s ease-in-out 3;">'
        f'<div style="font-size:1.4rem;font-weight:700;color:#FF5C6C;">'
        f'🚨 CRITICAL ALERT — MACHINE FAILURE IMMINENT'
        f'</div>'
        f'<div style="font-size:0.9rem;color:#FF9AA4;margin-top:6px;">'
        f'Failure probability: <b>{prob*100:.1f}%</b>{feature_note}'
        f'</div>'
        f'<div style="font-size:0.85rem;color:#697788;margin-top:4px;">'
        f'⚡ Action required: Schedule immediate maintenance inspection.'
        f'</div>'
        f'</div>'
        f'<style>'
        f'@keyframes flashAlert {{'
        f'0%{{opacity:1}} 50%{{opacity:0.4}} 100%{{opacity:1}}'
        f'}}'
        f'</style>',
        unsafe_allow_html=True,
    )
    trigger_sound_alert()


def show_warning_alert(prob: float):
    """
    Show a warning (yellow) alert banner for elevated failure probability.

    Parameters:
        prob (float): Failure probability (0-1)
    """
    st.markdown(
        f'<div style="'
        f'background: rgba(255,176,32,0.12);'
        f'border: 1.5px solid #FFB020;'
        f'border-radius: 12px;'
        f'padding: 16px 20px;'
        f'margin: 10px 0;">'
        f'<div style="font-size:1.15rem;font-weight:700;color:#FFB020;">'
        f'⚠️ WARNING — ELEVATED FAILURE RISK'
        f'</div>'
        f'<div style="font-size:0.85rem;color:#FFCB6B;margin-top:6px;">'
        f'Failure probability: <b>{prob*100:.1f}%</b> — Monitor this machine closely.'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def show_healthy_status(prob: float):
    """
    Show a healthy (green) status banner.

    Parameters:
        prob (float): Failure probability (0-1)