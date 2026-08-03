"""
report_generator.py — PDF Report Generator
Contextual Predictive Maintenance — IoT Edge AI

Packages model metrics and key charts into a single PDF report,
so judges have something to take away after the demo.

Usage:
    from src.report_generator import generate_report
    generate_report("output.pdf")
"""

import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


# Verified production model metrics (model_results.md)
MODEL_METRICS = {
    "Macro F1 Score": 0.8501,
    "Precision": 0.8233,
    "Recall": 0.8825,
}

MODEL_CONFIG = {
    "Model": "LightGBM + SMOTE",
    "n_estimators": 500,
    "learning_rate": 0.1,
    "num_leaves": 15,
    "scale_pos_weight": "Removed (SMOTE handles imbalance)",
}


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", fontSize=20, leading=24,
        textColor=colors.HexColor("#1A202C"), spaceAfter=6, fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", fontSize=11, leading=14,
        textColor=colors.HexColor("#718096"), spaceAfter=18
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading", fontSize=14, leading=18,
        textColor=colors.HexColor("#0694A2"), spaceBefore=16, spaceAfter=8,
        fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        name="BodyTextSmall", fontSize=9.5, leading=13,
        textColor=colors.HexColor("#4A5568")
    ))
    return styles


def _metrics_table():
    data = [["Metric", "Value"]] + [
        [k, f"{v:.4f}" if isinstance(v, float) else str(v)]
        for k, v in MODEL_METRICS.items()
    ]
    t = Table(data, colWidths=[220, 220])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0694A2")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4FB")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D9E6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def _config_table():
    data = [["Parameter", "Value"]] + [[k, str(v)] for k, v in MODEL_CONFIG.items()]
    t = Table(data, colWidths=[220, 220])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4A5568")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4FB")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D9E6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def generate_shap_chart_image(input_dict, output_image_path="shap_snapshot.png"):
    """
    Runs SHAP explanation for a single prediction and saves it as a PNG
    image, ready to be embedded into the PDF report.

    Parameters:
        input_dict (dict): the 9 sensor/context features
        output_image_path (str): where to save the PNG

    Returns:
        str: the output_image_path, or None if SHAP/model unavailable
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend, safe for scripts/servers
        import matplotlib.pyplot as plt
        from src.explain import explain_single

        result = explain_single(input_dict)
        feature_names = result['feature_names']
        shap_values = result['shap_values']

        sorted_pairs = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)
        names_sorted = [p[0] for p in sorted_pairs]
        values_sorted = [p[1] for p in sorted_pairs]
        colors_list = ['#DC2626' if v > 0 else '#059669' for v in values_sorted]

        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.barh(names_sorted, values_sorted, color=colors_list)
        ax.set_xlabel("SHAP Impact on Failure Probability")
        ax.set_title("Feature Contribution — This Prediction")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(output_image_path, dpi=150)
        plt.close(fig)

        return output_image_path
    except Exception as e:
        print(f"Could not generate SHAP chart image: {e}")
        return None


def generate_report(output_path="report.pdf", chart_image_paths=None, sample_prediction=None):
    """
    Generate a PDF report packaging model metrics, config, and optional charts.

    Parameters:
        output_path (str): where to save the PDF
        chart_image_paths (list[str], optional): list of PNG file paths to embed
            (e.g. SHAP chart, confusion matrix) — must already exist on disk
        sample_prediction (dict, optional): a single sensor reading + its
            prediction result, used to generate a live SHAP explanation
            snapshot embedded in the report. Expected keys:
            {'input': {...9 features...}, 'prediction': int, 'probability': float}

    Returns:
        str: the output_path, for convenience
    """
    styles = _build_styles()
    story = []

    story.append(Paragraph("Contextual Predictive Maintenance", styles["ReportTitle"]))
    story.append(Paragraph(
        f"IoT Edge AI — Model Performance Report · Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["ReportSubtitle"]
    ))

    story.append(Paragraph("Model Performance", styles["SectionHeading"]))
    story.append(_metrics_table())
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Target KPI: Macro F1 ≥ 0.85. Achieved via LightGBM + SMOTE, evaluated on a held-out test split.",
        styles["BodyTextSmall"]
    ))

    story.append(Paragraph("Model Configuration", styles["SectionHeading"]))
    story.append(_config_table())

    # ── Live prediction snapshot + SHAP explanation ──────────────
    if sample_prediction is not None:
        story.append(Paragraph("Sample Prediction — Explainability Snapshot", styles["SectionHeading"]))

        pred = sample_prediction.get("prediction")
        proba = sample_prediction.get("probability")
        verdict = "FAILURE PREDICTED" if pred == 1 else "HEALTHY"

        story.append(Paragraph(
            f"<b>Verdict:</b> {verdict} &nbsp;&nbsp; "
            f"<b>Failure Probability:</b> {proba*100:.2f}%",
            styles["BodyTextSmall"]
        ))
        story.append(Spacer(1, 8))

        shap_img_path = generate_shap_chart_image(sample_prediction.get("input", {}))
        if shap_img_path and os.path.exists(shap_img_path):
            story.append(Image(shap_img_path, width=440, height=260))
        else:
            story.append(Paragraph(
                "(SHAP explanation could not be generated for this sample.)",
                styles["BodyTextSmall"]
            ))

    if chart_image_paths:
        story.append(Paragraph("Supporting Charts", styles["SectionHeading"]))
        for img_path in chart_image_paths:
            if os.path.exists(img_path):
                try:
                    story.append(Spacer(1, 8))
                    story.append(Image(img_path, width=440, height=260))
                except Exception as e:
                    story.append(Paragraph(f"(Could not embed image: {img_path} — {e})", styles["BodyTextSmall"]))
            else:
                story.append(Paragraph(f"(Chart not found: {img_path})", styles["BodyTextSmall"]))

    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "Infotact Technical Internship Program 2026 · Contextual Predictive Maintenance (IoT Edge AI) · "
        "Tarun Saxena & Vaibhav Gautam",
        styles["BodyTextSmall"]
    ))

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=20 * mm, bottomMargin=20 * mm, leftMargin=20 * mm, rightMargin=20 * mm,
    )
    doc.build(story)
    return output_path


if __name__ == "__main__":
    sample = {
        "input": {
            'Air_temperature_K': 298.5, 'Process_temperature_K': 308.7,
            'Rotational_speed_rpm': 1500, 'Torque_Nm': 40.2,
            'Tool_wear_min': 10, 'Type_enc': 1,
            'ambient_temp_C': 28.0, 'factory_load_pct': 75.0, 'humidity_pct': 60.0
        },
        "prediction": 0,
        "probability": 0.0015,
    }
    path = generate_report("test_report.pdf", sample_prediction=sample)
    print(f"✅ Report generated: {path}")
