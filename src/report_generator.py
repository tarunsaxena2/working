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


def generate_report(output_path="report.pdf", chart_image_paths=None):
    """
    Generate a PDF report packaging model metrics, config, and optional charts.

    Parameters:
        output_path (str): where to save the PDF
        chart_image_paths (list[str], optional): list of PNG file paths to embed
            (e.g. SHAP chart, confusion matrix) — must already exist on disk

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
    path = generate_report("test_report.pdf")
    print(f"✅ Report generated: {path}")
