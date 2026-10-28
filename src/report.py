"""
PDF disaster response report — summary stats, review-queue count,
triage levels breakdown, top-priority building table, and a sample of paired
SAR+optical Grad-CAM heatmaps. Uses fpdf2 (pip install fpdf2), no other deps.
"""
import os
import json
import datetime
from fpdf import FPDF
from fpdf.enums import XPos, YPos

import config


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "DamageTriage - Disaster Response Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, "Event: 2023 Turkiye-Syria Earthquake - SAR + Optical", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def generate_report(predictions, review_queue, summary, out_path=None, heatmap_dir=None, top_n=15):
    out_path = out_path or os.path.join(
        config.OUTPUTS_DIR, "reports",
        f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pdf = Report()
    pdf.add_page()

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"Total buildings assessed: {summary.get('total_buildings', len(predictions))}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Severity Breakdown", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"  damaged        {summary.get('predicted_damaged', 0):>5}   ({summary.get('damaged_pct', 0)}%)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"  intact         {summary.get('predicted_intact', 0):>5}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # Triage levels section
    triage_counts = summary.get("triage_counts", {})
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Triage Levels", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(90, 6, "Triage Level", border=1)
    pdf.cell(45, 6, "Count", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    for lvl in config.TRIAGE_LEVELS:
        cnt = triage_counts.get(lvl, 0)
        pdf.cell(90, 6, lvl, border=1)
        pdf.cell(45, 6, str(cnt), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Needs Human Review (uncertain prediction)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    rev_cnt = summary.get("review_count", summary.get("flagged_for_review", len(review_queue)))
    tot_cnt = summary.get("total_buildings", len(predictions))
    pdf.multi_cell(
        0, 6,
        f"{rev_cnt} of {tot_cnt} predictions are categorized under intermediate review levels "
        f"({', '.join(config.REVIEW_LEVELS)}) and are flagged for manual review before dispatch. "
        f"Full list in review_queue.json."
    )
    pdf.ln(4)

    priority = sorted([p for p in predictions if p.get("predicted_label") == "damaged"],
                       key=lambda p: p.get("probability_damaged", 0.0), reverse=True)[:top_n]
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, f"Top {len(priority)} Priority Buildings (highest damage probability)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(70, 6, "Building ID (OSM)", border=1)
    pdf.cell(45, 6, "Triage Level", border=1)
    pdf.cell(35, 6, "P(damaged)", border=1)
    pdf.cell(40, 6, "Confidence", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 8)
    for p in priority:
        bid = p.get("building_id", "unknown")
        bid_short = bid if len(bid) <= 35 else bid[:32] + "..."
        t_level = p.get("triage_level", "N/A")
        prob_dam = p.get("probability_damaged", 0.0)
        conf = p.get("confidence", 0.0)
        pdf.cell(70, 6, bid_short, border=1)
        pdf.cell(45, 6, str(t_level), border=1)
        pdf.cell(35, 6, f"{prob_dam:.2f}", border=1)
        pdf.cell(40, 6, f"{conf:.2f}", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if heatmap_dir and os.path.isdir(heatmap_dir):
        imgs = sorted(os.listdir(heatmap_dir))[:8]
        if imgs:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, "Sample Damage Heatmaps (Grad-CAM, SAR + Optical)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 8)
            pdf.multi_cell(0, 5, "Red = region the model's decision relied on most, per modality.")
            x0, y0 = 10, pdf.get_y() + 4
            w = 55
            for i, img_name in enumerate(imgs):
                col, row = i % 3, i // 3
                pdf.image(os.path.join(heatmap_dir, img_name),
                          x=x0 + col * (w + 5), y=y0 + row * (w + 12), w=w)

    pdf.set_y(-30)
    pdf.set_font("Helvetica", "I", 7)
    pdf.multi_cell(
        0, 4,
        "Source imagery: Capella Space (SAR) and Maxar Open Data (optical), CC BY 4.0. "
        "Building footprints/labels: OpenStreetMap + HOT, ODbL. Dataset: QuickQuakeBuildings "
        "(Sun et al., 2024), CC BY-NC-SA 4.0 - academic/non-commercial use only."
    )

    pdf.output(out_path)
    print(f"Report saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    with open(os.path.join(config.OUTPUTS_DIR, "predictions.json")) as f:
        predictions = json.load(f)
    with open(os.path.join(config.OUTPUTS_DIR, "review_queue.json")) as f:
        review_queue = json.load(f)
    with open(os.path.join(config.OUTPUTS_DIR, "triage_summary.json")) as f:
        summary = json.load(f)
    generate_report(predictions, review_queue, summary,
                     heatmap_dir=os.path.join(config.OUTPUTS_DIR, "heatmaps"))
