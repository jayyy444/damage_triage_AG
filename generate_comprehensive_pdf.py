import os
import sys
from fpdf import FPDF, XPos, YPos

class ProjectDocumentationPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)

    def header(self):
        self.set_fill_color(10, 14, 20)
        self.rect(0, 0, 210, 12, style="F")
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(0, 212, 170)
        self.set_xy(15, 3)
        self.cell(100, 6, "DAMAGETRIAGE: MULTI-MODAL EARTHQUAKE ASSESSMENT", align="L")
        self.set_text_color(160, 175, 195)
        self.set_xy(115, 3)
        self.cell(80, 6, "MAJOR PROJECT COMPREHENSIVE BRIEF", align="R")
        self.set_y(18)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(120, 135, 150)
        self.cell(90, 8, "DamageTriage | QuickQuakeBuildings Benchmark", align="L")
        self.cell(90, 8, f"Page {self.page_no()}", align="R")

    def chapter_title(self, num_str, title_str):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(255, 255, 255)
        self.set_fill_color(17, 24, 32)
        self.cell(180, 8, f"  {num_str}. {title_str.upper()}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(0, 212, 170)
        self.set_line_width(0.5)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(2.5)

    def section_heading(self, heading):
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(0, 120, 110)
        self.cell(180, 5.5, heading, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(0.5)

    def body_p(self, text):
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(30, 40, 55)
        self.multi_cell(180, 4.4, text)
        self.ln(1.5)

    def bullet(self, bold_prefix, text):
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(15, 23, 42)
        self.cell(4, 4.4, "-")
        w_prefix = self.get_string_width(bold_prefix) + 1
        self.cell(w_prefix, 4.4, bold_prefix)
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(40, 50, 65)
        rem_w = 180 - 4 - w_prefix
        if rem_w < 50:
            rem_w = 180
            self.ln(4.4)
        self.multi_cell(rem_w, 4.4, text)
        self.ln(1)

    def table_row(self, col1, col2, is_header=False):
        if is_header:
            self.set_fill_color(220, 235, 245)
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(15, 23, 42)
        else:
            self.set_fill_color(248, 250, 252)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(40, 50, 65)
        self.cell(58, 5.2, f" {col1}", border=1, fill=True)
        self.cell(122, 5.2, f" {col2}", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

def generate_pdf():
    pdf = ProjectDocumentationPDF()
    pdf.add_page()

    # Title Banner
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(10, 14, 20)
    pdf.cell(180, 7, "DamageTriage: Project Specification & Viva Guide", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("Helvetica", "I", 9.5)
    pdf.set_text_color(0, 150, 130)
    pdf.cell(180, 5, "Multi-Modal SAR + Optical Building Damage Assessment & Calibrated Operational Triage", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # 1. Dataset
    pdf.chapter_title("1", "Dataset Architecture & Ingestion Details")
    pdf.body_p("The system is benchmarked on the QuickQuakeBuildings (QQB) dataset, curated from the 6 February 2023 Turkiye-Syria Earthquake sequence (Magnitude 7.8 Mw) in the Islahiye and Hatay sectors.")
    
    pdf.table_row("Specification Metric", "Benchmark Value / Detail", is_header=True)
    pdf.table_row("Benchmark Dataset", "QuickQuakeBuildings (QQB) - 2023 Turkiye Earthquake")
    pdf.table_row("Total Labeled Buildings", "4,029 individual building footprints (OpenStreetMap polygons)")
    pdf.table_row("Ground Truth Distribution", "Binary: 3,860 Intact (95.8%) vs 169 Damaged / Collapsed (4.2%)")
    pdf.table_row("Class Imbalance Ratio", "23 : 1 (~96% Intact, ~4% Damaged) - Extreme Natural Imbalance")
    pdf.table_row("Geographical Folds", "5 Geographically Disjoint Folds (~806 buildings each)")
    pdf.table_row("Held-Out Test Fold", "fold-2.txt (807 buildings - strictly unseen during training)")
    pdf.table_row("Pre-Disaster Optical", "RGB Visible Satellite Imagery (0.5m GSD resolution)")
    pdf.table_row("Post-Disaster Optical", "RGB Visible Rubble/Collapse Imagery (0.5m GSD resolution)")
    pdf.table_row("Pre-Disaster SAR", "Sentinel-1 C-band Radar (10m GSD, VV/VH polarization channels)")
    pdf.table_row("Post-Disaster SAR", "Sentinel-1 C-band Radar Pass (Loss of wall double-bounce scattering)")
    pdf.ln(2)

    # 2. How project is made
    pdf.chapter_title("2", "End-to-End Methodology (How It Was Built)")
    pdf.bullet("Phase 1: Multi-Modal Ingestion: ", "PyTorch QQBDataset pipeline ingesting 4 paired HDF5 (.mat) arrays per building footprint. Resampled to 128x128 pixels, converted SAR to dB scale, and enforced disjoint fold separation to prevent spatial autocorrelation leakage.")
    pdf.bullet("Phase 2: Siamese Multi-Branch Backbone: ", "Constructed 4-branch ResNet-18 feature extractors producing 128-d latent embeddings. Built a Learnable Gated Attention Fusion network with a softmax scalar gating mechanism: e_fused = sum(g_i * e_i).")
    pdf.bullet("Phase 3: Imbalance & Uncertainty Calibration: ", "Combated 96:4 imbalance via Binary Focal Loss (alpha=0.25, gamma=2.0) and Weighted Random Sampling. Implemented post-hoc Temperature Scaling fitted via L-BFGS optimization on validation Negative Log-Likelihood.")
    pdf.bullet("Phase 4: Explainability & Deployment: ", "Extracted decoupled Grad-CAM activation maps by hooking Layer-4 across independent sensor branches. Deployed interactive Streamlit command center and automated PDF incident dispatch reporting.")
    pdf.ln(2)

    # 3. Technicals & Triage Logic
    pdf.chapter_title("3", "Core Technical Specifications & Triage Logic")
    pdf.body_p("Rather than naive 50% thresholding, DamageTriage establishes a 4-tier decision-theoretic operational triage queue from calibrated probabilities P(damaged):")

    pdf.table_row("Calibrated Probability", "Operational Triage Tier & Search-and-Rescue Protocol", is_header=True)
    pdf.table_row("P(damaged) < 0.20", "INTACT: Structurally sound; bypass heavy physical search.")
    pdf.table_row("0.20 <= P < 0.50", "PROBABLY INTACT: Low structural risk; secondary sweep.")
    pdf.table_row("0.50 <= P < 0.80", "PROBABLY DAMAGED: Moderate risk; priority drone recon.")
    pdf.table_row("P(damaged) >= 0.80", "DAMAGED: Critical structural collapse; immediate heavy rescue dispatch.")
    pdf.ln(2)

    pdf.section_heading("Empirical Modality Ablation (SAR + Optical vs Optical-Only):")
    pdf.bullet("False Alarm Reduction: ", "In held-out test evaluations (top 20 priority queue), Optical-only classification suffered a 25.0% False Positive Rate (5 intact buildings falsely prioritized due to shadows and tarpaulins).")
    pdf.bullet("Fusion Performance: ", "SAR + Optical Fusion achieved a 0.0% False Positive Rate (100% precision in top priority queue) because radar verified the physical disappearance of vertical dihedral double-bounce reflections.")
    pdf.ln(2)

    # 4. References & Papers
    pdf.chapter_title("4", "Key Scientific References & Research Papers")
    pdf.bullet("[1] Lin et al. (2017): ", "'Focal Loss for Dense Object Detection', IEEE ICCV. (Modulating factor suppressing easy negative gradients in 96:4 imbalanced data).")
    pdf.bullet("[2] Guo et al. (2017): ", "'On Calibration of Modern Neural Networks', ICML. (L-BFGS Temperature Scaling formulation to optimize Expected Calibration Error without altering ROC ranking).")
    pdf.bullet("[3] Selvaraju et al. (2017): ", "'Grad-CAM: Visual Explanations from Deep Networks', IEEE ICCV. (Gradient-weighted class activation mapping adapted for Layer-4 Siamese branch hooks).")
    pdf.bullet("[4] QuickQuakeBuildings Benchmark (2023): ", "Multi-modal satellite building damage dataset for the 2023 Kahramanmaras, Turkiye earthquake sequence.")
    pdf.bullet("[5] Sentinel-1 SAR User Handbook: ", "European Space Agency (ESA). Synthetic Aperture Radar C-band microwave backscatter properties and dihedral corner reflection principles.")
    pdf.ln(2)

    # 5. Viva Defense Q&A
    pdf.chapter_title("5", "Most Probable Viva Defense Questions & Answers")

    viva_qa = [
        (
            "Q1: Why is SAR (Radar) necessary if high-resolution Optical imagery is already available?",
            "Optical cameras are passive sensors that fail during clouds, dust plumes, smoke, and nighttime. Furthermore, optical images only capture 2D surface color - tarps and roof shadows cause false alarms. Sentinel-1 SAR uses active 5.4 GHz microwave radar that operates 24/7 and directly detects structural collapse through loss of vertical wall-ground double-bounce reflection. Our ablation proved SAR eliminated all 25% optical false alarms."
        ),
        (
            "Q2: The QQB dataset has binary labels (0 and 1). Did you invent 4 classes?",
            "No, ground truth remains strictly binary (0 intact, 1 damaged). In disaster rescue, resources are scarce; a naive 50% cutoff is impractical. We implemented a decision-theoretic 4-tier triage queue derived from calibrated posterior probabilities P(damaged). We validated this using Temperature Scaling and proved monotonicity: actual empirical collapse rate strictly increases across our 4 triage tiers."
        ),
        (
            "Q3: Since 96% of buildings are intact, how did you prevent the model from predicting all 0s?",
            "We solved this on two levels: (1) Data Level: WeightedRandomSampler ensures each mini-batch sees balanced samples via inverse class frequencies. (2) Loss Level: Binary Focal Loss (alpha=0.25, gamma=2.0) downweights easy intact negative gradients by up to 90%, forcing the network to learn subtle collapse features. We evaluated on PR-AUC and AUROC rather than raw accuracy."
        ),
        (
            "Q4: How does Grad-CAM work across multiple sensor branches?",
            "In single-backbone CNNs, Grad-CAM hooks the final layer. In our 4-branch architecture, each sensor has an independent Layer-4 conv block. We attach hooks to one branch at a time while backpropagating the fused damage logit gradient. This generates paired heatmaps showing radar rubble scattering alongside optical roof pancaking for the exact same building."
        ),
        (
            "Q5: Why did you use geographically disjoint folds instead of random train-test split?",
            "Satellite imagery suffers from spatial autocorrelation: adjacent buildings share identical soil, illumination, and urban layout. A random split causes spatial data leakage between train and test sets. We strictly evaluated on held-out fold-2 (807 buildings), ensuring true generalization to unseen geographical sectors."
        )
    ]

    for q, a in viva_qa:
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(0, 80, 110)
        pdf.multi_cell(180, 4.4, q)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(35, 45, 60)
        pdf.multi_cell(180, 4.0, a)
        pdf.ln(1.8)

    out_pdf = "outputs/reports/DamageTriage_Project_Documentation_and_Viva_Guide.pdf"
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    pdf.output(out_pdf)
    print("Successfully generated:", out_pdf)
    return out_pdf

if __name__ == "__main__":
    generate_pdf()
