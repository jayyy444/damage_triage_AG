import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set clean high-resolution figure with modern dark presentation aesthetic
fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
fig.patch.set_facecolor("#0a0e14")
ax.set_facecolor("#0a0e14")
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

# ── Color Palette ──
BG_BOX = "#111820"
BORDER_BOX = "#1e2a3a"
ACCENT_CYAN = "#00d4aa"
ACCENT_BLUE = "#3a86ff"
ACCENT_ORANGE = "#f5a623"
ACCENT_RED = "#ff4d3d"
ACCENT_GREEN = "#3dd68c"
TEXT_WHITE = "#e8edf4"
TEXT_MUTED = "#8b9bb4"

# ── Helper function to draw rounded boxes ──
def draw_card(x, y, w, h, title, subtitle="", bg=BG_BOX, border=BORDER_BOX, border_width=1.5, title_color=TEXT_WHITE):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.15",
        facecolor=bg, edgecolor=border, linewidth=border_width, zorder=2
    )
    ax.add_patch(box)
    if title:
        ty = y + h - 0.35 if subtitle else y + h / 2
        ax.text(x + w / 2, ty, title, color=title_color, fontsize=10, fontweight="bold",
                ha="center", va="center", zorder=3)
    if subtitle:
        ax.text(x + w / 2, y + h - 0.75, subtitle, color=TEXT_MUTED, fontsize=8,
                ha="center", va="center", zorder=3)

# ── Title Header ──
ax.text(8.0, 8.55, "DamageTriage: End-to-End System Architecture",
        color=TEXT_WHITE, fontsize=18, fontweight="bold", ha="center", va="center")
ax.text(8.0, 8.2, "Multi-Modal SAR + Optical Building Damage Assessment & Calibrated Operational Triage",
        color=ACCENT_CYAN, fontsize=10, ha="center", va="center")

# ── Phase Column Headers ──
phases = [
    (2.0, "PHASE 1: INPUT MODALITIES", ACCENT_CYAN),
    (6.0, "PHASE 2: DEEP FEATURE FUSION", ACCENT_BLUE),
    (10.2, "PHASE 3: CALIBRATION & TRIAGE", ACCENT_ORANGE),
    (14.0, "PHASE 4: FIELD DEPLOYMENT", ACCENT_GREEN),
]
for px, ptitle, pcol in phases:
    ax.text(px, 7.6, ptitle, color=pcol, fontsize=9.5, fontweight="bold", ha="center", va="center")
    ax.plot([px - 1.4, px + 1.4], [7.4, 7.4], color=pcol, linewidth=1.5, alpha=0.7)

# ── 1. INPUT MODALITIES (Left column) ──
inputs = [
    (6.0, "Pre-Disaster Optical", "RGB Visible Imagery (3ch)"),
    (4.6, "Post-Disaster Optical", "RGB Visible Rubble/Debris (3ch)"),
    (3.2, "Pre-Disaster SAR", "Sentinel-1 Radar (VV/VH 2ch)"),
    (1.8, "Post-Disaster SAR", "Sentinel-1 Radar Backscatter (2ch)"),
]

for y, name, desc in inputs:
    draw_card(0.6, y, 2.7, 1.1, name, desc, border=BORDER_BOX)

# ── 2. FEATURE EXTRACTION & FUSION (Column 2) ──
for y, _, _ in inputs:
    draw_card(3.9, y + 0.1, 2.0, 0.9, "ResNet-18", "Branch Embed (128-d)", border="#2a3b53")
    # Arrow from Input to ResNet
    ax.annotate("", xy=(3.9, y + 0.55), xytext=(3.3, y + 0.55),
                arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=1.5))

# Gated Attention Fusion Block
draw_card(6.5, 3.0, 2.4, 2.6, "Learnable Gated\nAttention Fusion",
          "Softmax Scalar Gate g_i\nFused e = sum(g_i * e_i)", bg="#0d1b2a", border=ACCENT_BLUE, border_width=2, title_color=ACCENT_CYAN)

# Connect ResNet branches to Fusion
for y, _, _ in inputs:
    ax.annotate("", xy=(6.5, 4.3), xytext=(5.9, y + 0.55),
                arrowprops=dict(arrowstyle="->", color=ACCENT_BLUE, lw=1.2, connectionstyle="arc3,rad=-0.05"))

# Training Loss Badge below Fusion
draw_card(6.5, 1.5, 2.4, 1.1, "Optimization & Loss", "Binary Focal Loss (γ=2.0)\n+ Weighted Sampler (96:4)",
          bg="#141a24", border="#334155")
ax.annotate("", xy=(7.7, 3.0), xytext=(7.7, 2.6),
            arrowprops=dict(arrowstyle="->", color=ACCENT_ORANGE, lw=1.5, linestyle="--"))

# ── 3. CALIBRATION & 4-TIER TRIAGE (Column 3) ──
# Arrow from Fusion to Temperature Scaling
ax.annotate("", xy=(9.3, 4.3), xytext=(8.9, 4.3),
            arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=2))

draw_card(9.3, 3.6, 2.0, 1.4, "Temperature\nScaling", "Logits / T via L-BFGS\nMinimizing NLL (ECE)",
          bg="#1a1c24", border=ACCENT_ORANGE, border_width=1.8, title_color=ACCENT_ORANGE)

# 4 Triage Tiers
tiers = [
    (6.0, "Damaged (>= 0.80)", "Critical Heavy Rescue Dispatch", ACCENT_RED),
    (4.6, "Probably Damaged (0.50-0.80)", "Priority Drone/UAV Recon", "#ff7a00"),
    (3.2, "Probably Intact (0.20-0.50)", "Secondary Ground Sweep", ACCENT_ORANGE),
    (1.8, "Intact (< 0.20)", "Clear / Bypass Physical Search", ACCENT_GREEN),
]

for ty, tname, tdesc, tcol in tiers:
    draw_card(11.7, ty, 2.4, 1.1, tname, tdesc, bg="#111820", border=tcol, border_width=1.5, title_color=tcol)
    ax.annotate("", xy=(11.7, ty + 0.55), xytext=(11.3, 4.3),
                arrowprops=dict(arrowstyle="->", color=ACCENT_ORANGE, lw=1.2, connectionstyle="arc3,rad=0.0"))

# ── 4. FIELD DEPLOYMENT & XAI (Column 4) ──
draw_card(14.5, 4.9, 1.3, 2.1, "Decoupled\nGrad-CAM", "Layer-4 Hooks\nRadar & Optical\nHeatmaps",
          bg="#0d2328", border=ACCENT_CYAN, border_width=1.8, title_color=ACCENT_CYAN)

draw_card(14.5, 2.0, 1.3, 2.3, "Command\nCenter", "Live Streamlit\nPriority Queue\nIncident PDF",
          bg="#16221d", border=ACCENT_GREEN, border_width=1.8, title_color=ACCENT_GREEN)

# Connect Tiers to Outputs
ax.annotate("", xy=(14.5, 5.9), xytext=(14.1, 5.9),
            arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=1.5))
ax.annotate("", xy=(14.5, 3.1), xytext=(14.1, 3.1),
            arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, lw=1.5))

# ── Footer / Empirical Note ──
footer_text = "Empirical Validation: Monotonic Damage Rates across Bins | SAR + Optical Fusion eliminates false alarms (0% FPR vs 25% FPR in Optical-Only)"
ax.text(8.0, 0.45, footer_text, color=TEXT_MUTED, fontsize=8.5, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#111820", edgecolor="#223042", linewidth=1))

os.makedirs("outputs", exist_ok=True)
out_path = "outputs/system_architecture.png"
plt.tight_layout()
plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print("Saved system architecture image to:", out_path)
