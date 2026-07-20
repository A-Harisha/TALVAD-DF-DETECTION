"""
Publication-quality architecture diagram for TALVAD.
Run from project root:  python scripts/architecture_diagram.py
Output:  architecture_diagram.png  (300 DPI)
         architecture_diagram.pdf  (vector, for LaTeX)
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ── Colour palette ────────────────────────────────────────────────────────────
C_INPUT   = "#dbeafe"  # blue-100
C_AUD     = "#dcfce7"  # green-100
C_VID     = "#ffedd5"  # orange-100
C_GFC     = "#f3e8ff"  # purple-100
C_CONCAT  = "#f1f5f9"  # slate-100
C_TDFS    = "#e0f2fe"  # sky-100
C_FUSION  = "#ede9fe"  # violet-100
C_BIGRU   = "#cffafe"  # cyan-100
C_ATTN    = "#fef9c3"  # yellow-100
C_HEAD    = "#fee2e2"  # red-100
C_OUT     = "#d1fae5"  # emerald-100

E_NORM    = "#334155"  # slate-700
E_OPT     = "#94a3b8"  # slate-400  (optional / dashed components)
E_LOSS    = "#dc2626"  # red-600    (loss annotation)

FONT      = "Arial"

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 16))
ax.set_xlim(0, 11)
ax.set_ylim(0, 16)
ax.axis("off")
fig.patch.set_facecolor("white")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


# ── Helper: rounded box ───────────────────────────────────────────────────────
def box(x, y, w, h, title, sub="", fc="white", ec=E_NORM,
        fs=8.5, fw="bold", lw=1.4, ls="-", zorder=3):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.07",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, linestyle=ls, zorder=zorder,
    )
    ax.add_patch(p)
    cx, cy = x + w / 2, y + h / 2
    if sub:
        ax.text(cx, cy + h * 0.14, title, ha="center", va="center",
                fontsize=fs, fontweight=fw, fontfamily=FONT, color="#1e293b", zorder=zorder+1)
        ax.text(cx, cy - h * 0.18, sub, ha="center", va="center",
                fontsize=fs - 1.5, fontfamily=FONT, color="#475569",
                style="italic", zorder=zorder+1)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=fs, fontweight=fw, fontfamily=FONT, color="#1e293b", zorder=zorder+1)


# ── Helper: arrow (start-x, start-y → end-x, end-y) ─────────────────────────
def arr(x0, y0, x1, y1, lc=E_NORM, lw=1.2, ls="-", rad=0.0):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        zorder=5,
        arrowprops=dict(
            arrowstyle="-|>",
            color=lc, lw=lw, linestyle=ls,
            mutation_scale=10,
            connectionstyle=f"arc3,rad={rad}",
        ),
    )


def line(x0, y0, x1, y1, lc=E_NORM, lw=1.0, ls="-"):
    ax.plot([x0, x1], [y0, y1], color=lc, lw=lw, linestyle=ls, zorder=4)


def dim(x, y, text, color="#64748b", fs=7.2, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs,
            fontfamily=FONT, color=color, zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def note(x, y, text, color="#ef4444", fs=7.0, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs,
            fontfamily=FONT, color=color, style="italic", zorder=6)


# ═══════════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(5.5, 15.5, "TALVAD: Temporal Audio-Visual Deepfake Detector",
        ha="center", va="center", fontsize=12, fontweight="bold",
        fontfamily=FONT, color="#0f172a")
ax.text(5.5, 15.1,
        "MFCC+Δ+ΔΔ  ·  ResNet-18  ·  GFC  ·  TDFS  ·  Cross-Modal Attention  ·  BiGRU",
        ha="center", va="center", fontsize=8, fontfamily=FONT, color="#64748b")

# Divider
line(0.4, 14.85, 10.6, 14.85, lc="#e2e8f0", lw=1.0)

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT VIDEO
# ═══════════════════════════════════════════════════════════════════════════════
box(2.0, 14.1, 7.0, 0.5, "Input Video", fc=C_INPUT, fs=9)

# Arrows from input → branches (split at y=14.1)
arr(4.0, 14.1, 2.5, 13.45)   # → Audio Prep
arr(7.0, 14.1, 8.5, 13.45)   # → Video Prep

# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING BRANCH LABELS
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(0.35, 13.7, "Audio\nStream", ha="center", va="center",
        fontsize=7.5, fontfamily=FONT, color="#16a34a", fontweight="bold",
        rotation=90)
ax.text(10.65, 13.7, "Video\nStream", ha="center", va="center",
        fontsize=7.5, fontfamily=FONT, color="#ea580c", fontweight="bold",
        rotation=90)

# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
box(0.5, 12.9, 4.0, 0.55, "Audio Preprocessing",
    sub="16 kHz mono · pre-emphasis 0.97", fc=C_AUD, ec="#16a34a", lw=1.2, fs=8)

box(5.5, 12.9, 5.0, 0.55, "Video Preprocessing",
    sub="2 fps · 224×224 · RGB", fc=C_VID, ec="#ea580c", lw=1.2, fs=8)

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
arr(2.5, 12.9, 2.5, 12.3)
arr(7.0, 12.9, 7.0, 12.3)

# MFCC box
box(0.5, 11.4, 4.0, 0.9,
    "MFCC + Δ + ΔΔ",
    sub="librosa  ·  n_mfcc=40  ·  3× stacked",
    fc=C_AUD, ec="#16a34a", lw=1.2, fs=8.5)
dim(2.5, 11.1, "[T × 120]", color="#15803d")

# ResNet-18 box
box(5.5, 11.4, 3.1, 0.9,
    "ResNet-18",
    sub="ImageNet pretrained · frozen · penultimate",
    fc=C_VID, ec="#ea580c", lw=1.2, fs=8.5)
dim(7.05, 11.1, "[T × 512]", color="#c2410c")

# GFC box
box(8.8, 11.4, 1.8, 0.9,
    "GFC",
    sub="8-band\nFFT energy",
    fc=C_GFC, ec="#7c3aed", lw=1.2, fs=8)
dim(9.7, 11.1, "[T × 8]", color="#6d28d9")

# GFC annotation
note(9.7, 12.3, "Global Freq.\nConsistency", color="#7c3aed", fs=6.5)

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE CONCATENATION
# ═══════════════════════════════════════════════════════════════════════════════
# Lines from features → concat node at y=10.8
arr(2.5, 11.4, 2.5, 10.75)         # MFCC down
arr(7.05, 11.4, 7.05, 10.75)       # ResNet down
arr(9.7, 11.4, 9.7, 10.95)         # GFC diagonal

# Horizontal lines at y=10.8 converging to center
line(2.5, 10.8, 5.5, 10.8)
line(7.05, 10.8, 5.5, 10.8)

# GFC joins at y=11.0
line(9.7, 11.0, 9.7, 10.8)
line(9.7, 10.8, 7.05, 10.8)

# Concat node (dashed box)
box(1.8, 10.35, 7.4, 0.45,
    "Concatenate  [ audio ‖ video ‖ GFC ]",
    fc=C_CONCAT, ec="#94a3b8", lw=1.0, ls="--", fw="normal", fs=8.5)

line(5.5, 10.8, 5.5, 10.8)
arr(5.5, 10.8, 5.5, 10.35+0.45)   # arrow into concat box top

dim(5.5, 10.1, "[T × 640]  =  120 + 512 + 8", color="#475569", fs=7.5)

# ═══════════════════════════════════════════════════════════════════════════════
# TDFS
# ═══════════════════════════════════════════════════════════════════════════════
arr(5.5, 10.35, 5.5, 9.85)

box(1.2, 9.25, 8.6, 0.6,
    "TDFS — Temporal Differentiable Feature Selector",
    sub="BiGRU context net  ->  importance head  ->  sigmoid(logit / T)   per-timestep soft mask   L1 sparsity",
    fc=C_TDFS, ec="#0284c7", lw=1.6, fs=9, fw="bold")

# L1 loss annotation
note(10.1, 9.55, "L1\nloss", color="#dc2626", fs=7)
arr(9.8, 9.55, 9.8, 9.55, lc="#dc2626")  # dummy (just for visual)
line(9.8, 9.55, 10.0, 9.55, lc="#dc2626", lw=0.8, ls="--")

# GBWSO seed arrow (from left side)
note(0.22, 9.1, "GBWSO\nprior\n(seed)", color="#7c3aed", fs=6.5)
ax.annotate("", xy=(1.2, 9.55), xytext=(0.65, 9.1),
            zorder=5,
            arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=1.0,
                            linestyle="--", mutation_scale=8))

# ═══════════════════════════════════════════════════════════════════════════════
# SPLIT BACK INTO AUDIO / VIDEO BEFORE CROSS-ATTENTION
# ═══════════════════════════════════════════════════════════════════════════════
arr(5.5, 9.25, 5.5, 8.9)

# Small split label
dim(5.5, 8.73, "split: [T×120] audio  ‖  [T×520] video", color="#475569", fs=7.0)

# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-MODAL ATTENTION FUSION
# ═══════════════════════════════════════════════════════════════════════════════
arr(3.5, 8.62, 3.5, 8.32)   # audio side down
arr(7.5, 8.62, 7.5, 8.32)   # video side down

box(1.0, 7.35, 9.0, 0.97,
    "Cross-Modal Attention Fusion",
    sub="audio queries → video keys/values  |  video queries → audio keys/values  ·  4 heads  ·  LayerNorm + residual",
    fc=C_FUSION, ec="#7c3aed", lw=1.8, fs=9.5, fw="bold")

# Internal bidirectional arrows inside the fusion box
ax.annotate("", xy=(5.7, 7.83), xytext=(4.2, 7.83),
            zorder=6,
            arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=1.0, mutation_scale=8))
ax.annotate("", xy=(4.2, 7.63), xytext=(5.7, 7.63),
            zorder=6,
            arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=1.0, mutation_scale=8))
ax.text(3.3, 7.83, "audio → video", ha="center", va="center",
        fontsize=6.5, fontfamily=FONT, color="#6d28d9")
ax.text(6.7, 7.63, "video → audio", ha="center", va="center",
        fontsize=6.5, fontfamily=FONT, color="#6d28d9")

# A/V Agreement loss annotation
note(10.1, 7.83, "A/V\nAgreement\nloss", color="#dc2626", fs=6.5)
line(10.0, 7.83, 10.0, 7.83, lc="#dc2626", lw=0.8)
line(10.0, 7.83, 10.35, 7.83, lc="#dc2626", lw=0.8, ls="--")

# ═══════════════════════════════════════════════════════════════════════════════
# BIGRU STACK
# ═══════════════════════════════════════════════════════════════════════════════
arr(5.5, 7.35, 5.5, 6.9)
dim(5.5, 6.73, "[T × hidden]", color="#0e7490", fs=7.2)

box(2.0, 6.15, 7.0, 0.58,
    "Bidirectional GRU Stack",
    sub="BiGRU-1 (hidden=128, bidir) → Dropout → LayerNorm → BiGRU-2 (hidden=64, bidir)",
    fc=C_BIGRU, ec="#0e7490", lw=1.6, fs=9, fw="bold")

dim(5.5, 5.97, "[T × 256]  →  [T × 128]", color="#0e7490", fs=7.2)

# ═══════════════════════════════════════════════════════════════════════════════
# BRANCH SPLIT: Temporal Attention + Frame Head
# ═══════════════════════════════════════════════════════════════════════════════
arr(5.5, 6.15, 5.5, 5.85)

# Left branch: Temporal Attention
line(5.5, 5.85, 2.8, 5.85)
arr(2.8, 5.85, 2.8, 5.5)

# Right branch: TFLH (optional, dashed)
line(5.5, 5.85, 8.2, 5.85)
arr(8.2, 5.85, 8.2, 5.5, lc=E_OPT)

# Temporal Attention box
box(0.5, 4.75, 4.5, 0.75,
    "Temporal Attention",
    sub="score = Linear([T,H]→[T,1]) · softmax → context [H]",
    fc=C_ATTN, ec="#b45309", lw=1.4, fs=8.5)

# TFLH box (dashed, optional)
box(5.7, 4.75, 4.8, 0.75,
    "Frame-Level Head  (TFLH)",
    sub="Linear([T,H]→[T,1]) · per-timestep forgery logits",
    fc=C_HEAD, ec=E_OPT, lw=1.2, ls="--", fs=8, fw="bold")

ax.text(10.45, 5.13, "optional", ha="center", va="center",
        fontsize=6.5, fontfamily=FONT, color=E_OPT, style="italic", rotation=90)

# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION HEADS
# ═══════════════════════════════════════════════════════════════════════════════
arr(2.75, 4.75, 2.75, 4.2)
arr(8.1, 4.75, 8.1, 4.2, lc=E_OPT)

dim(2.75, 4.08, "[B × 1]", color="#92400e", fs=7.2)
dim(8.1, 4.08, "[B × T × 1]", color=E_OPT, fs=7.2)

# Video-level output
box(0.5, 3.15, 4.5, 0.9,
    "Video-Level Classifier",
    sub="FC(H→64) → GELU → FC(64→1) → Sigmoid  ·  BCEWithLogitsLoss",
    fc=C_OUT, ec="#059669", lw=1.5, fs=8.5)

# Frame-level output (dashed)
box(5.7, 3.15, 4.8, 0.9,
    "Frame Forgery Logits",
    sub="Binary cross-entropy vs. fake_periods  ·  weight λ=0.05",
    fc=C_HEAD, ec=E_OPT, lw=1.2, ls="--", fs=8)

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════
arr(2.75, 3.15, 2.75, 2.55)
arr(8.1, 3.15, 8.1, 2.55, lc=E_OPT)

box(1.0, 2.0, 3.5, 0.55, "Real / Fake  (video-level)",
    fc="#bbf7d0", ec="#059669", lw=1.8, fs=9, fw="bold")

box(6.3, 2.0, 3.4, 0.55, "Forgery Segments  (optional)",
    fc="#fecaca", ec=E_OPT, lw=1.2, ls="--", fs=8.5)

# ═══════════════════════════════════════════════════════════════════════════════
# LOSS SUMMARY FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
line(0.4, 1.72, 10.6, 1.72, lc="#e2e8f0", lw=0.8)
ax.text(5.5, 1.52,
        "Losses:  (1) BCE (video-level)   "
        "(2) lambda * BCE (frame-level, TFLH)   "
        "(3) L1 TDFS sparsity   "
        "(4) alpha * cosine (A/V agreement: real=high sim, fake=low sim)",
        ha="center", va="center", fontsize=7.3, fontfamily=FONT, color="#475569",
        style="italic")

line(0.4, 1.28, 10.6, 1.28, lc="#e2e8f0", lw=0.8)
ax.text(5.5, 1.08,
        "Augmentations:  temporal masking  ·  modality dropout  ·  Gaussian feature noise  ·  label smoothing",
        ha="center", va="center", fontsize=7.3, fontfamily=FONT, color="#475569",
        style="italic")

# ═══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (C_AUD,    "#16a34a", "Audio stream"),
    (C_VID,    "#ea580c", "Video stream"),
    (C_GFC,    "#7c3aed", "GFC (frequency)"),
    (C_TDFS,   "#0284c7", "TDFS (feature selection)"),
    (C_FUSION, "#7c3aed", "Cross-modal attention"),
    (C_BIGRU,  "#0e7490", "Temporal modelling"),
    (C_HEAD,   E_OPT,     "Optional component"),
]

lx, ly = 0.5, 0.68
for i, (fc, ec, label) in enumerate(legend_items):
    xi = lx + i * 1.43
    p = FancyBboxPatch((xi, ly - 0.13), 0.22, 0.22,
                       boxstyle="round,pad=0.03",
                       facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=5)
    ax.add_patch(p)
    ax.text(xi + 0.28, ly + 0.02, label,
            fontsize=6.5, fontfamily=FONT, color="#334155",
            va="center", zorder=6)

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════
for path in (Path("architecture_diagram.png"), Path("architecture_diagram.pdf")):
    fig.savefig(path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {path.resolve()}")

plt.close(fig)
