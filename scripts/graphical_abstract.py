"""
Graphical abstract / architecture overview for TALVAD (replaces the stale
scripts/architecture_diagram.py, which depicted an obsolete MFCC/ResNet/BiGRU
model). Horizontal left-to-right pipeline matching the paper's Methodology:

  Input video  ->  four frozen speech/artifact backbones (PCA, cached)
               ->  three Genuine Audio-Visual Correspondence signals
                   c(t) level (CMIS-2), e(t) plausibility (CMRD), b(t) seams (CBM)
               ->  fused A-V features + anchor-free 4-level temporal pyramid (E)
               ->  classification (C), regression (G, DFL), attention-pooled clip head
               ->  temporal forgery localization  +  video-level Real/Fake

Run from project root:  python scripts/graphical_abstract.py
Outputs (vector + raster):  paper/figures/architecture_diagram.pdf / .png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# ── palette ───────────────────────────────────────────────────────────────────
C_INPUT  = ("#dbeafe", "#2563eb")   # blue
C_FROZEN = ("#eef2f7", "#64748b")   # slate (frozen backbones)
C_CMIS   = ("#e0e7ff", "#4f46e5")   # indigo  -> c(t) level
C_CMRD   = ("#dcfce7", "#16a34a")   # green   -> e(t) plausibility
C_CBM    = ("#ffe4e6", "#e11d48")   # rose    -> b(t) seams
C_ENC    = ("#ede9fe", "#7c3aed")   # violet  -> encoder
C_HEAD   = ("#cffafe", "#0e7490")   # cyan    -> heads
C_LOC    = ("#fef9c3", "#ca8a04")   # amber   -> localization
C_DET    = ("#d1fae5", "#059669")   # emerald -> detection

AUD = "#d97706"   # audio-derived accent
VIS = "#0891b2"   # visual-derived accent
INK = "#0f172a"
SUB = "#475569"

fig, ax = plt.subplots(figsize=(14.2, 7.4))
ax.set_xlim(0, 14.2)
ax.set_ylim(0, 7.4)
ax.axis("off")
fig.patch.set_facecolor("white")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


def box(x, y, w, h, fc, ec, title="", sub="", fs=9.5, subfs=7.6,
        fw="bold", lw=1.6, ls="-", tcol=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.045",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                linestyle=ls, zorder=3))
    cx = x + w / 2
    tcol = tcol or INK
    if sub:
        ax.text(cx, y + h * 0.63, title, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=tcol, zorder=4)
        ax.text(cx, y + h * 0.28, sub, ha="center", va="center",
                fontsize=subfs, color=SUB, style="italic", zorder=4)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=tcol, zorder=4)


def arrow(x0, y0, x1, y1, ec="#334155", lw=1.6, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                 arrowstyle="-|>", mutation_scale=13, lw=lw, color=ec,
                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=5))


def header(x, text):
    ax.text(x, 6.85, text, ha="center", va="center", fontsize=10,
            fontweight="bold", color="#1e293b")


def dot(x, y, col, label):
    ax.add_patch(plt.Circle((x, y), 0.055, color=col, zorder=6))
    ax.text(x + 0.13, y, label, ha="left", va="center", fontsize=6.6,
            color=col, fontweight="bold", zorder=6)


# ── title ─────────────────────────────────────────────────────────────────────
ax.text(7.1, 7.18, "TALVAD — Genuine Audio–Visual Correspondence for "
        "Deepfake Detection and Temporal Localization",
        ha="center", va="center", fontsize=12.5, fontweight="bold", color=INK)

# ── stage headers ─────────────────────────────────────────────────────────────
header(1.15, "Input")
header(3.9,  r"Frozen backbones  $\mathfrak{B}$")
header(6.95, "Genuine A–V Correspondence")
header(10.0, r"Encoder  $\mathfrak{E}$  &  heads")
header(12.9, "Outputs")

# ── stage A: input ────────────────────────────────────────────────────────────
box(0.25, 3.2, 1.85, 1.25, *C_INPUT, title="Input video",
    sub=r"$v=(\Phi,\,h)$  ·  visual + audio", fs=10, subfs=7.0)

# ── stage B: four frozen streams ──────────────────────────────────────────────
streams = [
    (5.15, "Voice",      "WavLM-large / PCA  (632)",      AUD, "audio"),
    (3.95, "Lip motion", "auto_avsr Conformer / PCA (384)", VIS, "visual"),
    (2.75, "Appearance", "EfficientNet-B0 / PCA  (256)",  VIS, "visual"),
    (1.55, "Spectral",   "radial×angular FFT + phase (34)", VIS, "visual"),
]
bx, bw, bh = 2.55, 2.75, 0.95
for y, name, sub, acc, _ in streams:
    box(bx, y, bw, bh, *C_FROZEN, title=name, sub=sub, fs=9, subfs=6.9)
    # modality accent tab + frozen marker
    ax.add_patch(Rectangle((bx, y), 0.09, bh, color=acc, zorder=4))
    ax.text(bx + bw - 0.12, y + bh - 0.16, "❄", ha="center", va="center",
            fontsize=8, color="#64748b", zorder=5)
ax.text(3.9, 0.95, "PCA-reduced · cached · duration-normalized  "
        r"$t=192$", ha="center", va="center", fontsize=7.4,
        color=SUB, style="italic")

# input -> streams (fan)
for y, *_ in streams:
    arrow(2.12, 3.83, 2.53, y + bh / 2, lw=1.3)

# ── stage C: three GAVC signals ───────────────────────────────────────────────
sigs = [
    (4.72, "CMIS-2", r"level  $c(t)$", "pre-fusion voice–lip agreement (InfoNCE)", C_CMIS),
    (3.22, "CMRD",   r"plausibility  $e(t)$", "real-only reconstruction anomaly", C_CMRD),
    (1.72, "CBM",    r"seams  $b(t)$", "correspondence-change boundary energy", C_CBM),
]
sx, sw, sh = 5.55, 2.75, 1.05
for y, tag, sig, sub, col in sigs:
    box(sx, y, sw, sh, *col, title=f"{tag}  ·  {sig}", sub=sub, fs=9, subfs=6.9)

# correspondence streams (voice, lip) -> all three signals, via a bus
busx = 5.42
ax.plot([busx, busx], [2.25, 5.62], color="#94a3b8", lw=1.4, zorder=2)
for y, *_ in streams[:2]:                       # voice + lip feed the bus
    ax.plot([bx + bw, busx], [y + bh / 2, y + bh / 2], color="#94a3b8", lw=1.4, zorder=2)
for y, *_ in sigs:                              # bus feeds each signal box
    arrow(busx, y + sh / 2, sx, y + sh / 2, lw=1.3, ec="#64748b")

# artifact streams (appearance, spectral) -> fused features at the encoder,
# routed through the clean margins (below the signal row) to stay visible
mlx, mrx, lowy = 5.33, 8.55, 1.12
dash = dict(color="#94a3b8", lw=1.3, ls=(0, (4, 3)), zorder=6)
ax.plot([bx + bw, mlx], [2.75 + bh / 2, 2.75 + bh / 2], **dash)   # appearance -> rail
ax.plot([bx + bw, mlx], [1.55 + bh / 2, 1.55 + bh / 2], **dash)   # spectral   -> rail
ax.plot([mlx, mlx], [2.75 + bh / 2, lowy], **dash)               # left rail down
ax.plot([mlx, mrx], [lowy, lowy], **dash)                        # low bus across
ax.plot([mrx, mrx], [lowy, 4.30], **dash)                        # right rail up
arrow(mrx, 4.30, 8.70, 4.30, lw=1.3, ec="#94a3b8")               # into encoder
ax.text((mlx + mrx) / 2, lowy - 0.2, "appearance + spectral  →  fused features",
        ha="center", va="center", fontsize=6.6, color="#94a3b8", style="italic")

# ── stage D: encoder + heads ──────────────────────────────────────────────────
box(8.7, 4.0, 2.6, 1.35, *C_ENC,
    title=r"Fused A–V feats $\mathbf{x}\!\to\!\mathfrak{E}$",
    sub="anchor-free · 4-level temporal pyramid", fs=9, subfs=6.9)
box(8.7, 2.05, 2.6, 1.35, *C_HEAD,
    title="Prediction heads",
    sub=r"$\mathfrak{C}\!\to\!\hat p$ · $\mathfrak{G}\!\to\!\hat b$ (DFL) · "
        r"$\mathfrak{C}_{\mathrm{clip}}\!\to\!\hat y$", fs=9, subfs=7.2)

# signals bus -> encoder
sbus = 8.5
ax.plot([sbus, sbus], [2.25, 5.25], color="#94a3b8", lw=1.4, zorder=2)
for y, *_ in sigs:
    ax.plot([sx + sw, sbus], [y + sh / 2, y + sh / 2], color="#94a3b8", lw=1.4, zorder=2)
arrow(sbus, 4.675, 8.68, 4.675, lw=1.4, ec="#64748b")
arrow(9.99, 4.0, 9.99, 3.42, lw=1.6)            # encoder -> heads

# ── stage E: outputs ──────────────────────────────────────────────────────────
box(11.75, 4.0, 2.15, 1.35, *C_LOC, title="Temporal forgery", sub="localization", fs=9.5)
# timeline glyph inside the localization box
ax.add_patch(Rectangle((11.95, 4.18), 1.75, 0.16, color="#e5e7eb", zorder=4))
ax.add_patch(Rectangle((12.55, 4.18), 0.62, 0.16, color="#dc2626", zorder=5))
ax.text(12.86, 3.98, "fake segment", ha="center", va="center", fontsize=6.2,
        color="#b91c1c", zorder=5)

box(11.75, 2.05, 2.15, 1.35, *C_DET, title="Video-level", sub="Real / Fake", fs=9.5)

arrow(11.3, 3.0, 11.73, 4.55, lw=1.6, ec=C_LOC[1], rad=0.15)   # heads -> localization
arrow(11.3, 2.72, 11.73, 2.72, lw=1.6, ec=C_DET[1])            # heads -> detection

# ── legend for the three GAVC signals (color = identity) ──────────────────────
lx, ly = 0.35, 0.42
for i, (_, tag, sig, _, col) in enumerate(sigs):
    xi = lx + i * 3.3
    ax.add_patch(FancyBboxPatch((xi, ly - 0.11), 0.26, 0.22,
                 boxstyle="round,pad=0.02", facecolor=col[0], edgecolor=col[1],
                 linewidth=1.2, zorder=5))
    ax.text(xi + 0.36, ly, f"{tag} — {sig.split('$')[1] if '$' in sig else sig}",
            ha="left", va="center", fontsize=7.4, color="#334155", zorder=6)
ax.text(lx + 3 * 3.3 + 0.2, ly, "❄ frozen · PCA-reduced backbones",
        ha="left", va="center", fontsize=7.4, color="#64748b", zorder=6)

# ── save ──────────────────────────────────────────────────────────────────────
out = Path(__file__).resolve().parent.parent / "paper" / "figures"
out.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(out / f"architecture_diagram.{ext}", dpi=300,
                bbox_inches="tight", facecolor="white")
    print(f"saved {out / f'architecture_diagram.{ext}'}")
plt.close(fig)
