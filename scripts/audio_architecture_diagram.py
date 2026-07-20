"""
scripts/audio_architecture_diagram.py

Generates a research-quality diagram of the TALVAD audio feature extraction pipeline.
Saves: audio_architecture_diagram.png (300 DPI) and audio_architecture_diagram.pdf

Pipeline covered:
  Raw Audio → Preprocessing → STFT → Mel Filterbank → MFCC → Delta/DeltaDelta
  → Concatenation → Temporal Alignment → Z-score Norm → [T, 120] Tensor
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

# ── Colour palette (matches main architecture_diagram.py) ────────────────────
C = {
    "audio":   "#2ecc71",   # green  — audio inputs / outputs
    "proc":    "#3498db",   # blue   — preprocessing steps
    "spec":    "#9b59b6",   # purple — spectral analysis
    "mel":     "#e67e22",   # orange — mel filterbank
    "mfcc":    "#e74c3c",   # red    — MFCC / DCT
    "delta":   "#f39c12",   # yellow — temporal derivatives
    "concat":  "#1abc9c",   # teal   — concatenation
    "align":   "#3498db",   # blue   — alignment/normalisation
    "output":  "#8e44ad",   # dark purple — output tensor
    "formula": "#2c3e50",   # near-black  — formula text
    "dim":     "#7f8c8d",   # grey   — dimension labels
    "bg":      "#ffffff",   # white  — background
    "arrow":   "#2c3e50",   # dark   — arrows
    "wave":    "#2980b9",   # wave colour
}

FIG_W, FIG_H = 10, 20
FONT = "DejaVu Sans"
BOX_W = 6.5
BOX_H = 0.72
X_CTR = FIG_W / 2


def _font(size, bold=False):
    return {"family": FONT, "size": size, "weight": "bold" if bold else "normal"}


def box(ax, y_top, label, sublabel="", color="#3498db", alpha=0.85,
        dashed=False, width=BOX_W, font_size=11):
    x0 = X_CTR - width / 2
    style = "round,pad=0.05"
    lw = 1.5 if not dashed else 1.2
    ls = "--" if dashed else "-"
    rect = FancyBboxPatch(
        (x0, y_top - BOX_H), width, BOX_H,
        boxstyle=style, linewidth=lw, linestyle=ls,
        edgecolor=color, facecolor=color, alpha=alpha,
        zorder=3,
    )
    ax.add_patch(rect)
    y_ctr = y_top - BOX_H / 2
    ax.text(X_CTR, y_ctr + (0.12 if sublabel else 0),
            label, ha="center", va="center", zorder=4,
            color="white", fontdict=_font(font_size, bold=True))
    if sublabel:
        ax.text(X_CTR, y_ctr - 0.16, sublabel, ha="center", va="center",
                zorder=4, color="white", fontdict=_font(8.5))
    return y_top - BOX_H


def arr(ax, y_from, y_to, label="", color=None):
    color = color or C["arrow"]
    ax.annotate(
        "", xy=(X_CTR, y_to + 0.04), xytext=(X_CTR, y_from),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
        zorder=2,
    )
    if label:
        ax.text(X_CTR + 0.2, (y_from + y_to) / 2, label,
                ha="left", va="center", fontdict=_font(8), color=C["dim"])


def dim_label(ax, y, text, side="right"):
    xpos = X_CTR + BOX_W / 2 + 0.18 if side == "right" else X_CTR - BOX_W / 2 - 0.18
    ha = "left" if side == "right" else "right"
    ax.text(xpos, y - BOX_H / 2, text,
            ha=ha, va="center", fontdict=_font(8.5, bold=True), color=C["dim"],
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C["dim"], alpha=0.6))


def note(ax, y, text, color=C["formula"]):
    ax.text(X_CTR, y, text, ha="center", va="center",
            fontdict=_font(8), color=color, style="italic")


def _draw_waveform(ax, x_ctr, y_ctr, w=4.0, h=0.35, color=C["wave"]):
    """Stylised sine + noise waveform."""
    t = np.linspace(0, 4 * np.pi, 300)
    wave = np.sin(t) * 0.6 + 0.25 * np.sin(3 * t) + 0.08 * np.random.RandomState(7).randn(300)
    wave = wave / np.abs(wave).max() * (h / 2)
    x_vals = x_ctr - w / 2 + t / (4 * np.pi) * w
    ax.plot(x_vals, y_ctr + wave, color=color, lw=1.2, zorder=5, alpha=0.85)
    ax.axhline(y_ctr, xmin=(x_ctr - w / 2) / FIG_W, xmax=(x_ctr + w / 2) / FIG_W,
               color=color, lw=0.6, alpha=0.4, linestyle="--")


def _draw_spectrogram(ax, x_ctr, y_ctr, w=4.0, h=0.35):
    """Mini fake spectrogram heatmap."""
    rng = np.random.RandomState(42)
    data = rng.exponential(0.5, (20, 60))
    data[:5, :] *= 2          # brighter low-freq
    left = x_ctr - w / 2
    bottom = y_ctr - h / 2
    ax.imshow(data, extent=[left, left + w, bottom, bottom + h],
              aspect="auto", origin="lower", cmap="viridis",
              alpha=0.85, zorder=5, vmin=0, vmax=2)


def _draw_mfcc_heatmap(ax, x_ctr, y_ctr, w=4.0, h=0.35, n_coef=40):
    """Mini fake MFCC heatmap."""
    rng = np.random.RandomState(17)
    data = rng.randn(n_coef, 50)
    data[0, :] += 3
    left = x_ctr - w / 2
    bottom = y_ctr - h / 2
    ax.imshow(data, extent=[left, left + w, bottom, bottom + h],
              aspect="auto", origin="lower", cmap="RdBu_r",
              alpha=0.85, zorder=5, vmin=-3, vmax=3)


def build_diagram():
    np.random.seed(0)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    GAP = 0.30   # vertical gap between boxes

    # ── Title ────────────────────────────────────────────────────────────────
    ax.text(X_CTR, FIG_H - 0.35, "TALVAD — Audio Feature Extraction Pipeline",
            ha="center", va="top", fontdict=_font(14, bold=True), color=C["formula"])
    ax.text(X_CTR, FIG_H - 0.75, "MFCC + Delta + Delta-Delta  →  [T, 120] Audio Feature Tensor",
            ha="center", va="top", fontdict=_font(9), color=C["dim"])

    y = FIG_H - 1.10

    # ── 1. Raw Audio ─────────────────────────────────────────────────────────
    y = box(ax, y, "Raw Audio Signal",
            sublabel="WAV, 16 kHz, mono, 1-10 sec",
            color=C["audio"])
    dim_label(ax, y + BOX_H, "1-D signal")
    _draw_waveform(ax, X_CTR, y + BOX_H / 2, w=4.5, h=0.28)
    arr(ax, y, y - GAP)
    note(ax, y - GAP / 2 + 0.02, "Sample rate: 16,000 Hz")
    y -= GAP

    # ── 2. Preemphasis ───────────────────────────────────────────────────────
    y = box(ax, y, "Preemphasis Filter",
            sublabel="y[n] = x[n] - 0.97 * x[n-1]   (high-freq boost)",
            color=C["proc"])
    arr(ax, y, y - GAP)
    note(ax, y - GAP / 2 + 0.02, "Compensates spectral tilt; improves high-freq SNR")
    y -= GAP

    # ── 3. STFT ──────────────────────────────────────────────────────────────
    y = box(ax, y, "Short-Time Fourier Transform (STFT)",
            sublabel="Hann window | n_fft=2048 | hop=512 samples (32 ms stride)",
            color=C["spec"])
    dim_label(ax, y + BOX_H, "[T_frames, 1025]")
    _draw_spectrogram(ax, X_CTR, y + BOX_H / 2, w=4.5, h=0.28)
    arr(ax, y, y - GAP)
    note(ax, y - GAP / 2 + 0.02, "Freq resolution: 7.8 Hz/bin  |  Frame rate: 31.25 fps")
    y -= GAP

    # ── 4. Mel Filterbank ────────────────────────────────────────────────────
    y = box(ax, y, "Mel-Scale Filter Bank",
            sublabel="40 triangular filters  |  log(|Mel * S|^2 + 1e-9)",
            color=C["mel"])
    dim_label(ax, y + BOX_H, "[T_frames, 40]")
    arr(ax, y, y - GAP)
    note(ax, y - GAP / 2 + 0.02, "Maps linear Hz to perceptual mel scale")
    y -= GAP

    # ── 5. DCT → MFCCs ──────────────────────────────────────────────────────
    y = box(ax, y, "Discrete Cosine Transform (DCT) → MFCCs",
            sublabel="40 cepstral coefficients  |  log-energy decorrelation",
            color=C["mfcc"])
    dim_label(ax, y + BOX_H, "[T, 40]")
    _draw_mfcc_heatmap(ax, X_CTR, y + BOX_H / 2, w=4.5, h=0.28)
    arr(ax, y, y - GAP)
    y -= GAP

    # ── 6. Delta coefficients ────────────────────────────────────────────────
    # Show as two side-by-side sub-boxes
    half_w = (BOX_W - 0.2) / 2
    for i, (label, sub, color) in enumerate([
        ("Delta (Velocity)", "d/dt MFCC[t]  width=2", C["delta"]),
        ("Delta-Delta (Accel.)", "d^2/dt^2 MFCC[t]  width=2", C["delta"]),
    ]):
        xoff = X_CTR - BOX_W / 2 + i * (half_w + 0.2)
        rect = FancyBboxPatch(
            (xoff, y - BOX_H), half_w, BOX_H,
            boxstyle="round,pad=0.05", linewidth=1.5,
            edgecolor=color, facecolor=color, alpha=0.85, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(xoff + half_w / 2, y - BOX_H / 2 + 0.13, label,
                ha="center", va="center", zorder=4,
                color="white", fontdict=_font(9.5, bold=True))
        ax.text(xoff + half_w / 2, y - BOX_H / 2 - 0.13, sub,
                ha="center", va="center", zorder=4,
                color="white", fontdict=_font(7.5))
    # dimension labels
    for i, lbl in enumerate(["[T, 40]", "[T, 40]"]):
        xoff = X_CTR - BOX_W / 2 + i * (half_w + 0.2) + half_w + 0.05
        ax.text(xoff, y - BOX_H / 2, lbl,
                ha="left", va="center", fontdict=_font(8, bold=True), color=C["dim"])
    y -= BOX_H
    arr(ax, y, y - GAP)
    note(ax, y - GAP / 2 + 0.02, "Temporal gradient captures dynamics & phonetic transitions")
    y -= GAP

    # ── 7. Concatenation ─────────────────────────────────────────────────────
    y = box(ax, y, "Feature Concatenation",
            sublabel="[MFCC | Delta | DeltaDelta]   axis=-1",
            color=C["concat"])
    dim_label(ax, y + BOX_H, "[T, 120]")
    arr(ax, y, y - GAP)
    note(ax, y - GAP / 2 + 0.02, "40 + 40 + 40 = 120 dimensions per timestep")
    y -= GAP

    # ── 8. Temporal alignment ────────────────────────────────────────────────
    y = box(ax, y, "Temporal Resampling to Video FPS",
            sublabel="Align MFCC frames (31 fps) -> video fps=2  |  linear interpolation",
            color=C["align"])
    dim_label(ax, y + BOX_H, "[T_video, 120]")
    arr(ax, y, y - GAP)
    note(ax, y - GAP / 2 + 0.02, "Ensures audio and video feature arrays share same T axis")
    y -= GAP

    # ── 9. Z-score Normalization ─────────────────────────────────────────────
    y = box(ax, y, "Z-score Normalisation",
            sublabel="Global: mean/std fit on training set   (FeatureFusionNormalizer)",
            color=C["proc"])
    dim_label(ax, y + BOX_H, "[T, 120]")
    arr(ax, y, y - GAP)
    y -= GAP

    # ── 10. Output tensor ────────────────────────────────────────────────────
    y = box(ax, y, "Audio Feature Tensor",
            sublabel="Input to CrossModalAttentionFusion  (audio_dim=120)",
            color=C["output"])
    dim_label(ax, y + BOX_H, "[B, T, 120]")

    # ── Annotation: goes into the full model ─────────────────────────────────
    ax.annotate(
        "Concatenated with\nVideo [B,T,512] + GFC [B,T,8]\n→ Full input [B,T,640]",
        xy=(X_CTR + BOX_W / 2 + 0.05, y + BOX_H / 2),
        xytext=(X_CTR + BOX_W / 2 + 1.5, y + BOX_H / 2),
        ha="left", va="center",
        fontsize=7.5, color=C["output"],
        arrowprops=dict(arrowstyle="->", color=C["output"], lw=1.2),
    )

    # ── Footer: parameter table ───────────────────────────────────────────────
    footer_y = y - 0.55
    ax.axhline(footer_y + 0.3, color=C["dim"], lw=0.8, alpha=0.5,
               xmin=0.05, xmax=0.95)
    params = [
        ("Sample rate", "16,000 Hz"),
        ("n_mfcc", "40"),
        ("n_fft", "2,048  (128 ms)"),
        ("hop_length", "512  (32 ms)"),
        ("Preemphasis", "alpha = 0.97"),
        ("Video FPS", "2"),
        ("Sequence length", "24 frames  (12 sec)"),
        ("Output dim", "120  (40 x 3)"),
    ]
    ncol = 4
    col_w = FIG_W / ncol
    for i, (k, v) in enumerate(params):
        col = i % ncol
        row = i // ncol
        xpos = col * col_w + col_w / 2
        ypos = footer_y - row * 0.32
        ax.text(xpos, ypos + 0.05, k,
                ha="center", va="bottom", fontdict=_font(7.5, bold=True), color=C["formula"])
        ax.text(xpos, ypos - 0.05, v,
                ha="center", va="top", fontdict=_font(7.5), color=C["dim"])

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_entries = [
        ("Raw / Output", C["audio"]),
        ("Preprocessing", C["proc"]),
        ("Spectral Analysis", C["spec"]),
        ("Mel Filterbank", C["mel"]),
        ("MFCC / DCT", C["mfcc"]),
        ("Temporal Derivatives", C["delta"]),
    ]
    legend_y = 0.22
    col_w2 = FIG_W / len(legend_entries)
    for i, (label, color) in enumerate(legend_entries):
        xp = i * col_w2 + col_w2 / 2
        patch = mpatches.Patch(color=color, alpha=0.85)
        ax.legend(
            handles=[patch], labels=[label],
            loc="lower left",
            bbox_to_anchor=(xp / FIG_W - 0.06, legend_y / FIG_H - 0.01),
            bbox_transform=fig.transFigure,
            frameon=False,
            fontsize=7,
            handlelength=0.8,
        )

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    return fig


if __name__ == "__main__":
    out_png = Path("audio_architecture_diagram.png")
    out_pdf = Path("audio_architecture_diagram.pdf")
    print("Generating audio architecture diagram...")
    fig = build_diagram()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved PNG -> {out_png.resolve()}")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"  Saved PDF -> {out_pdf.resolve()}")
    plt.close(fig)
    print("Done.")
