"""
scripts/make_architecture_figures.py — TALVAD-X architecture schematics for the paper.

Renders TWO standalone, modern flat-vector graphical abstracts (IEEE / top-tier-ML
figure style), per the diagram spec in docs/PAPER_PROMPT.md:

  architecture_tier12.png — Tiers 1+2:  shared FROZEN feature extraction (audio /
                            appearance / spectral)  →  supervised TALVAD-X backbone
                            (cross-modal attention, CMIS-2, BiGRU, localization +
                            clip heads, optional DANN).
  architecture_tier3.png  — Tier 3:     target-blind zero-shot specialist ensemble
                            (correspondence prior / AVSR head / SyncNet) + the
                            supervised clip score  →  label-free rank fusion.

Pure matplotlib (no graphviz dep) so it runs in the same env as the other paper
figures. This is a SCHEMATIC — nothing is inferred from data.

Color code (restrained, consistent across both figures):
  BLUE = audio · GREEN = video/visual · PURPLE = spectral/frequency ·
  ORANGE/AMBER = fusion & decision · GREY dashed = frozen / pretrained (❄).

Usage:
  python scripts/make_architecture_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "visualizations" / "figures"

# ── palette (fill, edge) ─────────────────────────────────────────────────────
C = {
    "audio":    ("#dbeafe", "#2563eb"),   # blue
    "video":    ("#dcfce7", "#16a34a"),   # green
    "spectral": ("#ede9fe", "#7c3aed"),   # purple
    "fusion":   ("#ffedd5", "#ea580c"),   # orange / amber
    "neutral":  ("#eef2f7", "#64748b"),   # slate
    "decision": ("#ffe4cc", "#c2410c"),   # emphasized orange
}
PANEL = {"t1": "#f5f8fc", "t2": "#fbf7f4", "t3": "#f7faf8"}
SHADOW = [pe.withSimplePatchShadow(offset=(1.5, -1.5), alpha=0.16)]
INK = "#0f172a"


# ── primitives ───────────────────────────────────────────────────────────────

def panel(ax, x0, y0, x1, y1, header, color, accent):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.02,rounding_size=2.2",
                                facecolor=color, edgecolor="#dbe3ec", lw=1.4, zorder=0))
    ax.text(x0 + 2.2, y1 - 2.4, header, ha="left", va="center", fontsize=11,
            fontweight="bold", color=accent, zorder=1)


def box(ax, x, y, w, h, text, kind="neutral", *, dashed=False, frozen=False,
        fs=9.5, bold=False, lw=1.7):
    fc, ec = C[kind]
    if frozen:
        dashed = True
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                           boxstyle="round,pad=0.02,rounding_size=1.1",
                           linewidth=lw, edgecolor=ec, facecolor=fc,
                           linestyle=(0, (4, 2.6)) if dashed else "solid", zorder=3)
    if not dashed:
        patch.set_path_effects(SHADOW)
    ax.add_patch(patch)
    ax.text(x, y, ("❄ " + text) if frozen else text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", color=INK,
            zorder=4, linespacing=1.35)
    return {"x": x, "y": y, "w": w, "h": h}


def tag(ax, b, text, color):
    ax.text(b["x"], b["y"] - b["h"] / 2 - 1.3, text, ha="center", va="top",
            fontsize=8.4, style="italic", color=color, zorder=4)


def R(b): return (b["x"] + b["w"] / 2, b["y"])
def L(b): return (b["x"] - b["w"] / 2, b["y"])
def T(b): return (b["x"], b["y"] + b["h"] / 2)
def B(b): return (b["x"], b["y"] - b["h"] / 2)


def arrow(ax, p0, p1, *, color="#475569", dashed=False, lw=1.7, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15,
                                 connectionstyle=f"arc3,rad={rad}", linewidth=lw,
                                 color=color, zorder=2,
                                 linestyle=(0, (4, 2.6)) if dashed else "solid"))


def title_band(ax, x0, x1, ytop, title, sub):
    ax.add_patch(FancyBboxPatch((x0, ytop - 4.0), x1 - x0, 4.0,
                                boxstyle="round,pad=0.02,rounding_size=1.0",
                                facecolor="#0f172a", edgecolor="none", zorder=5))
    ax.text((x0 + x1) / 2, ytop - 2.0, title, ha="center", va="center",
            fontsize=15.5, fontweight="bold", color="white", zorder=6)
    ax.text((x0 + x1) / 2, ytop - 5.6, sub, ha="center", va="center",
            fontsize=9.8, color="#64748b", zorder=6)


def legend_row(ax, cx, y):
    items = [("Audio", C["audio"][1]), ("Video / visual", C["video"][1]),
             ("Spectral / freq.", C["spectral"][1]), ("Fusion / decision", C["fusion"][1]),
             ("❄ frozen / pretrained", "#94a3b8")]
    gap = 30
    x = cx - gap * (len(items) - 1) / 2
    for txt, col in items:
        ax.add_patch(FancyBboxPatch((x - 1.0, y - 0.9), 1.8, 1.8,
                                    boxstyle="round,pad=0.02,rounding_size=0.5",
                                    facecolor=col, edgecolor="none", zorder=4))
        ax.text(x + 1.6, y, txt, ha="left", va="center", fontsize=8.8, color="#334155", zorder=4)
        x += gap


def _canvas(xlim=(0, 162), ylim=(0, 96)):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


# ── Figure A : Tiers 1 + 2 ───────────────────────────────────────────────────

def fig_tier12(out_path: Path):
    fig, ax = _canvas()
    title_band(ax, 3, 159, 95,
               "TALVAD-X — Frozen Feature Extraction  →  Supervised Backbone",
               "AV clip (T = 48 frames @ 12 fps)   ·   audio | appearance | spectral   →   in-domain detection + localization")

    panel(ax, 4, 22, 77, 85, "TIER 1 · shared frozen feature extraction", PANEL["t1"], "#475569")
    panel(ax, 81, 22, 159, 85, "TIER 2 · supervised TALVAD-X backbone", PANEL["t2"], "#c2410c")
    legend_row(ax, 81, 13)

    # --- Tier 1 ---
    inp  = box(ax, 12, 53, 13, 9, "AV clip\nT=48 @ 12 fps", "neutral", bold=True, fs=9)
    mfcc = box(ax, 34, 72, 20, 7, "MFCC + Δ + ΔΔ\n(120-d)", "audio", fs=9)
    w2v  = box(ax, 34, 62, 20, 7, "wav2vec2-base\n→ PCA-256", "audio", frozen=True, fs=9)
    afeat= box(ax, 53, 67, 13, 8, "Audio feat.\n(376-d)", "audio", bold=True, fs=9)
    eff  = box(ax, 34, 48, 23, 8, "MTCNN face crop →\nEfficientNet-B0 → PCA-256\n(appearance 256-d)",
               "video", frozen=True, fs=8.2)
    spec = box(ax, 34, 35, 23, 8, "Enriched spectral FFT\nradial×angular + HF-power + phase\n(34-d)",
               "spectral", fs=8.2)
    xt   = box(ax, 67, 52, 17, 13, "Per-frame feature\nx_t  ∈  ℝ⁶⁶⁶\n(audio 376 |\napp. 256 | spec. 34)",
               "neutral", bold=True, fs=8.6)

    arrow(ax, R(inp), L(mfcc), rad=0.16)
    arrow(ax, R(inp), L(w2v), rad=0.05)
    arrow(ax, R(inp), L(eff), rad=-0.05)
    arrow(ax, R(inp), L(spec), rad=-0.18)
    arrow(ax, R(mfcc), (afeat["x"], afeat["y"] + 2.6), rad=-0.10, color="#2563eb")
    arrow(ax, R(w2v),  (afeat["x"], afeat["y"] - 2.6), rad=0.10, color="#2563eb")
    arrow(ax, R(afeat), (xt["x"], xt["y"] + 4.5), rad=-0.10)
    arrow(ax, R(eff),  (xt["x"] - xt["w"] / 2, xt["y"]), rad=0.05)
    arrow(ax, R(spec), (xt["x"] - xt["w"] / 2, xt["y"] - 3.5), rad=0.14)

    # --- Tier 2 ---
    cma  = box(ax, 97, 68, 23, 8, "Cross-Modal Attention Fusion\nbidirectional A↔V · residual+LN → 128-d",
               "fusion", fs=8.2)
    cmis = box(ax, 97, 48, 25, 13,
               "CMIS-2 · inconsistency-gated\nspectral attention\n"
               "incon_t = 1 − cos(z_a , z_v)\ngates spectral bands →\n[ fused ‖ spec-64 ‖ incon-16 ]",
               "decision", bold=True, fs=8.0, lw=2.3)
    gru  = box(ax, 124, 58, 20, 8, "BiGRU-1 (256) → Drop+LN\n→ BiGRU-2 (128)", "neutral", fs=8.6)
    frame= box(ax, 148, 70, 22, 8, "Frame head (TFLH)\nper-frame prob → Localization", "video", fs=8.2)
    clip = box(ax, 148, 51, 22, 8, "Temporal-attention pool → MLP\nclip  REAL / FAKE  score",
               "decision", bold=True, fs=8.4)
    dann = box(ax, 148, 36, 22, 6.5, "DANN domain head (GRL)\ndomain-generalization · optional",
               "neutral", dashed=True, fs=8.0)

    arrow(ax, R(xt), (cma["x"], cma["y"] - 4), rad=0.10)
    arrow(ax, R(xt), L(cmis), rad=-0.04)
    arrow(ax, B(cma), T(cmis), color="#ea580c")
    arrow(ax, R(cma), (gru["x"], gru["y"] + 4), rad=-0.10)
    arrow(ax, R(cmis), (gru["x"], gru["y"] - 4), rad=0.12)
    arrow(ax, R(gru), L(frame), rad=-0.10)
    arrow(ax, R(gru), L(clip), rad=0.04)
    arrow(ax, B(clip), T(dann), dashed=True, color="#94a3b8")

    # localization timeline strip above the frame head
    sx, sw, sy = 137, 22, 79
    ax.add_patch(FancyBboxPatch((sx, sy - 1.1), sw, 2.2, boxstyle="round,pad=0.02,rounding_size=0.4",
                                facecolor="#e2e8f0", edgecolor="#94a3b8", lw=1, zorder=3))
    ax.add_patch(FancyBboxPatch((sx + sw * 0.44, sy - 1.1), sw * 0.26, 2.2,
                                boxstyle="round,pad=0.02,rounding_size=0.4",
                                facecolor="#16a34a", edgecolor="none", zorder=4))
    ax.text(sx + sw / 2, sy + 2.3, "forged interval", ha="center", va="bottom",
            fontsize=7.6, color="#16a34a", zorder=4)
    arrow(ax, T(frame), (sx + sw / 2, sy - 1.2), color="#16a34a", lw=1.3)

    fig.savefig(out_path, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out_path}")


# ── Figure B : Tier 3 ────────────────────────────────────────────────────────

def fig_tier3(out_path: Path):
    fig, ax = _canvas()
    title_band(ax, 3, 159, 95,
               "TALVAD-X — Target-Blind Zero-Shot Specialist Ensemble",
               "one specialist per attack type   ·   label-free rank fusion   →   cross-dataset (unseen-generator) detection")
    legend_row(ax, 81, 13)

    inp = box(ax, 14, 55, 17, 9, "AV clip\n(unseen target\ngenerator)", "neutral", bold=True, fs=9)
    sup = box(ax, 14, 32, 19, 8, "Supervised clip score\n(from Tier 2)", "decision", dashed=True, fs=8.6)

    # dashed panel around the three specialists
    px0, px1, py0, py1 = 34, 110, 22, 82
    ax.add_patch(FancyBboxPatch((px0, py0), px1 - px0, py1 - py0,
                                boxstyle="round,pad=0.02,rounding_size=2.2",
                                facecolor=PANEL["t3"], edgecolor="#9ca3af", lw=1.6,
                                linestyle=(0, (5, 3.5)), zorder=0))
    ax.text((px0 + px1) / 2, py1 - 2.6,
            "Target-blind zero-shot specialists  →  label-free rank fusion",
            ha="center", va="center", fontsize=10.8, fontweight="bold", color="#334155", zorder=1)

    corr = box(ax, 72, 66, 60, 9,
               "Correspondence prior  (self-supervised on REAL clips only)\n"
               "audio ↔ lip-motion temporal InfoNCE  →  per-frame deviation",
               "audio", fs=8.8)
    tag(ax, corr, "audio-dub specialist", "#2563eb")
    avsr = box(ax, 72, 48, 60, 9,
               "AVSR head  ·  auto_avsr / AV-HuBERT-class conformer\n(768 → pool 1536)  →  logistic regression",
               "video", frozen=True, fs=8.8)
    tag(ax, avsr, "visual-manipulation specialist", "#16a34a")
    sync = box(ax, 72, 30, 60, 9,
               "SyncNet  (lip ↔ audio embeddings)\nL2 lip-sync distance",
               "video", frozen=True, fs=8.8)
    tag(ax, sync, "lip-sync / face-swap specialist", "#16a34a")

    arrow(ax, R(inp), L(corr), rad=0.16)
    arrow(ax, R(inp), L(avsr), rad=0.0)
    arrow(ax, R(inp), L(sync), rad=-0.16)

    fuse = box(ax, 132, 50, 22, 13, "Rank-normalize →\nmean fusion\n(one specialist\nper attack type)",
               "fusion", bold=True, fs=9, lw=2.3)
    out = box(ax, 152, 50, 14, 9, "Cross-dataset\ndeepfake score", "decision", bold=True, fs=8.8)

    arrow(ax, R(corr), (fuse["x"], fuse["y"] + 5), rad=-0.08, color="#2563eb")
    arrow(ax, R(avsr), L(fuse), rad=0.0, color="#16a34a")
    arrow(ax, R(sync), (fuse["x"], fuse["y"] - 5), rad=0.08, color="#16a34a")
    arrow(ax, R(sup), (fuse["x"] - fuse["w"] / 2, fuse["y"] - 3.5), rad=-0.10,
          dashed=True, color="#c2410c")
    arrow(ax, R(fuse), L(out), color="#ea580c", lw=2.1)

    fig.savefig(out_path, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")
    fig_tier12(OUT_DIR / "architecture_tier12.png")
    fig_tier3(OUT_DIR / "architecture_tier3.png")
    print("Done. 2 architecture figures written.")


if __name__ == "__main__":
    main()
