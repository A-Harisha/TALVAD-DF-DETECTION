"""
scripts/plot_robustness.py — AuViRe-style robustness grid (their Fig. 3):
3x4 panel grid, one panel per distortion (7 visual, cyan tint; 4 audio, yellow
tint) plus an Overall Average panel (green tint). Each panel plots AP@0.5 and
clip AUC (%) vs distortion level 0..5, level 0 = clean baseline. Series are
distinguished by color AND marker/linestyle (CVD-safe secondary encoding);
single shared y-axis semantics ("Performance (%)"), shared legend below.

Reads logs/robustness/<distortion>_L<level>.json written by eval_robustness_l.py
(keys: distortion, level, AP@50, clip_auc). clean_L0.json (or clean.json) is the
level-0 point for every panel.

Usage:
  python scripts/plot_robustness.py                       # -> figures/fig_robustness.pdf/.png
  python scripts/plot_robustness.py --demo --out <dir>    # synthetic smoke render only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
VISUAL = ["saturation", "contrast", "blockwise", "vnoise", "blur", "jpeg", "vcompress"]
AUDIO = ["anoise", "pitch", "reverb", "acompress"]
TITLES = {"saturation": "Saturation", "contrast": "Contrast", "blockwise": "Block-wise",
          "vnoise": "Gaussian Noise", "blur": "Gaussian Blur", "jpeg": "JPEG Compression",
          "vcompress": "Video Compression", "anoise": "Background Noise",
          "pitch": "Pitch Shift", "reverb": "Reverberation", "acompress": "Audio Compression"}
# panel tints + edges (visual cyan / audio yellow / overall green, as in AuViRe)
TINT = {"visual": ("#eaf6f6", "#188f8f"), "audio": ("#fdf7e0", "#c9a227"),
        "overall": ("#ecf5ec", "#3c8a3c")}
# series: color + distinct marker + distinct linestyle (identity never color-alone)
SERIES = [
    ("AP@0.5", "AP@50", "#0B5394", "o", "-"),
    ("clip AUC", "clip_auc", "#C1272D", "x", "--"),
]


def load_results(log_dir: Path) -> dict:
    """-> {distortion: {level: {metric: value}}}; 'clean' level 0 broadcast later."""
    out = {}
    for f in sorted(log_dir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out.setdefault(d["distortion"], {})[int(d["level"])] = d
    return out


def demo_results() -> dict:
    rng = np.random.default_rng(7)
    out = {"clean": {0: {"AP@50": 0.9898, "clip_auc": 0.9997}}}
    for d in VISUAL + AUDIO:
        sag = rng.uniform(0.001, 0.02)  # per-distortion degradation slope
        out[d] = {lv: {"AP@50": 0.9898 - sag * lv ** 1.5 + rng.normal(0, 0.001),
                       "clip_auc": 0.9997 - 0.7 * sag * lv ** 1.5 + rng.normal(0, 0.001)}
                  for lv in range(1, 6)}
    return out


def panel(ax, levels, series_vals, title, kind, ymin):
    face, edge = TINT[kind]
    ax.set_facecolor(face)
    for spine in ax.spines.values():
        spine.set_color(edge); spine.set_linewidth(1.4)
    for (label, _, color, marker, ls), vals in zip(SERIES, series_vals):
        ax.plot(levels, vals, color=color, marker=marker, linestyle=ls,
                linewidth=1.6, markersize=5, markerfacecolor="none",
                markeredgewidth=1.4, label=label)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Distortion Level", fontsize=8)
    ax.set_ylabel("Performance (%)", fontsize=8)
    ax.set_xticks(levels)
    ax.set_ylim(ymin, 100.4)
    ax.tick_params(labelsize=7)
    ax.grid(True, color="white", linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="logs/robustness")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--name", default="fig_robustness")
    ap.add_argument("--ymin", type=float, default=50.0)
    ap.add_argument("--demo", action="store_true", help="render synthetic data (smoke only)")
    args = ap.parse_args()

    res = demo_results() if args.demo else load_results(REPO / args.log_dir)
    if not args.demo and "clean" not in res:
        raise SystemExit("no clean baseline in logs/robustness — run run_robustness.py first")
    clean = res["clean"][0]

    order = VISUAL + AUDIO  # 11 panels + overall = 12
    fig, axes = plt.subplots(3, 4, figsize=(13.2, 8.6),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.3})
    axes = axes.ravel()
    per_metric_all = {key: [] for _, key, _, _, _ in SERIES}

    for i, dist in enumerate(order):
        kind = "visual" if dist in VISUAL else "audio"
        dres = res.get(dist, {})
        levels = [0] + sorted(dres.keys())
        series_vals = []
        for _, key, _, _, _ in SERIES:
            vals = [100.0 * clean[key]] + [100.0 * dres[lv][key] for lv in sorted(dres.keys())]
            series_vals.append(vals)
            if len(vals) == 6:  # only full sweeps enter the overall average
                per_metric_all[key].append(vals)
        panel(axes[i], levels, series_vals, TITLES[dist], kind, args.ymin)

    # overall average panel
    levels = list(range(6))
    series_vals = []
    for _, key, _, _, _ in SERIES:
        arr = np.array(per_metric_all[key]) if per_metric_all[key] else np.full((1, 6), np.nan)
        series_vals.append(arr.mean(axis=0).tolist())
    panel(axes[11], levels, series_vals, "Overall Average", "overall", args.ymin)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, [f"TALVAD {l}" for l in labels], loc="lower center",
               ncol=len(SERIES), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.95, bottom=0.09)

    out_dir = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name + ("_demo" if args.demo else "")
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {out_dir / stem}.pdf/.png" + ("  (DEMO DATA — do not publish)" if args.demo else ""))


if __name__ == "__main__":
    main()
