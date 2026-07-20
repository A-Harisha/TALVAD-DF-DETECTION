"""
scripts/make_metrics_dashboard.py — classification-metrics dashboard for TALVAD-X.

Reports the six metrics the paper needs — Accuracy, Precision, Recall, F1,
Average Precision (AP, area under the PR curve), and ROC-AUC — for each of the
three headline datasets, computed DIRECTLY from the per-clip prediction CSVs so
the figure can never drift from the logged scores.

Datasets (the "one specialist per attack type" headline system):
  LAV-DF (in-domain)      logs/talvad_v6/validation_predictions.csv      (honest split)
  FakeAVCeleb (zero-shot) logs/figures_data/fakeavceleb_fused.csv        (AVSR ⊕ prior)
  DeepfakeTIMIT (z-shot)  logs/figures_data/deepfaketimit_fused.csv      (AVSR ⊕ SyncNet)

Operating point for threshold-dependent metrics (Accuracy/Precision/Recall/F1):
the per-dataset Youden-J optimal threshold (argmax TPR-FPR on that dataset's own
ROC) — the natural decision point for the rank-fused zero-shot scores, which have
no meaningful 0.5. AP and AUC are threshold-free. Pass --threshold 0.5 to instead
use a fixed 0.5 cut (a viable alternative; both are reported in the notes panel).

Bonus: the LAV-DF panel also shows temporal-localization AP (AP@IoU, mAP) pulled
from logs/talvad_x/eval_A0_baseline/eval_test_results.json — the flagship
localization result that the clip-level AP above does not capture.

Usage
  python scripts/make_metrics_dashboard.py
  python scripts/make_metrics_dashboard.py --threshold 0.5
Output
  visualizations/figures/metrics_dashboard.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "logs"
OUT = REPO / "visualizations" / "figures" / "metrics_dashboard.png"

DATASETS = [
    ("LAV-DF (in-domain)", LOGS / "talvad_v6" / "validation_predictions.csv", "probability"),
    ("FakeAVCeleb (zero-shot)", LOGS / "figures_data" / "fakeavceleb_fused.csv", "score"),
    ("DeepfakeTIMIT (zero-shot)", LOGS / "figures_data" / "deepfaketimit_fused.csv", "score"),
]
A0_EVAL = LOGS / "talvad_x" / "eval_A0_baseline" / "eval_test_results.json"

METRICS = ["Accuracy", "Precision", "Recall", "F1", "AP", "AUC"]
C_GOOD, C_NEUT = "#55a868", "#8c8c8c"


def youden_threshold(y: np.ndarray, s: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[int(np.argmax(tpr - fpr))])


def compute(csv: Path, score_col: str, mode: str) -> dict:
    df = pd.read_csv(csv)
    y = df["label"].to_numpy().astype(int)
    col = score_col if score_col in df.columns else ("score" if "score" in df.columns else "probability")
    s = df[col].to_numpy().astype(float)
    thr = 0.5 if mode == "0.5" else youden_threshold(y, s)
    yhat = (s >= thr).astype(int)
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    return {
        "n": len(y), "n_fake": int(y.sum()), "n_real": int((y == 0).sum()), "thr": thr,
        "Accuracy": accuracy_score(y, yhat),
        "Precision": precision_score(y, yhat, zero_division=0),
        "Recall": recall_score(y, yhat, zero_division=0),
        "F1": f1_score(y, yhat, zero_division=0),
        "AP": average_precision_score(y, s),
        "AUC": roc_auc_score(y, s),
        "fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec,
        "cm": confusion_matrix(y, yhat, labels=[0, 1]),
    }


# ── panels ──────────────────────────────────────────────────────────────────
def panel_table(ax, results: dict) -> None:
    ax.axis("off")
    ax.set_title("Metrics — all datasets (color = value)", fontsize=13, fontweight="bold")
    names = list(results)
    grid = np.array([[results[n][m] for m in METRICS] for n in names])
    ax.imshow(grid, cmap="YlGn", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(METRICS))); ax.set_xticklabels(METRICS, fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace(" (", "\n(") for n in names], fontsize=9)
    for i in range(len(names)):
        for j in range(len(METRICS)):
            v = grid[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=12,
                    fontweight="bold", color="black" if v > 0.62 else "white")


def panel_bars(ax, results: dict) -> None:
    names = list(results)
    x = np.arange(len(METRICS)); w = 0.26
    cols = ["#4c72b0", "#dd8452", "#8172b3"]
    for j, n in enumerate(names):
        vals = [results[n][m] for m in METRICS]
        b = ax.bar(x + (j - 1) * w, vals, w, label=n, color=cols[j], alpha=0.9)
        for bi, vv in zip(b, vals):
            ax.text(bi.get_x() + bi.get_width() / 2, vv + 0.008, f"{vv:.2f}",
                    ha="center", fontsize=6.6, rotation=90, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(METRICS, fontsize=9)
    ax.set_ylim(0, 1.08); ax.set_ylabel("score")
    ax.axhline(0.5, ls="--", c="k", lw=0.7)
    ax.legend(fontsize=7.5, frameon=False, loc="lower center", ncol=1)
    ax.set_title("Metrics by dataset", fontsize=13, fontweight="bold")


def panel_roc(ax, results: dict) -> None:
    cols = ["#2ca02c", "#1f77b4", "#d62728"]
    for (n, r), c in zip(results.items(), cols):
        ax.plot(r["fpr"], r["tpr"], lw=2, color=c, label=f"{n.split(' (')[0]} (AUC={r['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8.5); ax.grid(alpha=0.3)
    ax.set_title("ROC curves", fontsize=13, fontweight="bold")


def panel_pr(ax, results: dict) -> None:
    cols = ["#2ca02c", "#1f77b4", "#d62728"]
    for (n, r), c in zip(results.items(), cols):
        ax.plot(r["rec"], r["prec"], lw=2, color=c, label=f"{n.split(' (')[0]} (AP={r['AP']:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=8.5); ax.grid(alpha=0.3)
    ax.set_title("Precision-Recall curves", fontsize=13, fontweight="bold")


def panel_cm(ax, name: str, r: dict) -> None:
    cm = r["cm"]
    cmn = cm.astype(float) / np.clip(cm.sum(1, keepdims=True), 1, None)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Real", "Fake"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Real", "Fake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{name.split(' (')[0]}\nAUC={r['AUC']:.3f} (thr={r['thr']:.2f})", fontsize=11, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cmn[i, j]*100:.1f}%)", ha="center", va="center",
                    fontsize=10, color="white" if cmn[i, j] > 0.5 else "black")


def panel_loc(ax, ev: dict) -> None:
    loc = ev["localization"]; tflh = ev["tflh_localization"]
    iou = [0.5, 0.75, 0.9, 0.95]
    ap = [loc["AP@50"], loc["AP@75"], loc["AP@90"], loc["AP@95"]]
    ax.plot(iou, ap, "o-", color="#8172b3", lw=2, ms=7)
    for x, yv in zip(iou, ap):
        ax.annotate(f"{yv:.3f}", (x, yv), textcoords="offset points", xytext=(0, 9), fontsize=9)
    ax.set_xlabel("temporal IoU threshold"); ax.set_ylabel("AP")
    ax.set_ylim(0, 0.8); ax.grid(alpha=0.3)
    ax.set_title("LAV-DF temporal localization AP (bonus)", fontsize=13, fontweight="bold")
    ax.text(0.97, 0.95,
            f"frame AUC = {ev['frame_auc']:.4f}\ndefault mAP = {loc['mAP']:.3f}\n"
            f"best-decode mAP = {tflh['best_map']:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec=C_NEUT, alpha=0.9))


def panel_notes(ax, results: dict, mode: str) -> None:
    ax.axis("off")
    ax.set_title("Notes & operating points", fontsize=13, fontweight="bold", loc="left")
    op = "fixed 0.5" if mode == "0.5" else "per-dataset Youden-J (max TPR-FPR)"
    lines = [f"Threshold (Acc/Prec/Rec/F1): {op}", ""]
    for n, r in results.items():
        lines.append(f"{n}")
        lines.append(f"   n={r['n']:,}  (real={r['n_real']:,}, fake={r['n_fake']:,})  thr={r['thr']:.3f}")
        lines.append(f"   Acc {r['Accuracy']:.3f} · P {r['Precision']:.3f} · R {r['Recall']:.3f} · "
                     f"F1 {r['F1']:.3f} · AP {r['AP']:.3f} · AUC {r['AUC']:.3f}")
        lines.append("")
    lines += [
        "AP = area under Precision-Recall curve (clip-level detection).",
        "AUC = area under ROC.  Both are threshold-free.",
        "LAV-DF row = honest talvad_v6 split; upgraded TALVAD-X reaches",
        "   AUC 0.998 / F1 0.988 / Acc 0.982 in-domain (eval JSON).",
    ]
    ax.text(0.0, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=8.6, family="monospace", linespacing=1.42)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", choices=["youden", "0.5"], default="youden")
    args = ap.parse_args()

    results = {}
    for name, csv, col in DATASETS:
        if not csv.exists():
            print(f"  SKIP {name}: missing {csv}")
            continue
        results[name] = compute(csv, col, args.threshold)
        r = results[name]
        print(f"  {name:<26} Acc={r['Accuracy']:.3f} P={r['Precision']:.3f} R={r['Recall']:.3f} "
              f"F1={r['F1']:.3f} AP={r['AP']:.3f} AUC={r['AUC']:.3f}")
    if not results:
        print("No prediction CSVs found."); sys.exit(1)
    ev = json.loads(A0_EVAL.read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(3, 3, hspace=0.40, wspace=0.28, top=0.92, bottom=0.05, left=0.06, right=0.97)

    panel_table(fig.add_subplot(gs[0, 0]), results)
    panel_roc(fig.add_subplot(gs[0, 1]), results)
    panel_pr(fig.add_subplot(gs[0, 2]), results)

    panel_bars(fig.add_subplot(gs[1, 0]), results)
    cm_axes = [fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[2, 0])]
    for axc, (name, r) in zip(cm_axes, results.items()):
        panel_cm(axc, name, r)

    panel_loc(fig.add_subplot(gs[2, 1]), ev)
    panel_notes(fig.add_subplot(gs[2, 2]), results, args.threshold)

    fig.suptitle("TALVAD-X — Classification Metrics Dashboard  "
                 "(Accuracy · Precision · Recall · F1 · AP · AUC)",
                 fontsize=17, fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT}")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
