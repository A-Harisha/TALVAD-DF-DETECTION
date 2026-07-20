"""Generate TALVAD paper figures from verified log numbers.
Run from the repo root: .venv/Scripts/python.exe scripts/make_paper_figures.py
Outputs vector PDFs (+PNG previews) into figures/ (paths are relative to the
current working directory, so always invoke this from the repo root).
Every value is traceable to logs/ (see comments)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({
    "font.size": 9, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 200,
})
# print-friendly, colourblind-safe
C = {"ours": "#0B5394", "a": "#6FA8DC", "b": "#F6B26B",
     "c": "#93C47D", "d": "#C27BA0", "grey": "#999999"}

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote figures/%s.pdf" % name)

# ---------------------------------------------------------------------------
# Fig. 1 — LAV-DF localization SOTA (docs/SOTA_COMPARISON.md; ours: loc_eval_test.json)
# Curated representative set; AP@0.5 / AP@0.75 / AP@0.95.
methods = ["BA-TFD+", "UMMAFormer", "DiMoDif", "AuViRe", "EVAS",
           "UniSkip-M.", "TALVAD-L"]
ap50 = [96.30, 98.83, 95.50, 98.9, 98.41, 98.6, 98.98]
ap75 = [84.96, 95.54, 87.91, 96.0, 96.61, 97.1, 96.26]
ap95 = [4.44, 37.61, 20.61, 46.5, 88.63, 63.4, 70.25]
x = np.arange(len(methods)); w = 0.26
fig, ax = plt.subplots(figsize=(7.0, 2.9))
bars = [
    ax.bar(x - w, ap50, w, label="AP@0.5", color=C["a"]),
    ax.bar(x, ap75, w, label="AP@0.75", color=C["b"]),
    ax.bar(x + w, ap95, w, label="AP@0.95", color=C["c"]),
]
# highlight ours
for grp in bars:
    grp[-1].set_edgecolor(C["ours"]); grp[-1].set_linewidth(1.8)
ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20, ha="right")
ax.set_ylabel("Average Precision (%)"); ax.set_ylim(0, 105)
ax.set_title("Temporal forgery localization on LAV-DF test", fontsize=10)
ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.42))
for xi, v in zip(x + w, ap95):
    ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=6.5)
save(fig, "fig_localization_sota")

# ---------------------------------------------------------------------------
# Fig. 2 — What transfers: fingerprint collapses, correspondence transfers.
# in-domain fingerprint 0.998 (SOTA table); zero-shot fingerprint 0.444 (gate);
# correspondence prior 0.753 bal (correspondence_eval.json);
# full TALVAD-L +instnorm 0.894 (crossdataset_fakeavceleb_bal_instnorm.json).
labels = ["Fingerprint\n(in-domain\nLAV-DF)", "Fingerprint\n(zero-shot\nFakeAVCeleb)",
          "Correspondence\nprior\n(zero-shot)", "TALVAD-L\nfull\n(zero-shot)"]
vals = [0.998, 0.444, 0.753, 0.894]
cols = [C["grey"], C["d"], C["b"], C["ours"]]
fig, ax = plt.subplots(figsize=(4.3, 3.0))
bars = ax.bar(labels, vals, color=cols, width=0.66)
ax.axhline(0.5, ls="--", lw=1, color="k", alpha=0.6)
ax.text(3.35, 0.515, "chance", fontsize=7, ha="right", va="bottom", alpha=0.7)
ax.set_ylabel("Detection AUC"); ax.set_ylim(0, 1.05)
ax.set_title("What transfers across generators", fontsize=10)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.3f}",
            ha="center", fontsize=8, fontweight="bold")
save(fig, "fig_what_transfers")

# ---------------------------------------------------------------------------
# Fig. 3 — Per-fake-type complementarity (logs/fusion_avsr_eval.json).
types = ["FakeVid+\nFakeAud", "FakeVid+\nRealAud", "RealVid+\nFakeAud (dub)", "Overall"]
avsr = [0.933, 0.928, 0.579, 0.814]
prior = [0.820, 0.608, 0.832, 0.753]
fused = [0.949, 0.871, 0.804, 0.875]
x = np.arange(len(types)); w = 0.26
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.bar(x - w, avsr, w, label="AVSR head", color=C["a"])
ax.bar(x, prior, w, label="Correspondence prior", color=C["b"])
ax.bar(x + w, fused, w, label="Label-free fusion", color=C["ours"])
ax.axhline(0.5, ls="--", lw=0.8, color="k", alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(types, fontsize=8)
ax.set_ylabel("Detection AUC"); ax.set_ylim(0, 1.05)
ax.set_title("Complementarity on FakeAVCeleb by fake type", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower left")
# annotate the dub collapse/rescue
ax.annotate("AVSR blind\nto dubs", xy=(2 - w, 0.579), xytext=(2 - w - 0.15, 0.30),
            fontsize=6.5, ha="center",
            arrowprops=dict(arrowstyle="->", lw=0.7))
save(fig, "fig_pertype")
print("done")
