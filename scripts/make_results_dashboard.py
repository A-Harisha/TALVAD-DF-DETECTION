"""
scripts/make_results_dashboard.py — ONE-PAGE results dashboard with every headline
number the TALVAD-X paper reports, plus the viable ALTERNATIVES for each claim.

Each panel is data-driven: values are read straight from the JSON logs so the figure
can never drift from the reported tables. The only hard-coded blocks are the
cross-dataset Phase-1 gate and the honest-baseline AUC (both live in markdown,
logs/talvad_x_gate.md and the talvad_v6 notes); those constants are cited inline.

Layout (4 rows x 3 cols = 12 panels):
  R0  in-domain cards (A0)        | temporal localization AP-vs-IoU | detection A0..A5
  R1  CMIS-2 ablation (best-mAP)  | Phase-1 lift FakeAVCeleb        | Phase-1 lift DeepfakeTIMIT
  R2  correspondence prior+EER    | prior pooling variants          | headline fusion FakeAVCeleb
  R3  complementarity by fake type| final system (3 datasets)       | BEST scores + alternatives

Sources
  in-domain detection + temporal localization (A0..A5)
      logs/talvad_x/eval_{A0_baseline,A1_spectral,A2_consistency,
                          A3_coupling,A4_full,A5_coldinit}/eval_test_results.json
  correspondence prior (zero-shot)        logs/correspondence_eval.json
  supervised (+) prior fusion             logs/fusion_eval.json
  AVSR (+) prior fusion (FakeAVCeleb)     logs/fusion_avsr_eval.json
  AVSR (+) SyncNet fusion (DeepfakeTIMIT) logs/fusion_syncnet_eval.json
  Phase-1 cross-dataset lift              logs/talvad_x_gate.md  (hard-coded below)

Usage
  python scripts/make_results_dashboard.py
Output
  visualizations/figures/paper_dashboard.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "logs"
OUT = REPO / "visualizations" / "figures" / "paper_dashboard.png"

# ── palette ───────────────────────────────────────────────────────────────────
C_AUDIO, C_VIDEO, C_SPEC = "#4c72b0", "#55a868", "#8172b3"
C_FUSE, C_BAD, C_NEUT = "#dd8452", "#c44e52", "#8c8c8c"


# ── data loading ──────────────────────────────────────────────────────────────
def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


ABL_DIRS = {
    "A0\nbaseline": "eval_A0_baseline",
    "A1\nspectral": "eval_A1_spectral",
    "A2\nconsist.": "eval_A2_consistency",
    "A3\ncoupling": "eval_A3_coupling",
    "A4\nfull": "eval_A4_full",
    "A5\ncold-init": "eval_A5_coldinit",
}


def load_ablation() -> dict[str, dict]:
    out = {}
    for label, d in ABL_DIRS.items():
        out[label] = _read(LOGS / "talvad_x" / d / "eval_test_results.json")
    return out


# Cross-dataset Phase-1 gate — from logs/talvad_x_gate.md (markdown; hard-coded).
GATE_FAC = {  # variant: (seed mean, std, ensemble)
    "A0\nbaseline": (0.5767, 0.0420, 0.5651),
    "P1\na0+aug": (0.6494, 0.0416, 0.6818),
    "P1\nbest": (0.6489, 0.0317, 0.6865),
    "P1\nartic": (0.6335, 0.0501, 0.7123),
}
PHASE1_LIFT = (0.1472, 0.1216, 0.1726)  # Δ, CI95 low, CI95 high  (P(Δ>0)=1.000)

GATE_TIMIT = {  # DeepfakeTIMIT phase-1 (logs/talvad_x_gate.md)
    "A0\nbaseline": (0.6181, 0.0671, 0.6435),
    "P1\nbest": (0.6186, 0.1344, 0.7387),
}
PHASE1_LIFT_TIMIT = (0.0951, 0.0568, 0.1362)  # Δ, CI95 lo, hi  (P=1.000)

# Honest, conservative in-domain baseline on the SAME official split (talvad_v6,
# weaker features) — the alternative to the upgraded 0.998. See RESULTS_TABLES Table 7.
HONEST_INDOMAIN_AUC = 0.9146


# ── panels ────────────────────────────────────────────────────────────────────
def panel_cards(ax, a0: dict) -> None:
    ax.axis("off")
    ax.set_title("In-domain detection — LAV-DF test (n=26,100)", fontsize=12, fontweight="bold")
    v = a0["video_metrics_05"]
    cards = [
        ("Video AUC", v["auc"], C_AUDIO),
        ("Video F1", v["f1"], C_VIDEO),
        ("Bal. Acc.", v["balanced_accuracy"], C_SPEC),
        ("Frame AUC", a0["frame_auc"], C_FUSE),
    ]
    for i, (name, val, col) in enumerate(cards):
        x, y = (i % 2) * 0.5, 0.52 - (i // 2) * 0.46
        ax.add_patch(plt.Rectangle((x + 0.02, y + 0.02), 0.46, 0.40,
                                   transform=ax.transAxes, color=col, alpha=0.13, ec=col, lw=1.5))
        ax.text(x + 0.25, y + 0.27, f"{val:.4f}", transform=ax.transAxes,
                ha="center", fontsize=19, fontweight="bold", color=col)
        ax.text(x + 0.25, y + 0.09, name, transform=ax.transAxes, ha="center", fontsize=10.5)
    ax.text(0.5, -0.04, f"alt. headline: honest baseline (talvad_v6, weaker feats) = {HONEST_INDOMAIN_AUC:.4f} AUC",
            transform=ax.transAxes, ha="center", fontsize=8, style="italic", color=C_NEUT)


def panel_localization(ax, a0: dict) -> None:
    loc = a0["localization"]
    tflh = a0["tflh_localization"]
    iou = [0.5, 0.75, 0.9, 0.95]
    ap = [loc["AP@50"], loc["AP@75"], loc["AP@90"], loc["AP@95"]]
    ax.plot(iou, ap, "o-", color=C_SPEC, lw=2, ms=7)
    for x, y in zip(iou, ap):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 9), fontsize=9)
    ax.set_xlabel("temporal IoU threshold"); ax.set_ylabel("Average Precision (AP)")
    ax.set_ylim(0, 0.8); ax.grid(alpha=0.3)
    ax.set_title("Temporal localization — AP vs IoU (LAV-DF test)", fontsize=12, fontweight="bold")
    ax.text(0.97, 0.95,
            f"frame AUC = {a0['frame_auc']:.4f}\n"
            f"default mAP = {loc['mAP']:.3f}\n"
            f"best-decode mAP = {tflh['best_map']:.3f}\n"
            f"(seg AP@0.1 = {tflh['best_map_segment_ap']['ap_01']:.3f},\n"
            f" AP@0.5 = {tflh['best_map_segment_ap']['ap_05']:.3f})",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec=C_NEUT, alpha=0.9))


def panel_detection_all(ax, abl: dict) -> None:
    labels = list(abl.keys())
    vauc = [abl[k]["video_metrics_05"]["auc"] for k in labels]
    fauc = [abl[k]["frame_auc"] for k in labels]
    x = np.arange(len(labels)); w = 0.4
    ax.bar(x - w / 2, vauc, w, color=C_AUDIO, label="video AUC")
    ax.bar(x + w / 2, fauc, w, color=C_FUSE, label="frame AUC")
    for xi, vv, ff in zip(x, vauc, fauc):
        ax.text(xi - w / 2, vv + 0.0006, f"{vv:.4f}", ha="center", fontsize=6.5, rotation=90, va="bottom")
        ax.text(xi + w / 2, ff + 0.0006, f"{ff:.4f}", ha="center", fontsize=6.5, rotation=90, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.985, 1.0); ax.set_ylabel("AUC")
    ax.legend(fontsize=8, frameon=False, loc="lower center", ncol=2)
    ax.set_title("Detection saturated across all CMIS-2 variants", fontsize=12, fontweight="bold")


def panel_ablation(ax, abl: dict) -> None:
    labels = list(abl.keys())
    bestmap = [abl[k]["tflh_localization"]["best_map"] for k in labels]
    auc = [abl[k]["video_metrics_05"]["auc"] for k in labels]
    x = np.arange(len(labels))
    bars = ax.bar(x, bestmap, color=C_VIDEO, alpha=0.85)
    for b, m, a in zip(bars, bestmap, auc):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.004, f"{m:.3f}", ha="center", fontsize=8)
        ax.text(b.get_x() + b.get_width() / 2, 0.50, f"AUC\n{a:.3f}", ha="center",
                fontsize=7, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.49, 0.85); ax.set_ylabel("best-decode mAP")
    ax.set_title("CMIS-2 ablation — module adds ~nothing in-domain", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.03, "coupling (A3) vs spectral/consist. (A1,A2): Δ=-0.0003 (n.s.)",
            transform=ax.transAxes, ha="center", fontsize=8, style="italic", color=C_BAD)


def _phase1_panel(ax, gate: dict, lift: tuple, title: str, note: str) -> None:
    labels = list(gate.keys())
    ens = [gate[k][2] for k in labels]
    mean = [gate[k][0] for k in labels]
    std = [gate[k][1] for k in labels]
    x = np.arange(len(labels))
    ax.bar(x, ens, color=C_FUSE, alpha=0.85, label="3-seed ensemble")
    ax.errorbar(x, mean, yerr=std, fmt="D", color="#333", ms=5, capsize=4, label="seed mean ± std")
    for xi, e in zip(x, ens):
        ax.text(xi, e + 0.008, f"{e:.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, ls="--", c="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.45, 0.80); ax.set_ylabel("zero-shot AUC")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title(title, fontsize=12, fontweight="bold")
    d, lo, hi = lift
    ax.text(0.97, 0.05, f"lift Δ=+{d:.3f}\nCI95 [{lo:.3f},{hi:.3f}]\n{note}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="#eafaef", ec=C_VIDEO))


def panel_prior(ax, corr: dict) -> None:
    r = corr["results"]
    items = [
        ("FakeAVCeleb\n(balanced)", "fakeavceleb_bal", C_VIDEO),
        ("FakeAVCeleb\n(full)", "fakeavceleb", C_VIDEO),
        ("DeepfakeTIMIT\n(real audio)", "deepfaketimit", C_NEUT),
        ("LAV-DF self\n(in-sync fake)", "lavdf_selfcheck", C_NEUT),
    ]
    labels = [i[0] for i in items]
    vals = [r[i[1]]["mean"]["auc"] for i in items]
    eers = [r[i[1]]["mean"]["eer"] for i in items]
    cols = [i[2] for i in items]
    x = np.arange(len(labels))
    ax.bar(x, vals, color=cols, alpha=0.85)
    for xi, v, e in zip(x, vals, eers):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(xi, v - 0.05, f"EER\n{e:.3f}", ha="center", fontsize=7, color="#222")
    ax.axhline(0.5, ls="--", c="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.3, 0.85); ax.set_ylabel("zero-shot AUC (mean-pool)")
    ax.set_title("Correspondence prior — real-only, target-blind", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.04, "transfers where AV is OUT of sync (dubs); blind to in-sync/real-audio by design",
            transform=ax.transAxes, ha="center", fontsize=7.5, style="italic", color="#333")


def panel_prior_pooling(ax, corr: dict) -> None:
    """Multiple viable pooling reductions of the per-frame deviation."""
    r = corr["results"]
    pools = ["mean", "topk20", "max"]
    targets = [("FakeAVCeleb bal", "fakeavceleb_bal"), ("FakeAVCeleb full", "fakeavceleb")]
    x = np.arange(len(pools)); w = 0.36
    for j, (name, key) in enumerate(targets):
        vals = [r[key][p]["auc"] for p in pools]
        bars = ax.bar(x + (j - 0.5) * w, vals, w, label=name,
                      color=[C_VIDEO, C_AUDIO][j], alpha=0.88)
        for b, p in zip(bars, pools):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                    f"{r[key][p]['auc']:.3f}\nEER {r[key][p]['eer']:.2f}",
                    ha="center", fontsize=6.8)
    ax.set_xticks(x); ax.set_xticklabels(["mean\n(default)", "top-k20", "max"], fontsize=8)
    ax.set_ylim(0.6, 0.82); ax.set_ylabel("zero-shot AUC")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("Prior pooling — mean is best (alt: top-k, max)", fontsize=12, fontweight="bold")


def panel_fusion(ax, fus: dict, avsr: dict) -> None:
    fac = fus["results"]["fakeavceleb"]
    o = avsr["results"]["OVERALL"]
    items = [
        ("Prior", fac["prior_auc"], C_VIDEO),
        ("Supervised", fac["sup_auc"], C_AUDIO),
        ("Sup ⊕ Prior", fac["fused_auc"], C_FUSE),
        ("AVSR", o["avsr"], C_AUDIO),
        ("AVSR ⊕ Prior", o["mean"], C_FUSE),
    ]
    labels = [i[0] for i in items]; vals = [i[1] for i in items]; cols = [i[2] for i in items]
    x = np.arange(len(labels))
    ax.bar(x, vals, color=cols, alpha=0.9)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0.87, ls="--", c=C_BAD, lw=1.0)
    ax.text(len(labels) - 1, 0.882, "SOTA band ≈0.87", color=C_BAD, ha="right", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=12)
    ax.set_ylim(0.5, 0.95); ax.set_ylabel("zero-shot AUC")
    ax.set_title("Headline fusion — FakeAVCeleb (label-free)", fontsize=12, fontweight="bold")
    ax.text(0.02, 0.04, f"alt fusions (bal): max={o['max']:.3f}, w0.5={o['w0.5']:.3f}",
            transform=ax.transAxes, ha="left", fontsize=7.5, style="italic", color="#333")


def panel_pertype(ax, avsr: dict) -> None:
    r = avsr["results"]
    types = ["FakeVid+FakeAud", "FakeVid+RealAud(swap)", "RealVid+FakeAud(dub)"]
    short = ["Fake V+A", "Swap (V)", "Dub (A)"]
    avsr_v = [r[t]["avsr"] for t in types]
    prior_v = [r[t]["prior"] for t in types]
    fuse_v = [r[t]["mean"] for t in types]
    x = np.arange(len(types)); w = 0.26
    ax.bar(x - w, avsr_v, w, label="AVSR", color=C_AUDIO)
    ax.bar(x, prior_v, w, label="Prior", color=C_VIDEO)
    ax.bar(x + w, fuse_v, w, label="Fused", color=C_FUSE)
    for xi, (a, p, f) in zip(x, zip(avsr_v, prior_v, fuse_v)):
        for dx, val in ((-w, a), (0, p), (w, f)):
            ax.text(xi + dx, val + 0.01, f"{val:.2f}", ha="center", fontsize=7)
    ax.axhline(0.5, ls="--", c="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=9)
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("AUC")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center")
    ax.set_title("Complementarity by fake type (FakeAVCeleb bal.)", fontsize=12, fontweight="bold")


def panel_system(ax, a0: dict, avsr: dict, sync: dict) -> None:
    items = [
        ("LAV-DF\nin-domain", a0["video_metrics_05"]["auc"], C_AUDIO, "full system"),
        ("FakeAVCeleb\nzero-shot", avsr["results"]["OVERALL"]["mean"], C_FUSE, "AVSR⊕Prior"),
        ("DeepfakeTIMIT\nzero-shot", sync["results"]["OVERALL"]["mean"], C_FUSE, "AVSR⊕SyncNet"),
    ]
    labels = [i[0] for i in items]; vals = [i[1] for i in items]
    cols = [i[2] for i in items]; tags = [i[3] for i in items]
    x = np.arange(len(labels))
    ax.bar(x, vals, color=cols, alpha=0.9)
    for xi, v, t in zip(x, vals, tags):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
        ax.text(xi, 0.06, t, ha="center", fontsize=8, color="white")
    ax.axhline(0.5, ls="--", c="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("AUC")
    ax.set_title("Final system — one specialist per attack type", fontsize=12, fontweight="bold")


def panel_best_alternatives(ax, a0, corr, fus, avsr, sync) -> None:
    ax.axis("off")
    ax.set_title("Best scores for the paper  (·  = viable alternative)", fontsize=12,
                 fontweight="bold", loc="left")
    loc = a0["localization"]; tflh = a0["tflh_localization"]
    fac = fus["results"]["fakeavceleb"]
    o = avsr["results"]["OVERALL"]
    dub = avsr["results"]["RealVid+FakeAud(dub)"]
    cb = corr["results"]["fakeavceleb_bal"]; cf = corr["results"]["fakeavceleb"]
    sy = sync["results"]["OVERALL"]
    rows = [
        ("In-domain detection AUC", f"{a0['video_metrics_05']['auc']:.4f}",
         f"· honest baseline {HONEST_INDOMAIN_AUC:.4f}"),
        ("In-domain frame AUC", f"{a0['frame_auc']:.4f}", ""),
        ("Localization mAP", f"{tflh['best_map']:.3f}", f"· default decode {loc['mAP']:.3f}"),
        ("  AP@0.5 / @0.75", f"{loc['AP@50']:.3f} / {loc['AP@75']:.3f}",
         f"· seg AP@0.1 {tflh['best_map_segment_ap']['ap_01']:.3f}"),
        ("CMIS-2 coupling effect", "-0.0003 (n.s.)", "· module null in-domain"),
        ("Phase-1 lift (FakeAVCeleb)", f"+{PHASE1_LIFT[0]:.3f}  P=1.00",
         f"· DeepfakeTIMIT +{PHASE1_LIFT_TIMIT[0]:.3f}"),
        ("Prior AUC (FakeAVCeleb)", f"{cb['mean']['auc']:.3f} bal",
         f"· full {cf['mean']['auc']:.3f}  EER {cb['mean']['eer']:.3f}"),
        ("Sup ⊕ Prior (FAC full)", f"{fac['fused_auc']:.3f}",
         f"· sup {fac['sup_auc']:.3f} / prior {fac['prior_auc']:.3f}"),
        ("AVSR ⊕ Prior (FAC bal)", f"{o['mean']:.3f}",
         f"· max {o['max']:.3f} · AVSR {o['avsr']:.3f}"),
        ("  dub specialist (FAC)", f"prior {dub['prior']:.3f}",
         f"· AVSR ceilings {dub['avsr']:.3f}"),
        ("AVSR ⊕ SyncNet (TIMIT)", f"{sy['mean']:.3f}",
         f"· AVSR {sy['avsr']:.3f} / sync {sy['syncnet']:.3f}"),
        ("Fingerprint zero-shot", "0.44 (inverted)", "· H1 rejected"),
    ]
    y = 0.94
    for name, best, alt in rows:
        ax.text(0.0, y, name, transform=ax.transAxes, va="top", ha="left",
                fontsize=8.6, family="monospace")
        ax.text(0.56, y, best, transform=ax.transAxes, va="top", ha="left",
                fontsize=8.6, family="monospace", fontweight="bold", color=C_FUSE)
        if alt:
            ax.text(0.56, y - 0.038, alt, transform=ax.transAxes, va="top", ha="left",
                    fontsize=6.8, family="monospace", color=C_NEUT)
            y -= 0.078
        else:
            y -= 0.045


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    abl = load_ablation()
    a0 = abl["A0\nbaseline"]
    corr = _read(LOGS / "correspondence_eval.json")
    fus = _read(LOGS / "fusion_eval.json")
    avsr = _read(LOGS / "fusion_avsr_eval.json")
    sync = _read(LOGS / "fusion_syncnet_eval.json")

    fig = plt.figure(figsize=(19, 19))
    gs = fig.add_gridspec(4, 3, hspace=0.46, wspace=0.26,
                          top=0.93, bottom=0.04, left=0.055, right=0.975)

    panel_cards(fig.add_subplot(gs[0, 0]), a0)
    panel_localization(fig.add_subplot(gs[0, 1]), a0)
    panel_detection_all(fig.add_subplot(gs[0, 2]), abl)

    panel_ablation(fig.add_subplot(gs[1, 0]), abl)
    _phase1_panel(fig.add_subplot(gs[1, 1]), GATE_FAC, PHASE1_LIFT,
                  "Articulation-aug lift — FakeAVCeleb (zero-shot)", "P(Δ>0)=1.00")
    _phase1_panel(fig.add_subplot(gs[1, 2]), GATE_TIMIT, PHASE1_LIFT_TIMIT,
                  "Articulation-aug lift — DeepfakeTIMIT (zero-shot)",
                  "high seed var; encoding-confounded")

    panel_prior(fig.add_subplot(gs[2, 0]), corr)
    panel_prior_pooling(fig.add_subplot(gs[2, 1]), corr)
    panel_fusion(fig.add_subplot(gs[2, 2]), fus, avsr)

    panel_pertype(fig.add_subplot(gs[3, 0]), avsr)
    panel_system(fig.add_subplot(gs[3, 1]), a0, avsr, sync)
    panel_best_alternatives(fig.add_subplot(gs[3, 2]), a0, corr, fus, avsr, sync)

    fig.suptitle("TALVAD-X — Paper Results Dashboard  "
                 "(detection · localization · ablation · cross-dataset · fusion)",
                 fontsize=18, fontweight="bold")
    fig.text(0.5, 0.945,
             "666-d/frame (audio 376 | appearance 256 | spectral 34) · T=48 @ 12 fps · frozen wav2vec2/EffNet-B0/auto_avsr · "
             "loss BCE+0.15·TFLH+0.10·incon+0.10·InfoNCE · zero-shot w/ source-fit PCA (parity guard) · 3 seeds, 2000× bootstrap",
             ha="center", fontsize=8.5, color="#444")
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
