"""
scripts/eval_syncnet.py — score the frozen SyncNet features (zero training) and report
cross-dataset AUC, plus a per-fake-type breakdown.

Fake-score candidates (higher => more likely fake):
  neg_conf  = -sync_conf   (low lip-sync confidence => fake)
  min_dist  =  min_dist    (poor best-aligned audio/lip match => fake)
  mean_dist =  mean_dist

The per-type table is the honest test of the known caveat: SyncNet should nail
audio-dubbed / face-swap-with-real-audio fakes (broken sync) and may MISS wav2lip-style
fakes (sync is optimised). modify_video/modify_audio identify the type.

  python scripts/eval_syncnet.py --target fakeavceleb_bal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def _eer(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    args = ap.parse_args()

    npz = REPO / "data" / "talvad_x_crossdataset" / args.target / "syncnet" / "features.npz"
    d = np.load(npz)
    y = d["label"].astype(int)
    scores = {"neg_conf": -d["sync_conf"], "min_dist": d["min_dist"], "mean_dist": d["mean_dist"]}

    print(f"\n{'='*64}\n  SyncNet zero-shot scoring  ({args.target})"
          f"   n={len(y)} real={int((y==0).sum())} fake={int((y==1).sum())}\n{'='*64}")
    best_name, best_auc = None, -1
    for name, s in scores.items():
        auc, eer = _auc(y, s), _eer(y, s)
        print(f"  {name:10s} AUC={auc:.4f}  EER={eer:.4f}")
        if auc > best_auc:
            best_auc, best_name, best_s = auc, name, s
    print(f"  -> best raw score: {best_name}  AUC={best_auc:.4f}")

    # Per-fake-type breakdown (reals vs each fake subtype) using the best score.
    print(f"\n  per-fake-type AUC (reals vs each type, score={best_name}):")
    mv, ma = d["modify_video"].astype(int), d["modify_audio"].astype(int)
    real = y == 0
    names = {(1, 1): "FakeVid+FakeAud", (1, 0): "FakeVid+RealAud(swap)",
             (0, 1): "RealVid+FakeAud(dub)"}
    for (cv, ca), label in names.items():
        sub = real | ((y == 1) & (mv == cv) & (ma == ca))
        nft = int(((y == 1) & (mv == cv) & (ma == ca)).sum())
        if nft == 0:
            continue
        auc = _auc(y[sub], best_s[sub])
        print(f"    {label:24s} n_fake={nft:4d}  AUC={auc:.4f}")

    print(f"\n  baseline to beat: correspondence-on-old-features = 0.68 (FakeAVCeleb)\n{'='*64}")


if __name__ == "__main__":
    main()
