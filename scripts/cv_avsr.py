"""
scripts/cv_avsr.py — quick sanity: do the frozen auto_avsr (AV-HuBERT-class) pooled
features separate real vs fake AT ALL? 5-fold stratified CV with logistic regression on
the 1536-d pooled features.

This is IN-DOMAIN separability (train+test both FakeAVCeleb) — NOT the cross-dataset claim.
It tells us whether the features carry deepfake signal before we invest in the LAV-DF train
extraction (supervised cross-dataset) or the audio-visual correspondence head.

Runs in the MAIN env (needs sklearn):
  python scripts/cv_avsr.py --target fakeavceleb_bal
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="fakeavceleb_bal")
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    d = np.load(REPO / "data" / "talvad_x_crossdataset" / args.target / "avsr" / "features.npz")
    X, y = d["pooled"], d["label"].astype(int)
    mv, ma = d["modify_video"].astype(int), d["modify_audio"].astype(int)
    print(f"\n  AVSR pooled features: {X.shape}  real={int((y==0).sum())} fake={int((y==1).sum())}")

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)
    print(f"  5-fold CV AUC (in-domain separability): {auc:.4f}")

    real = y == 0
    names = {(1, 1): "FakeVid+FakeAud", (1, 0): "FakeVid+RealAud(swap)", (0, 1): "RealVid+FakeAud(dub)"}
    print("  per-fake-type (reals vs each type, same CV scores):")
    for (cv_, ca), label in names.items():
        m = real | ((y == 1) & (mv == cv_) & (ma == ca))
        nft = int(((y == 1) & (mv == cv_) & (ma == ca)).sum())
        if nft:
            print(f"    {label:24s} n_fake={nft:4d}  AUC={roc_auc_score(y[m], proba[m]):.4f}")
    print("  (in-domain only; cross-dataset is the next experiment)\n")


if __name__ == "__main__":
    main()
