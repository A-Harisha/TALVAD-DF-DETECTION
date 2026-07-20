"""
scripts/train_test_avsr.py — CROSS-DATASET train+test on frozen auto_avsr (AV-HuBERT-class)
pooled features. Train a head on one dataset, zero-shot test on another (leakage-free:
scaler fit on train only, nothing fit on test).

  python scripts/train_test_avsr.py --train_target lavdf_train --test_target fakeavceleb_bal

Reports cross-dataset AUC (the number that matters, vs the 0.68 old-features baseline),
a per-fake-type breakdown on the test set, and the in-domain CV AUC on the train set for
reference. Runs in the MAIN env (sklearn).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _load(target):
    d = np.load(REPO / "data" / "talvad_x_crossdataset" / target / "avsr" / "features.npz")
    return d["pooled"], d["label"].astype(int), d["modify_video"].astype(int), d["modify_audio"].astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_target", default="lavdf_train")
    ap.add_argument("--test_target", default="fakeavceleb_bal")
    ap.add_argument("--clf", choices=["logreg", "mlp"], default="logreg")
    ap.add_argument("--instance_norm", action="store_true",
                    help="per-sample standardize pooled vec (label-free domain-shift fix)")
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    Xtr, ytr, _, _ = _load(args.train_target)
    Xte, yte, mv, ma = _load(args.test_target)

    if args.instance_norm:  # per-row standardization
        Xtr = (Xtr - Xtr.mean(1, keepdims=True)) / (Xtr.std(1, keepdims=True) + 1e-6)
        Xte = (Xte - Xte.mean(1, keepdims=True)) / (Xte.std(1, keepdims=True) + 1e-6)

    scaler = StandardScaler().fit(Xtr)             # fit on TRAIN only (leakage-free)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    clf = (LogisticRegression(max_iter=3000, C=1.0) if args.clf == "logreg"
           else MLPClassifier(hidden_layer_sizes=(256,), max_iter=600, early_stopping=True, random_state=42))

    print(f"\n{'='*66}\n  CROSS-DATASET  train={args.train_target}({len(ytr)})  ->  "
          f"test={args.test_target}({len(yte)})   clf={args.clf} inorm={args.instance_norm}\n{'='*66}")

    # in-domain CV on train (reference)
    cv_auc = roc_auc_score(ytr, cross_val_predict(
        clf, Xtr_s, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        method="predict_proba")[:, 1])

    clf.fit(Xtr_s, ytr)
    proba = clf.predict_proba(Xte_s)[:, 1]
    auc = roc_auc_score(yte, proba)
    print(f"  in-domain CV AUC (train domain) : {cv_auc:.4f}")
    print(f"  >>> CROSS-DATASET AUC (test)    : {auc:.4f}   (baseline to beat = 0.68)")

    real = yte == 0
    names = {(1, 1): "FakeVid+FakeAud", (1, 0): "FakeVid+RealAud(swap)", (0, 1): "RealVid+FakeAud(dub)"}
    print("  per-fake-type (reals vs each type):")
    for (cv_, ca), label in names.items():
        m = real | ((yte == 1) & (mv == cv_) & (ma == ca))
        nft = int(((yte == 1) & (mv == cv_) & (ma == ca)).sum())
        if nft:
            print(f"    {label:24s} n_fake={nft:4d}  AUC={roc_auc_score(yte[m], proba[m]):.4f}")
    print('='*66 + "\n")


if __name__ == "__main__":
    main()
