"""
scripts/eval_fusion_syncnet.py — AVSR head ⊕ SyncNet lip-sync fusion (TALVAD-X v2).

The DeepfakeTIMIT counterpart of eval_fusion_avsr.py. There the dub-catcher is the
correspondence prior; here it is frozen SyncNet, which is the natural specialist for
DeepfakeTIMIT (face-swap with REAL audio => broken lip-sync, SyncNet's sweet spot).

Fuses two COMPLEMENTARY zero-shot detectors on the SAME clips:
  (A) AVSR head    — logistic-regression on frozen auto_avsr pooled features, trained
                     on LAV-DF, applied zero-shot. Catches visual manipulation.
  (B) SyncNet      — frozen lip-sync model, zero training. Score = mean aligned-window
                     audio/lip distance (higher => worse sync => more likely fake).

Both feature sets are extracted from the SAME CSV in row order, so they align
positionally (verified at runtime). Reports each alone and rank-normalized fusions.

Run (MAIN env):
  python scripts/eval_fusion_syncnet.py --train_target lavdf_train --test_target deepfaketimit
  python scripts/eval_fusion_syncnet.py --test_target deepfaketimit --syncnet_score mean_dist --w 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from eval_fusion_avsr import _avsr_scores, _rank_norm, XBASE  # noqa: E402

# SyncNet fake-score candidates (higher => more fake). neg_conf/min/mean per extract_syncnet.
_SN_SCORES = {
    "mean_dist": lambda d: d["mean_dist"].astype(float),
    "min_dist":  lambda d: d["min_dist"].astype(float),
    "neg_conf":  lambda d: -d["sync_conf"].astype(float),
}


def _syncnet_scores(test_target: str, which: str):
    npz = np.load(XBASE / test_target / "syncnet" / "features.npz", allow_pickle=True)
    return {"ids": [str(i) for i in npz["id"]], "score": _SN_SCORES[which](npz),
            "label": npz["label"].astype(int),
            "mv": npz["modify_video"].astype(int), "ma": npz["modify_audio"].astype(int)}


def main():
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_target", default="lavdf_train")
    ap.add_argument("--test_target", default="deepfaketimit")
    ap.add_argument("--syncnet_score", default="mean_dist", choices=list(_SN_SCORES))
    ap.add_argument("--w", type=float, default=0.5, help="weight on AVSR in weighted fusion")
    ap.add_argument("--dump_predictions", default=None,
                    help="if set, write per-clip CSV (id,label,avsr,syncnet,fused) for figures")
    args = ap.parse_args()
    from sklearn.metrics import roc_auc_score

    A = _avsr_scores(args.train_target, args.test_target)
    S = _syncnet_scores(args.test_target, args.syncnet_score)
    ida, ids = A["ids"], S["ids"]

    if len(ida) == len(ids) and all(ida[i] == ids[i] for i in range(len(ida))):
        a, s = A["score"], S["score"]
        y, mv, ma = A["label"], A["mv"], A["ma"]
        aligned_ids = list(ida)
        align_mode = f"positional — {len(a)} clips, all ids matched in order"
    else:  # fallback: id intersection
        amap = {ida[i]: i for i in range(len(ida))}
        smap = {ids[i]: i for i in range(len(ids))}
        keys = [k for k in amap if k in smap]
        if not keys:
            raise RuntimeError("No overlap between AVSR and SyncNet ids — alignment failed.")
        ai = [amap[k] for k in keys]; si = [smap[k] for k in keys]
        a, s = A["score"][ai], S["score"][si]
        y, mv, ma = A["label"][ai], A["mv"][ai], A["ma"][ai]
        aligned_ids = list(keys)
        align_mode = f"id-intersection — {len(keys)} clips"
    n_aligned = len(a)

    ar, sr = _rank_norm(a), _rank_norm(s)

    if args.dump_predictions:
        import pandas as pd
        fused = (ar + sr) / 2.0  # rank-mean fusion in [0,1] (the reported headline)
        outp = Path(args.dump_predictions)
        outp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": aligned_ids, "label": y.astype(int),
                      "avsr": np.round(a, 6), "syncnet": np.round(s, 6),
                      "score": np.round(fused, 6)}).to_csv(outp, index=False)
        print(f"  [dump] per-clip predictions -> {outp}")
    fusions = {
        "avsr":         a,
        "syncnet":      s,
        "mean":         ar + sr,
        "max":          np.maximum(ar, sr),
        f"w{args.w:g}": args.w * ar + (1 - args.w) * sr,
    }

    def auc_on(mask):
        yy = y[mask]
        if len(np.unique(yy)) < 2:
            return {k: float("nan") for k in fusions}
        return {k: roc_auc_score(yy, v[mask]) for k, v in fusions.items()}

    print(f"\n{'='*78}\n  FUSION  AVSR head ({args.train_target}->{args.test_target}) + "
          f"SyncNet ({args.syncnet_score})\n  align: {align_mode}\n{'='*78}")
    hdr = "  " + " ".join(f"{k:>8s}" for k in fusions)
    print(f"  {'split':24s}{hdr}")

    real = y == 0
    rows = [("OVERALL", np.ones(len(y), bool))]
    names = {(1, 1): "FakeVid+FakeAud", (1, 0): "FakeVid+RealAud(swap)", (0, 1): "RealVid+FakeAud(dub)"}
    for (cvv, caa), label in names.items():
        ftype = (y == 1) & (mv == cvv) & (ma == caa)
        if ftype.sum():
            rows.append((label, real | ftype))

    report = {}
    for label, mask in rows:
        aucs = auc_on(mask)
        nf = int((y[mask] == 1).sum())
        report[label] = {k: round(float(v), 4) for k, v in aucs.items()} | {"n_fake": nf}
        cells = " ".join(f"{aucs[k]:8.4f}" for k in fusions)
        print(f"  {label:24s}  {cells}   n_fake={nf}")

    best = max(fusions, key=lambda k: report["OVERALL"][k])
    print(f"{'-'*78}\n  best OVERALL fusion: {best} = {report['OVERALL'][best]:.4f}   "
          f"(avsr={report['OVERALL']['avsr']:.4f}, syncnet={report['OVERALL']['syncnet']:.4f})")
    print(f"  >>> complementary IF a fusion column beats BOTH avsr and syncnet alone\n{'='*78}")

    out = REPO / "logs" / "fusion_syncnet_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"train": args.train_target, "test": args.test_target,
                               "syncnet_score": args.syncnet_score, "n_aligned": n_aligned,
                               "align_mode": align_mode, "results": report}, indent=2), encoding="utf-8")
    print(f"  -> {out}\n")


if __name__ == "__main__":
    main()
