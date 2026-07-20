"""
scripts/eval_fusion_avsr.py — THE headline cross-dataset fusion (TALVAD-X v2, Route B).

Fuses the two COMPLEMENTARY zero-shot detectors ON THE SAME CLIPS:
  (A) AVSR head            — logistic-regression on frozen auto_avsr (AV-HuBERT-class)
                             pooled features, trained on LAV-DF, applied zero-shot.
                             Strong on VISUAL fakes (~0.93), BLIND to audio dubs (~0.58).
  (B) correspondence prior — self-supervised audio<->lip-motion prior trained on REAL
                             clips ONLY (models/correspondence.py). The dub specialist:
                             flags audio-lip MISMATCH, silent on in-sync LAV-DF fakes.

Both are scored on the SAME balanced FakeAVCeleb set (which carries per-fake-type
labels), aligned by video stem. We report each detector alone and several label-free
fusions, OVERALL and PER FAKE TYPE. The RealVid+FakeAud(dub) row is the headline:
the AVSR head ceilings at ~0.58 there, and the prior is meant to lift it.

Why this and not eval_fusion.py: that script fuses the OLD supervised detector (~0.60)
with the prior. This one fuses the STRONG AVSR head (0.81) with the prior — the
combination that should clear the ~0.87 SOTA band.

Prereqs (run first):
  # AVSR pooled feats for train + target (already extracted):
  #   data/talvad_x_crossdataset/{lavdf_train,fakeavceleb_bal}/avsr/features.npz
  # 666-d pack for the SAME bal clips:
  python scripts/extract_talvad_x.py --config configs/lavdf_talvad_x.yaml \
      --target_csv data/fakeavceleb_balanced_metadata.csv --target_name fakeavceleb_bal \
      --root_dir datasets/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2
  # the prior:
  python scripts/train_correspondence.py --config configs/lavdf_talvad_x.yaml --epochs 20

Run (MAIN env — needs sklearn + torch):
  python scripts/eval_fusion_avsr.py --config configs/lavdf_talvad_x.yaml
  python scripts/eval_fusion_avsr.py --config configs/lavdf_talvad_x.yaml \
      --train_target lavdf_train --test_target fakeavceleb_bal --prior_dir checkpoints/correspondence
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from models.correspondence import CrossModalCorrespondence  # noqa: E402

XBASE = REPO / "data" / "talvad_x_crossdataset"


def _rank_norm(s: np.ndarray) -> np.ndarray:
    """Map scores to [0,1] percentile ranks (robust fusion across heterogeneous sources)."""
    return s.argsort().argsort().astype(np.float64) / max(len(s) - 1, 1)


def _avsr_scores(train_target: str, test_target: str):
    """Train the AVSR head on `train_target` (leakage-free scaler), score `test_target`.
    Returns dict stem -> {score, label, mv, ma}. Higher score => more fake."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    tr = np.load(XBASE / train_target / "avsr" / "features.npz")
    te = np.load(XBASE / test_target / "avsr" / "features.npz")
    Xtr, ytr = tr["pooled"], tr["label"].astype(int)
    Xte, yte = te["pooled"], te["label"].astype(int)
    ids = [str(i) for i in te["id"]]
    mv, ma = te["modify_video"].astype(int), te["modify_audio"].astype(int)

    scaler = StandardScaler().fit(Xtr)              # fit on TRAIN only
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(scaler.transform(Xtr), ytr)
    proba = clf.predict_proba(scaler.transform(Xte))[:, 1]
    # Ordered arrays in CSV/file order (NOT a stem-keyed dict — FakeAVCeleb reuses
    # filenames across identity folders, so stems collide and a dict would collapse them).
    return {"ids": ids, "score": proba, "label": yte, "mv": mv, "ma": ma}


def _prior_scores(test_target: str, prior_dir: Path, device: str):
    """Run the correspondence prior on the target's 666-d pack.
    Returns dict stem -> anomaly (higher => more fake). Stem recovered from the
    `{stem}_{md5}` packed id via rsplit (md5 is the only trailing underscore-segment)."""
    pack = XBASE / test_target / "packed"
    if not (pack / "features.npy").exists():
        raise FileNotFoundError(
            f"No 666-d pack at {pack}. Extract it first:\n"
            f"  python scripts/extract_talvad_x.py --config <cfg> "
            f"--target_csv data/fakeavceleb_balanced_metadata.csv "
            f"--target_name {test_target} --root_dir <FakeAVCeleb root>")
    feats = np.load(pack / "features.npy").astype(np.float32)
    ids = json.loads((pack / "ids.json").read_text(encoding="utf-8"))
    man = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    ad, app = int(man["audio_dim"]), int(man["appearance_dim"])

    blob = torch.load(prior_dir / "prior.pt", map_location=device)
    pc = blob["cfg"]
    model = CrossModalCorrespondence(audio_dim=pc["audio_dim"], visual_dim=pc["app_dim"],
                                     emb_dim=pc["emb_dim"], motion=pc["motion"],
                                     recon_weight=pc["recon_weight"]).to(device)
    model.load_state_dict(blob["model"]); model.eval()
    from joblib import load as jload
    scaler = jload(prior_dir / "scaler.joblib")
    feats = scaler.transform(feats.reshape(-1, feats.shape[-1])).reshape(feats.shape).astype(np.float32)

    out = []
    with torch.no_grad():
        for i in range(0, len(feats), 256):
            x = torch.from_numpy(feats[i:i + 256]).to(device)
            out.extend(model.anomaly(x[..., :ad], x[..., ad:ad + app]).cpu().numpy().tolist())
    # Recover the stem from the `{stem}_{md5}` packed id; keep FILE ORDER for positional align.
    stems = [str(i).rsplit("_", 1)[0] for i in ids]
    return {"stems": stems, "anomaly": np.asarray(out, dtype=float)}


def main():
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)  # accepted for parity; not strictly needed
    ap.add_argument("--train_target", default="lavdf_train")
    ap.add_argument("--test_target", default="fakeavceleb_bal")
    ap.add_argument("--prior_dir", default="checkpoints/correspondence")
    ap.add_argument("--w", type=float, default=0.5,
                    help="weight on AVSR in the weighted fusion (1-w on prior)")
    ap.add_argument("--dump_predictions", default=None,
                    help="if set, write per-clip CSV (id,label,avsr,prior,fused) for figures")
    args = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prior_dir = REPO / args.prior_dir

    A = _avsr_scores(args.train_target, args.test_target)
    P = _prior_scores(args.test_target, prior_dir, device)
    ida, idp = A["ids"], P["stems"]

    # Preferred: POSITIONAL alignment. Both feature sets are extracted from the SAME CSV
    # in row order, so avsr[i] and packed[i] are the same physical clip — collision-proof
    # and lossless (no stem-dict collapse). Verify stems agree at every index first.
    if len(ida) == len(idp) and all(ida[i] == idp[i] for i in range(len(ida))):
        a, p = A["score"], P["anomaly"]
        y, mv, ma = A["label"], A["mv"], A["ma"]
        aligned_ids = list(ida)
        miss_a = miss_p = 0
        align_mode = f"positional — {len(a)} clips, all stems matched in order"
    else:
        # Fallback: unique-stem intersection (LOSSY if stems collide).
        amap = {ida[i]: i for i in range(len(ida))}
        pmap = {idp[i]: i for i in range(len(idp))}
        keys = [s for s in amap if s in pmap]
        if not keys:
            raise RuntimeError("No overlap between AVSR and prior — alignment failed.")
        ai = [amap[s] for s in keys]; pi = [pmap[s] for s in keys]
        a, p = A["score"][ai], P["anomaly"][pi]
        y, mv, ma = A["label"][ai], A["mv"][ai], A["ma"][ai]
        aligned_ids = list(keys)
        miss_a, miss_p = len(ida) - len(keys), len(idp) - len(keys)
        align_mode = (f"stem-intersection — {len(keys)} unique; WARNING collisions "
                      f"collapsed avsr {len(ida)}->{len(set(ida))} (positional align unavailable)")
    n_aligned = len(a)

    ar, pr = _rank_norm(a), _rank_norm(p)

    if args.dump_predictions:
        import pandas as pd
        fused = (ar + pr) / 2.0  # rank-mean fusion in [0,1] (the reported headline)
        outp = Path(args.dump_predictions)
        outp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": aligned_ids, "label": y.astype(int),
                      "avsr": np.round(a, 6), "prior": np.round(p, 6),
                      "score": np.round(fused, 6)}).to_csv(outp, index=False)
        print(f"  [dump] per-clip predictions -> {outp}")

    fusions = {
        "avsr":     a,
        "prior":    p,
        "mean":     ar + pr,                       # equal-weight (the blunt baseline)
        "max":      np.maximum(ar, pr),            # noisy-OR: fake if EITHER fires
        f"w{args.w:g}": args.w * ar + (1 - args.w) * pr,
    }

    def auc_on(mask):
        yy = y[mask]
        if len(np.unique(yy)) < 2:
            return {k: float("nan") for k in fusions}
        return {k: roc_auc_score(yy, v[mask]) for k, v in fusions.items()}

    print(f"\n{'='*78}\n  FUSION  AVSR head ({args.train_target}->{args.test_target}) + "
          f"correspondence prior ({prior_dir.name})\n"
          f"  align: {align_mode}  (avsr-only dropped={miss_a}, prior-only dropped={miss_p})\n{'='*78}")
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
        nf = int(((y[mask] == 1)).sum())
        report[label] = {k: round(float(v), 4) for k, v in aucs.items()} | {"n_fake": nf}
        cells = " ".join(f"{aucs[k]:8.4f}" for k in fusions)
        tag = " <- dub (AVSR ceiling ~0.58)" if label.startswith("RealVid") else ""
        print(f"  {label:24s}  {cells}   n_fake={nf}{tag}")

    best = max(fusions, key=lambda k: report["OVERALL"][k])
    print(f"{'-'*78}\n  best OVERALL fusion: {best} = {report['OVERALL'][best]:.4f}   "
          f"(avsr={report['OVERALL']['avsr']:.4f}, prior={report['OVERALL']['prior']:.4f})")
    print(f"  >>> complementary IF a fusion column beats BOTH avsr and prior alone\n{'='*78}")

    out = REPO / "logs" / "fusion_avsr_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"train": args.train_target, "test": args.test_target,
                               "prior": prior_dir.name, "n_aligned": n_aligned,
                               "align_mode": align_mode, "results": report}, indent=2), encoding="utf-8")
    print(f"  -> {out}\n")


if __name__ == "__main__":
    main()
