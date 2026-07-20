"""
scripts/eval_correspondence.py — zero-shot eval of the Phase-2 correspondence PRIOR.

Scores each target clip by its per-clip correspondence DEVIATION (higher -> more
likely fake) and reports AUC/EER. Zero-shot: the prior's saved LAV-DF-real scaler is
applied as-is; nothing is fit on the target.

  python scripts/eval_correspondence.py --config configs/lavdf_talvad_x.yaml --targets fakeavceleb deepfaketimit lavdf_selfcheck
  python scripts/eval_correspondence.py --config configs/lavdf_talvad_x.yaml --targets fakeavceleb --prior_dir checkpoints/correspondence_m1
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
from utils.config import load_config  # noqa: E402


def _eer(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def _load_pack(target: str):
    if target == "lavdf_selfcheck":
        pack = REPO / "data" / "talvad_x_crossdataset" / "lavdf_selfcheck" / "packed"
    else:
        pack = REPO / "data" / "talvad_x_crossdataset" / target / "packed"
    feats = np.load(pack / "features.npy").astype(np.float32)
    labels = np.load(pack / "labels.npy").astype(np.int64)
    manifest = json.loads((pack / "manifest.json").read_text())
    return feats, labels, int(manifest["audio_dim"]), int(manifest["appearance_dim"])


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--targets", nargs="*", default=["fakeavceleb", "deepfaketimit", "lavdf_selfcheck"])
    ap.add_argument("--prior_dir", default="checkpoints/correspondence")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prior_dir = REPO / args.prior_dir
    blob = torch.load(prior_dir / "prior.pt", map_location=device)
    pc = blob["cfg"]
    model = CrossModalCorrespondence(audio_dim=pc["audio_dim"], visual_dim=pc["app_dim"],
                                     emb_dim=pc["emb_dim"], motion=pc["motion"],
                                     recon_weight=pc["recon_weight"]).to(device)
    model.load_state_dict(blob["model"]); model.eval()
    from joblib import load as jload
    scaler = jload(prior_dir / "scaler.joblib")
    from sklearn.metrics import roc_auc_score

    print(f"\n{'='*72}\n  Correspondence PRIOR zero-shot eval  ({prior_dir.name})\n"
          f"  per-clip reductions of the per-frame deviation: mean | top20% | max\n{'='*72}")
    rows = {}
    for tname in args.targets:
        feats, labels, audio_dim, app_dim = _load_pack(tname)
        feats = scaler.transform(feats.reshape(-1, feats.shape[-1])).reshape(feats.shape).astype(np.float32)
        dev = []
        with torch.no_grad():
            for i in range(0, len(feats), 256):
                x = torch.from_numpy(feats[i:i + 256]).to(device)
                a = x[..., :audio_dim]; v = x[..., audio_dim:audio_dim + app_dim]
                dev.append(model.deviation(a, v).cpu().numpy())
        dev = np.concatenate(dev, axis=0)          # [N, T]  per-frame deviation
        y = labels.astype(int)
        T = dev.shape[1]; k = max(1, int(round(0.2 * T)))
        reductions = {"mean": dev.mean(1),
                      "topk20": np.sort(dev, 1)[:, -k:].mean(1),
                      "max": dev.max(1)}
        rows[tname] = {"n": int(len(y)), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())}
        line = f"  {tname:16s}"
        for rname, s in reductions.items():
            if len(np.unique(y)) > 1:
                auc = roc_auc_score(y, s); eer = _eer(y, s)
            else:
                auc = float("nan"); eer = float("nan")
            rows[tname][rname] = {"auc": round(float(auc), 4), "eer": round(float(eer), 4)}
            line += f"  {rname}={auc:.4f}"
        print(line + f"  (real={int((y==0).sum())}, fake={int((y==1).sum())})")

    out = REPO / "logs" / "correspondence_eval.json"
    out.write_text(json.dumps({"prior": prior_dir.name, "results": rows}, indent=2), encoding="utf-8")
    print(f"\n  -> {out}\n{'='*60}")


if __name__ == "__main__":
    main()
