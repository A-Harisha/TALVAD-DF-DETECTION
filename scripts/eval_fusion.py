"""
scripts/eval_fusion.py — fuse complementary cross-dataset detectors (TALVAD-X v2).

Combines (a) a supervised detector (the Phase-1 spectral-aug variant, ensembled over
seeds) with (b) the Phase-2 self-supervised correspondence prior's anomaly score.
Each source is rank-normalized per target (robust to scale/shape), then averaged.
Reports AUC for each source alone and the fused score — the question being whether
the two signals are complementary (fused > best single).

Zero-shot throughout: supervised uses its saved LAV-DF scaler; the prior uses its
saved LAV-DF-real scaler; nothing is fit on the target.

  python scripts/eval_fusion.py --config configs/lavdf_talvad_x.yaml \
      --supervised P1_a0_aug --seeds 42 1337 2024 \
      --prior_dir checkpoints/correspondence --targets fakeavceleb deepfaketimit lavdf_selfcheck
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
sys.path.insert(0, str(REPO / "scripts"))

from models.talvad_x import TALVADXDetector  # noqa: E402
from models.correspondence import CrossModalCorrespondence  # noqa: E402
from utils.config import load_config  # noqa: E402
from ablate_talvad_x import VARIANTS  # noqa: E402


def _apply(cfg, sets):
    for kv in sets or []:
        k, val = kv.split("=", 1)
        v = val
        if val.lower() in ("true", "false"):
            v = val.lower() == "true"
        else:
            try:
                v = int(val) if val.isdigit() else float(val)
            except ValueError:
                v = val
        node = cfg
        for p in k.split(".")[:-1]:
            node = node.setdefault(p, {})
        node[k.split(".")[-1]] = v


def _rank_norm(s: np.ndarray) -> np.ndarray:
    """Map scores to [0,1] percentile ranks (robust fusion across heterogeneous sources)."""
    order = s.argsort().argsort().astype(np.float64)
    return order / max(len(s) - 1, 1)


def _load_pack(target: str):
    pack = REPO / "data" / "talvad_x_crossdataset" / target / "packed"
    feats = np.load(pack / "features.npy").astype(np.float32)
    labels = np.load(pack / "labels.npy").astype(np.int64)
    manifest = json.loads((pack / "manifest.json").read_text())
    return feats, labels, int(manifest["audio_dim"]), int(manifest["appearance_dim"])


def _supervised_scores(cfg_base, variant, seeds, feats_raw, audio_dim, device):
    """Mean sigmoid prob across seeds for a supervised variant (None if no ckpts)."""
    probs = []
    for s in seeds:
        ck = REPO / "checkpoints" / f"talvad_x_{variant}_s{s}"
        if not (ck / "best_model.pt").exists():
            continue
        cfg = json.loads(json.dumps(cfg_base))
        _apply(cfg, VARIANTS[variant])
        from joblib import load as jload
        feats = feats_raw
        if (ck / "scaler.joblib").exists():
            sc = jload(ck / "scaler.joblib")
            feats = sc.transform(feats_raw.reshape(-1, feats_raw.shape[-1])).reshape(feats_raw.shape).astype(np.float32)
        model = TALVADXDetector(cfg["model"]).to(device)
        blob = torch.load(ck / "best_model.pt", map_location=device)
        model.load_state_dict(blob.get("model", blob) if isinstance(blob, dict) else blob)
        model.eval()
        p = []
        with torch.no_grad():
            for i in range(0, len(feats), 64):
                x = torch.from_numpy(feats[i:i + 64]).to(device)
                lo, _, _ = model(x[..., :audio_dim], x[..., audio_dim:])
                p.extend(torch.sigmoid(lo).squeeze(-1).cpu().numpy().tolist())
        probs.append(np.asarray(p))
    if not probs:
        return None
    return np.mean(probs, axis=0)


def _prior_scores(prior_dir, feats_raw, device):
    blob = torch.load(prior_dir / "prior.pt", map_location=device)
    pc = blob["cfg"]
    model = CrossModalCorrespondence(audio_dim=pc["audio_dim"], visual_dim=pc["app_dim"],
                                     emb_dim=pc["emb_dim"], motion=pc["motion"],
                                     recon_weight=pc["recon_weight"]).to(device)
    model.load_state_dict(blob["model"]); model.eval()
    from joblib import load as jload
    sc = jload(prior_dir / "scaler.joblib")
    feats = sc.transform(feats_raw.reshape(-1, feats_raw.shape[-1])).reshape(feats_raw.shape).astype(np.float32)
    ad, app = pc["audio_dim"], pc["app_dim"]
    out = []
    with torch.no_grad():
        for i in range(0, len(feats), 256):
            x = torch.from_numpy(feats[i:i + 256]).to(device)
            out.extend(model.anomaly(x[..., :ad], x[..., ad:ad + app]).cpu().numpy().tolist())
    return np.asarray(out)


def main():
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--supervised", default="P1_a0_aug")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1337, 2024])
    ap.add_argument("--prior_dir", default="checkpoints/correspondence")
    ap.add_argument("--targets", nargs="*", default=["fakeavceleb", "deepfaketimit", "lavdf_selfcheck"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prior_dir = REPO / args.prior_dir
    from sklearn.metrics import roc_auc_score

    print(f"\n{'='*66}\n  FUSION  supervised={args.supervised}(x{len(args.seeds)} seeds) + prior={prior_dir.name}\n{'='*66}")
    report = {}
    for t in args.targets:
        feats, labels, ad, app = _load_pack(t)
        y = labels.astype(int)
        sup = _supervised_scores(cfg, args.supervised, args.seeds, feats, ad, device)
        cor = _prior_scores(prior_dir, feats, device) if (prior_dir / "prior.pt").exists() else None
        row = {"n": int(len(y)), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())}
        if len(np.unique(y)) < 2:
            print(f"  {t}: single-class, skipping AUC"); report[t] = row; continue
        if sup is not None:
            row["sup_auc"] = round(float(roc_auc_score(y, sup)), 4)
        if cor is not None:
            row["prior_auc"] = round(float(roc_auc_score(y, cor)), 4)
        if sup is not None and cor is not None:
            fused = _rank_norm(sup) + _rank_norm(cor)
            row["fused_auc"] = round(float(roc_auc_score(y, fused)), 4)
        report[t] = row
        print(f"  {t:16s} sup={row.get('sup_auc','-')}  prior={row.get('prior_auc','-')}  "
              f"FUSED={row.get('fused_auc','-')}  (n={row['n']}, fake={row['n_fake']})")

    out = REPO / "logs" / "fusion_eval.json"
    out.write_text(json.dumps({"supervised": args.supervised, "prior": prior_dir.name,
                               "seeds": args.seeds, "results": report}, indent=2), encoding="utf-8")
    print(f"\n  -> {out}\n{'='*66}")


if __name__ == "__main__":
    main()
