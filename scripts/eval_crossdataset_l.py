"""
scripts/eval_crossdataset_l.py — zero-shot cross-dataset DETECTION eval for TALVAD-L.

TALVAD-L is a LAV-DF localizer, but its clip head produces a clip-level fake score.
Whole-clip deepfake datasets (FakeAVCeleb, DeepfakeTIMIT, DFDC) have no temporal
fake_periods, so localization AP@IoU is N/A there — what transfers is the clip-level
detection AUC. This script:

  1. Loads a target pack produced by
     `extract_talvad_l.py --target_csv ... --target_name ...`
     (data/talvad_l_crossdataset/<name>/packed/), already projected with the SAVED
     LAV-DF PCA bases (feature parity).
  2. Applies the SAVED LAV-DF scaler (checkpoints/talvad_l/scaler.joblib) AS-IS —
     nothing is fit on the target (this is what makes it honest / zero-shot).
  3. Runs TALVADLLocalizer's clip head and reports AUC / EER / ACC + confusion at the
     0.5 and (optionally) a LAV-DF-tuned threshold.

Usage:
  .venv\\Scripts\\python.exe scripts\\eval_crossdataset_l.py \
      --config configs/lavdf_talvad_l.yaml --target_name fakeavceleb_bal
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

from utils.config import load_config  # noqa: E402
from models.talvad_l import TALVADLLocalizer  # noqa: E402


def _split_streams(x: torch.Tensor, off: dict):
    a, ar, ap, sp = off["audio"], off["artic"], off["appearance"], off["spectral"]
    return (x[..., a[0]:a[1]], x[..., ar[0]:ar[1]],
            x[..., ap[0]:ap[1]], x[..., sp[0]:sp[1]])


def _metrics(labels: np.ndarray, scores: np.ndarray, thr: float = 0.5) -> dict:
    from sklearn.metrics import roc_auc_score, roc_curve
    out = {}
    if len(np.unique(labels)) > 1:
        out["auc"] = float(roc_auc_score(labels, scores))
        fpr, tpr, _ = roc_curve(labels, scores)
        fnr = 1 - tpr
        eer_i = int(np.nanargmin(np.abs(fnr - fpr)))
        out["eer"] = float((fpr[eer_i] + fnr[eer_i]) / 2.0)
    else:
        out["auc"] = float("nan"); out["eer"] = float("nan")
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum()); tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum()); fn = int(((pred == 0) & (labels == 1)).sum())
    out["acc"] = float((tp + tn) / max(len(labels), 1))
    out["tp"], out["tn"], out["fp"], out["fn"] = tp, tn, fp, fn
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    out["recall_fake"] = float(tp / max(n_pos, 1))
    out["recall_real"] = float(tn / max(n_neg, 1))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--target_name", required=True)
    ap.add_argument("--ckpt", default=None, help="defaults to <checkpoint_dir>/best_model.pt")
    ap.add_argument("--scaler", default=None, help="defaults to <ckpt dir>/scaler.joblib")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--instance_norm", action="store_true",
                    help="apply per-sample InstanceFeatureNorm (across time+feat) after the LAV-DF "
                         "scaler — the label-free cross-dataset de-confounding lever")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pack = REPO / "data" / "talvad_l_crossdataset" / args.target_name / "packed"
    if not (pack / "features.npy").exists():
        raise FileNotFoundError(f"No target pack at {pack}. Run extract_talvad_l.py target mode first.")

    feats = np.load(pack / "features.npy", mmap_mode="r")
    labels = np.load(pack / "labels.npy").astype(np.int64)
    manifest = json.loads((pack / "manifest.json").read_text())
    offsets = manifest["offsets"]

    ckpt = args.ckpt or str(Path(cfg["training"]["checkpoint_dir"]) / "best_model.pt")
    scaler_path = args.scaler or str(Path(ckpt).parent / "scaler.joblib")
    scaler = None
    if Path(scaler_path).exists():
        from joblib import load as jload
        scaler = jload(scaler_path)
        print(f"  applied LAV-DF scaler: {scaler_path}")
    else:
        print(f"  WARNING: no scaler at {scaler_path} — running on raw (unscaled) features.")

    mcfg = dict(cfg["model"])
    mcfg["dims"] = {"audio": manifest["audio_dim"], "artic": manifest["artic_dim"],
                    "appearance": manifest["appearance_dim"], "spectral": manifest["spectral_dim"]}
    model = TALVADLLocalizer(mcfg).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    print(f"  loaded TALVAD-L: {ckpt}")
    print(f"  target '{args.target_name}': {len(labels)} clips  "
          f"(real={int((labels==0).sum())} fake={int((labels==1).sum())})")

    scores = np.zeros(len(labels), dtype=np.float32)
    bs = args.batch_size
    for st in range(0, len(labels), bs):
        chunk = np.asarray(feats[st:st + bs], dtype=np.float32)              # [b,T,D]
        if scaler is not None:
            b, T, D = chunk.shape
            chunk = scaler.transform(chunk.reshape(-1, D)).reshape(b, T, D).astype(np.float32)
        if args.instance_norm:                                               # per-sample, across (T,D)
            mu = chunk.mean(axis=(1, 2), keepdims=True)
            sigma = np.maximum(chunk.std(axis=(1, 2), keepdims=True), 1e-6)
            chunk = ((chunk - mu) / sigma).astype(np.float32)
        x = torch.from_numpy(chunk).float().to(device)
        a, ar, apc, sp = _split_streams(x, offsets)
        out = model(a, ar, apc, sp)
        p = torch.sigmoid(out["clip_logits"]).squeeze(-1).cpu().numpy()
        scores[st:st + len(p)] = p

    m = _metrics(labels, scores, thr=args.thr)
    print(f"\n  TALVAD-L zero-shot cross-dataset detection [{args.target_name}]")
    print(f"   AUC        : {m['auc']:.4f}")
    print(f"   EER        : {m['eer']:.4f}")
    print(f"   ACC@{args.thr:.2f}   : {m['acc']:.4f}   (recall fake={m['recall_fake']:.3f} real={m['recall_real']:.3f})")
    print(f"   confusion  : tp={m['tp']} fp={m['fp']} tn={m['tn']} fn={m['fn']}")
    print(f"   score stats: mean={scores.mean():.3f} fake_mean={scores[labels==1].mean():.3f} "
          f"real_mean={scores[labels==0].mean():.3f}")

    out_dir = REPO / "logs" / cfg["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_instnorm" if args.instance_norm else ""
    out_file = out_dir / f"crossdataset_{args.target_name}{tag}.json"
    out_file.write_text(json.dumps({"target": args.target_name, "n": len(labels),
                                    "instance_norm": bool(args.instance_norm), **m,
                                    "fake_mean": float(scores[labels == 1].mean()),
                                    "real_mean": float(scores[labels == 0].mean())}, indent=2),
                        encoding="utf-8")
    print(f"  saved -> {out_file}")


if __name__ == "__main__":
    main()
