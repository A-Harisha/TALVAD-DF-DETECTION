"""
scripts/eval_robustness_l.py — TALVAD-L localization + detection eval on a
distorted robustness pack (built by make_robustness_videos.py and extracted in
target mode by extract_talvad_l.py).

Unlike eval_crossdataset_l.py (clip AUC only), this also decodes segments and
scores AP@IoU / AR@N against the LAV-DF ground truth carried in
datasets/robustness/gt.json — the distorted clips keep the original stems, and
target-pack ids are '<stem>_<md5:12>', so stems recover the annotations.

Usage:
  .venv\\Scripts\\python.exe scripts\\eval_robustness_l.py \
      --config configs/lavdf_talvad_l.yaml --target_name rob_vnoise_L3 \
      --gt datasets/robustness/gt.json --distortion vnoise --level 3 \
      --out logs/robustness/vnoise_L3.json
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

from utils.config import load_config          # noqa: E402
from models.talvad_l import TALVADLLocalizer  # noqa: E402
from eval_localization import decode_sample   # noqa: E402
from eval_temporal import compute_full_localization_metrics  # noqa: E402


def _split_streams(x: torch.Tensor, off: dict):
    a, ar, ap, sp = off["audio"], off["artic"], off["appearance"], off["spectral"]
    return (x[..., a[0]:a[1]], x[..., ar[0]:ar[1]],
            x[..., ap[0]:ap[1]], x[..., sp[0]:sp[1]])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--target_name", required=True)
    ap.add_argument("--gt", default="datasets/robustness/gt.json")
    ap.add_argument("--distortion", default="")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--ckpt", default=None,
                    help="default: checkpoints/talvad_l_full/best_model.pt (the DFL paper "
                         "model); the config's default talvad_l dir holds a stale pre-DFL run")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = int(cfg["training"]["sequence_length"])
    dcfg = cfg["localization"]["decode"]

    pack = REPO / "data" / "talvad_l_crossdataset" / args.target_name / "packed"
    feats = np.load(pack / "features.npy", mmap_mode="r")
    labels = np.load(pack / "labels.npy").astype(np.int64)
    ids = json.loads((pack / "ids.json").read_text())
    manifest = json.loads((pack / "manifest.json").read_text())
    offsets = manifest["offsets"]
    gt = json.loads((REPO / args.gt).read_text(encoding="utf-8"))

    ckpt = args.ckpt or str(REPO / "checkpoints" / "talvad_l_full" / "best_model.pt")
    scaler = None
    for cand in (Path(ckpt).parent / "scaler.joblib",
                 REPO / "checkpoints" / "talvad_l" / "scaler.joblib"):
        if cand.exists():
            from joblib import load as jload
            scaler = jload(cand)
            print(f"  applied LAV-DF scaler: {cand}")
            break

    mcfg = dict(cfg["model"])
    mcfg["dims"] = {"audio": manifest["audio_dim"], "artic": manifest["artic_dim"],
                    "appearance": manifest["appearance_dim"], "spectral": manifest["spectral_dim"]}
    model = TALVADLLocalizer(mcfg).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    print(f"  loaded {ckpt}; pack '{args.target_name}' n={len(labels)}")

    all_preds, all_gts, clip_scores = [], [], np.zeros(len(labels), np.float32)
    bs = args.batch_size
    for st in range(0, len(labels), bs):
        chunk = np.asarray(feats[st:st + bs], dtype=np.float32)
        if scaler is not None:
            b, T, D = chunk.shape
            chunk = scaler.transform(chunk.reshape(-1, D)).reshape(b, T, D).astype(np.float32)
        x = torch.from_numpy(chunk).float().to(device)
        a, ar, apc, spc = _split_streams(x, offsets)
        out = model(a, ar, apc, spc)
        clip_scores[st:st + x.shape[0]] = \
            torch.sigmoid(out["clip_logits"]).squeeze(-1).cpu().numpy()

        L = len(out["level_cls"])
        cls_np = [out["level_cls"][l].squeeze(-1).cpu().numpy() for l in range(L)]
        reg_np = [out["level_reg"][l].cpu().numpy() for l in range(L)]
        qual_np = ([out["level_qual"][l].squeeze(-1).cpu().numpy() for l in range(L)]
                   if "level_qual" in out else None)
        ctr_np = [out["level_centers"][l].cpu().numpy() for l in range(L)]
        cbm_np = out["cbm_b"].cpu().numpy() if "cbm_b" in out else np.zeros((x.shape[0], seq_len))
        rec_np = out["recon_err"].cpu().numpy() if "recon_err" in out else np.zeros((x.shape[0], seq_len))

        for bi in range(x.shape[0]):
            pid = ids[st + bi]
            stem = pid.rsplit("_", 1)[0]          # strip md5 suffix
            g = gt.get(stem)
            if g is None:
                print(f"  WARNING: no gt for {stem}, skipped"); continue
            dur = float(g["duration"])
            segs = decode_sample(
                [cls_np[l][bi] for l in range(L)],
                [reg_np[l][bi] for l in range(L)],
                ctr_np, cbm_np[bi], rec_np[bi], dur, seq_len, dcfg,
                level_qual=[qual_np[l][bi] for l in range(L)] if qual_np is not None else None)
            for (s, e, sc) in segs:
                all_preds.append({"video_id": stem, "t_start": s, "t_end": e, "score": sc})
            for (gs, ge) in g.get("fake_periods", []):
                all_gts.append({"video_id": stem, "t_start": float(gs), "t_end": float(ge)})

    m = compute_full_localization_metrics(all_preds, all_gts)
    from sklearn.metrics import roc_auc_score
    m["clip_auc"] = (float(roc_auc_score(labels, clip_scores))
                     if len(np.unique(labels)) > 1 else float("nan"))
    m.update({"distortion": args.distortion, "level": args.level,
              "target": args.target_name, "n_clip": int(len(labels)),
              "n_pred": len(all_preds), "n_gt": len(all_gts)})

    print(f"\n  robustness eval [{args.distortion} L{args.level}]  n={len(labels)}")
    for k in ["AP@50", "AP@75", "AP@95", "mAP", "AR@50", "clip_auc"]:
        if k in m:
            print(f"   {k:8s}: {m[k]:.4f}")
    out_path = Path(args.out) if args.out else \
        REPO / "logs" / "robustness" / f"{args.distortion}_L{args.level}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                    for k, v in m.items()}, indent=2), encoding="utf-8")
    print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
