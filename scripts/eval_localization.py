"""
scripts/eval_localization.py — TALVAD-L localization evaluation.

Decodes the pyramid outputs into time segments (anchor-free, via utils.loc_ops),
applies the CBM boundary-energy gate + CMRD reconstruction-error gate, runs
Soft-NMS, and scores AP@IoU / AR@N with the EXISTING metric functions in
scripts/eval_temporal.py. Also reports clip-level AUC (detection).

`evaluate_localization(...)` is imported by training/trainer_loc.py for dev model
selection; `main()` runs a standalone eval of a saved checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from utils import loc_ops  # noqa: E402
from eval_temporal import compute_full_localization_metrics  # noqa: E402


def split_streams(x: torch.Tensor, off: Dict) -> Tuple[torch.Tensor, ...]:
    a, ar, ap, sp = off["audio"], off["artic"], off["appearance"], off["spectral"]
    return (x[..., a[0]:a[1]], x[..., ar[0]:ar[1]],
            x[..., ap[0]:ap[1]], x[..., sp[0]:sp[1]])


def decode_sample(
    level_cls: List[np.ndarray],      # [L] of [Tl]   (logits)
    level_reg: List[np.ndarray],      # [L] of [Tl,2] (level-0 units)
    level_centers: List[np.ndarray],  # [L] of [Tl]
    cbm_b: np.ndarray,                # [T0]
    recon_err: np.ndarray,            # [T0]
    duration: float,
    seq_len: int,
    dcfg: Dict,
    level_qual: List[np.ndarray] = None,   # [L] of [Tl] centerness logits
) -> List[Tuple[float, float, float]]:
    T0 = len(cbm_b)
    scores, regs, ctrs = [], [], []
    for li, (cls, reg, ctr) in enumerate(zip(level_cls, level_reg, level_centers)):
        s = 1.0 / (1.0 + np.exp(-cls))                      # sigmoid -> actionness
        if level_qual is not None and dcfg.get("use_qual", True):
            q = 1.0 / (1.0 + np.exp(-level_qual[li]))        # centerness -> quality
            s = np.sqrt(np.clip(s * q, 0.0, 1.0))            # geometric mean
        idx = np.clip(ctr.astype(int), 0, T0 - 1)
        if dcfg.get("use_cbm_gate", False):
            s = s * cbm_b[idx]
        if dcfg.get("use_recon_gate", False):
            s = s * np.clip(recon_err[idx], 0.0, 1.0)
        scores.append(s); regs.append(reg); ctrs.append(ctr)

    # decode in FRAME coords so boundaries can be CBM-refined, then -> seconds.
    fsegs = loc_ops.decode_frames(scores, regs, ctrs, float(dcfg.get("score_thr", 0.05)))
    if dcfg.get("cbm_refine", False):
        fsegs = loc_ops.refine_boundaries(fsegs, cbm_b, window=float(dcfg.get("cbm_window", 4.0)))
    if str(dcfg.get("method", "fusion")) == "fusion":
        fsegs = loc_ops.weighted_fusion(fsegs, iou_thr=float(dcfg.get("fusion_iou", 0.5)),
                                        max_keep=int(dcfg.get("max_segments", 100)))
    else:
        fsegs = loc_ops.soft_nms(fsegs, sigma=float(dcfg.get("soft_nms_sigma", 0.5)),
                                 score_thr=float(dcfg.get("soft_nms_thr", 0.001)),
                                 max_keep=int(dcfg.get("max_segments", 100)))
    out = []
    for sf, ef, sc in fsegs:
        s, e = loc_ops.frames_to_seconds(sf, ef, duration, seq_len)
        out.append((s, e, sc))
    return out


@torch.no_grad()
def evaluate_localization(model, loader, offsets, seq_len, dcfg, device) -> Dict:
    model.eval()
    ds = loader.dataset
    all_preds: List[Dict] = []
    all_gts: List[Dict] = []
    clip_scores, clip_labels = [], []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].numpy()
        idxs = batch["idx"].numpy()
        durs = batch["duration"].numpy()
        a, ar, ap, sp = split_streams(x, offsets)
        out = model(a, ar, ap, sp)
        clip_p = torch.sigmoid(out["clip_logits"]).squeeze(-1).cpu().numpy()
        clip_scores.extend(clip_p.tolist()); clip_labels.extend(y.astype(int).tolist())

        L = len(out["level_cls"])
        cls_np = [out["level_cls"][l].squeeze(-1).cpu().numpy() for l in range(L)]   # [L][B,Tl]
        reg_np = [out["level_reg"][l].cpu().numpy() for l in range(L)]               # [L][B,Tl,2]
        qual_np = [out["level_qual"][l].squeeze(-1).cpu().numpy() for l in range(L)] if "level_qual" in out else None
        ctr_np = [out["level_centers"][l].cpu().numpy() for l in range(L)]           # [L][Tl]
        cbm_np = out["cbm_b"].cpu().numpy() if "cbm_b" in out else np.zeros((x.shape[0], seq_len))
        rec_np = out["recon_err"].cpu().numpy() if "recon_err" in out else np.zeros((x.shape[0], seq_len))

        for bi in range(x.shape[0]):
            sid = ds.ids[int(idxs[bi])]
            dur = float(durs[bi])
            segs = decode_sample(
                [cls_np[l][bi] for l in range(L)],
                [reg_np[l][bi] for l in range(L)],
                ctr_np,
                cbm_np[bi], rec_np[bi], dur, seq_len, dcfg,
                level_qual=[qual_np[l][bi] for l in range(L)] if qual_np is not None else None,
            )
            for (s, e, sc) in segs:
                all_preds.append({"video_id": sid, "t_start": s, "t_end": e, "score": sc})
            for (gs, ge) in ds.gt_seconds[int(idxs[bi])]:
                all_gts.append({"video_id": sid, "t_start": gs, "t_end": ge})

    metrics = compute_full_localization_metrics(all_preds, all_gts)
    # clip AUC
    from sklearn.metrics import roc_auc_score
    cl = np.array(clip_labels); cs = np.array(clip_scores)
    metrics["clip_auc"] = float(roc_auc_score(cl, cs)) if len(np.unique(cl)) > 1 else float("nan")
    metrics["n_pred"] = len(all_preds); metrics["n_gt"] = len(all_gts)
    return metrics


def _build_loader_from_pack(cfg, split, batch_size, scaler_path=None):
    from torch.utils.data import DataLoader
    from datasets.sequence_dataset import LocalizationPackedDataset
    seq_len = int(cfg["training"]["sequence_length"])
    fps = int(cfg["preprocessing"]["fps"])
    pack = REPO / "data" / "talvad_l" / f"fps{fps}_sl{seq_len}" / "packed"
    feats = np.load(pack / "features.npy", mmap_mode="r")        # lazy; scaler applied per-sample
    scaler = None
    if scaler_path and Path(scaler_path).exists():
        from joblib import load as _jload
        scaler = _jload(scaler_path)
        print(f"  applied scaler: {scaler_path}")
    labels = np.load(pack / "labels.npy")
    ids = json.loads((pack / "ids.json").read_text())
    splits = json.loads((pack / "splits.json").read_text())
    manifest = json.loads((pack / "manifest.json").read_text())
    sel = np.where(np.array([s == split for s in splits]))[0]
    lc = cfg["localization"]
    ds = LocalizationPackedDataset(
        feats, labels[sel], [ids[i] for i in sel],              # full mmap + index_map
        sequence_length=seq_len, annotation_json=cfg["dataset"]["annotation_json"],
        strides=[2 ** i for i in range(int(cfg["model"]["n_levels"]))],
        ranges=[tuple(r) for r in lc["regression_ranges"]],
        center_radius=float(lc["center_radius"]), cbm_sigma=float(cfg["training"].get("cbm_sigma", 1.0)),
        scaler=scaler, index_map=sel,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False), manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default=None, help="defaults to <checkpoint_dir>/best_model.pt")
    args = ap.parse_args()
    from utils.config import load_config
    from models.talvad_l import TALVADLLocalizer
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = int(cfg["training"]["sequence_length"])
    ckpt = args.ckpt or str(Path(cfg["training"]["checkpoint_dir"]) / "best_model.pt")
    scaler_path = Path(ckpt).parent / "scaler.joblib"
    loader, manifest = _build_loader_from_pack(cfg, args.split, int(cfg["inference"]["batch_size"]), scaler_path)

    mcfg = dict(cfg["model"])
    mcfg["dims"] = {"audio": manifest["audio_dim"], "artic": manifest["artic_dim"],
                    "appearance": manifest["appearance_dim"], "spectral": manifest["spectral_dim"]}
    model = TALVADLLocalizer(mcfg).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)

    m = evaluate_localization(model, loader, manifest["offsets"], seq_len,
                              cfg["localization"]["decode"], device)
    print(f"\n  TALVAD-L localization eval [{args.split}]  ({m['n_pred']} preds / {m['n_gt']} gts)")
    for k in ["AP@50", "AP@75", "AP@90", "AP@95", "mAP", "AR@100", "AR@50", "AR@10", "clip_auc"]:
        if k in m:
            print(f"   {k:8s}: {m[k]:.4f}")
    out = Path(cfg["training"]["logs_dir"]) / cfg["experiment_name"] / f"loc_eval_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                               for k, v in m.items()}, indent=2), encoding="utf-8")
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
