"""
scripts/eval_localization_df1m.py — sliding-window localization eval for TALVAD-L
on AV-Deepfake1M (DF1M).

For each (variable-length) clip: tile it into `win`-frame windows (utils.window_ops),
run the model per window, decode each window's pyramid outputs to ABSOLUTE-second
segments, pool across all windows, and STITCH with Soft-NMS. Score with the existing
scripts/eval_temporal.compute_full_localization_metrics (AP@IoU / AR@N) and report a
clip-level detection AUC (max window clip-score per clip).

`evaluate_localization_windowed(...)` is reused as the dev evaluator during training
(injected into TrainerL via `make_dev_eval_fn`); `main()` runs a standalone test eval.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from utils import window_ops  # noqa: E402
from datasets.windowed_localization_dataset import load_clip_feats, crop_window  # noqa: E402
from eval_temporal import compute_full_localization_metrics  # noqa: E402


def _split_streams(x: torch.Tensor, off: Dict):
    a, ar, ap, sp = off["audio"], off["artic"], off["appearance"], off["spectral"]
    return (x[..., a[0]:a[1]], x[..., ar[0]:ar[1]],
            x[..., ap[0]:ap[1]], x[..., sp[0]:sp[1]])


@torch.no_grad()
def _decode_clip(model, feats: np.ndarray, n_frames: int, fps: float, win: int, hop: int,
                 strides: List[int], offsets: Dict, dcfg: Dict, scaler, device) -> Tuple[List, float]:
    """All windows of one clip -> stitched (s,e,score) absolute-seconds + clip detection score."""
    starts = window_ops.make_windows(n_frames, win, hop)
    batch = np.stack([crop_window(feats, ws, win) for ws in starts], 0)        # [W,win,D]
    if scaler is not None:
        W, T, D = batch.shape
        batch = scaler.transform(batch.reshape(-1, D)).reshape(W, T, D)
    x = torch.from_numpy(np.ascontiguousarray(batch)).float().to(device)
    a, ar, ap, sp = _split_streams(x, offsets)
    out = model(a, ar, ap, sp)
    L = len(out["level_cls"])
    clip_score = float(torch.sigmoid(out["clip_logits"]).squeeze(-1).max().cpu())
    use_qual = bool(dcfg.get("use_qual", False))
    score_thr = float(dcfg.get("score_thr", 0.05))

    pooled: List[Tuple[float, float, float]] = []
    for wi, ws in enumerate(starts):
        scores, regs, ctrs = [], [], []
        for l in range(L):
            cls = out["level_cls"][l][wi].squeeze(-1).float().cpu().numpy()
            s = 1.0 / (1.0 + np.exp(-cls))
            if use_qual and "level_qual" in out:
                q = 1.0 / (1.0 + np.exp(-out["level_qual"][l][wi].squeeze(-1).float().cpu().numpy()))
                s = np.sqrt(np.clip(s * q, 0.0, 1.0))
            scores.append(s)
            regs.append(out["level_reg"][l][wi].float().cpu().numpy())
            ctrs.append(out["level_centers"][l].float().cpu().numpy())
        pooled += window_ops.window_decode_to_sec(scores, regs, ctrs, ws, fps, score_thr)

    final = window_ops.stitch(pooled, sigma=float(dcfg.get("soft_nms_sigma", 0.5)),
                              score_thr=float(dcfg.get("soft_nms_thr", 0.001)),
                              max_keep=int(dcfg.get("max_segments", 100)))
    return final, clip_score


@torch.no_grad()
def evaluate_localization_windowed(
    model, feats_dir, index: Dict[str, dict], ids: List[str], frames: Dict[str, int],
    fps: float, win: int, hop: int, strides: List[int], offsets: Dict, dcfg: Dict,
    scaler, device, max_clips: int = 0,
) -> Dict:
    model.eval()
    feats_dir = Path(feats_dir)
    if max_clips and max_clips > 0:
        ids = ids[:max_clips]
    all_preds: List[Dict] = []
    all_gts: List[Dict] = []
    clip_scores, clip_labels = [], []
    for cid in ids:
        n = int(frames.get(cid, 0))
        if n <= 0:
            continue
        feats = load_clip_feats(feats_dir, cid)
        final, cscore = _decode_clip(model, feats, n, fps, win, hop, strides, offsets, dcfg, scaler, device)
        for (s, e, sc) in final:
            all_preds.append({"video_id": cid, "t_start": s, "t_end": e, "score": sc})
        for (gs, ge) in index.get(cid, {}).get("fake_periods", []):
            all_gts.append({"video_id": cid, "t_start": float(gs), "t_end": float(ge)})
        clip_scores.append(cscore)
        clip_labels.append(int(index.get(cid, {}).get("label", 0)))

    metrics = compute_full_localization_metrics(all_preds, all_gts)
    try:
        from sklearn.metrics import roc_auc_score
        cl = np.array(clip_labels)
        metrics["clip_auc"] = (float(roc_auc_score(cl, np.array(clip_scores)))
                               if len(np.unique(cl)) > 1 else float("nan"))
    except Exception:
        metrics["clip_auc"] = float("nan")
    metrics["n_pred"] = len(all_preds); metrics["n_gt"] = len(all_gts); metrics["n_clip"] = len(clip_scores)
    return metrics


def make_dev_eval_fn(feats_dir, index, ids, frames, fps, win, hop, strides, offsets,
                     dcfg, scaler, device, max_clips: int = 0) -> Callable:
    """Closure (model -> metrics) for injection into TrainerL.dev_eval_fn."""
    def _fn(model):
        return evaluate_localization_windowed(model, feats_dir, index, ids, frames, fps, win,
                                              hop, strides, offsets, dcfg, scaler, device, max_clips)
    return _fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pack", required=True, help="data/talvad_l_df1m/<split> dir with feats/ + manifest.json")
    ap.add_argument("--index", required=True, help="DF1M annotation min-index json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max_clips", type=int, default=0)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    from utils.config import load_config
    from models.talvad_l import TALVADLLocalizer
    from datasets.windowed_localization_dataset import load_pack
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pack = Path(args.pack)
    manifest, ids_by_split, frames, _ = load_pack(pack)
    index = json.loads(Path(args.index).read_text())
    ids = ids_by_split.get(args.split, list(frames.keys()))

    win = int(cfg["window"]["win_frames"]); hop = int(cfg["window"]["hop_frames"])
    fps = float(cfg["preprocessing"]["fps"])
    strides = [2 ** i for i in range(int(cfg["model"]["n_levels"]))]
    scaler = None
    ckpt = args.ckpt or str(Path(cfg["training"]["checkpoint_dir"]) / "best_model.pt")
    sp = Path(ckpt).parent / "scaler.joblib"
    if sp.exists():
        from joblib import load as jload
        scaler = jload(sp)

    mcfg = dict(cfg["model"])
    mcfg["dims"] = {"audio": manifest["audio_dim"], "artic": manifest["artic_dim"],
                    "appearance": manifest["appearance_dim"], "spectral": manifest["spectral_dim"]}
    model = TALVADLLocalizer(mcfg).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)

    m = evaluate_localization_windowed(model, pack / "feats", index, ids, frames, fps, win, hop,
                                       strides, manifest["offsets"], cfg["localization"]["decode"],
                                       scaler, device, args.max_clips)
    print(f"\n  TALVAD-L DF1M sliding-window eval [{args.split}]  "
          f"({m['n_clip']} clips, {m['n_pred']} preds / {m['n_gt']} gts)")
    for k in ["AP@50", "AP@75", "AP@90", "AP@95", "mAP", "AR@100", "AR@50", "AR@10", "clip_auc"]:
        if k in m:
            print(f"   {k:8s}: {m[k]:.4f}")


if __name__ == "__main__":
    main()
