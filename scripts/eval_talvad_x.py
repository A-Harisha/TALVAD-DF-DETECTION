"""
scripts/eval_talvad_x.py — evaluation for the TALVAD-X detector.

Reports the full benchmark suite:
  - Video-level: AUC, F1, BalAcc at fixed (0.5) and tuned thresholds.
  - Frame-level: AUC, best-F1 (threshold-swept on eval set).
  - Segment-level TAD protocol: AP@{0.5, 0.75, 0.9, 0.95}, mAP, AR@{100,50,30,20,10,5}.
  - CMIS-2 inconsistency map localisation (threshold-swept diagnostic).
  - Subtype breakdown (audio-only / video-only / audio+video fakes).

Usage (in-domain, official LAV-DF test split):
  python scripts/eval_talvad_x.py --config configs/lavdf_talvad_x.yaml

Cross-dataset (FakeAVCeleb, DeepfakeTIMIT) — use eval_crossdataset.py which now
respects the TALVAD-X model class automatically when --model_class talvad_x is set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from datasets.annotation_loader import _load_metadata_index  # noqa: E402
from eval_temporal import (  # noqa: E402 (reuse existing metric functions)
    compute_ar_at_n,
    compute_full_localization_metrics,
    compute_segment_ap,
    frame_scores_to_segments,
    localization_from_scores,
    segment_iou,
)
from models.talvad_x import TALVADXDetector  # noqa: E402
from utils.config import load_config  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────

def _apply_overrides(cfg: dict, sets: list) -> None:
    """Apply dotted key=value overrides (e.g. model.cmis.spectral_branch=false).
    Needed so an ablation checkpoint is loaded into the SAME architecture it was
    trained with — A0/A1/A2/A3 toggle CMIS sub-branches that change layer dims,
    so evaluating them under the default (full-CMIS) config fails to load."""
    for kv in sets or []:
        key, val = kv.split("=", 1)
        v: object = val
        if val.lower() in ("true", "false"):
            v = val.lower() == "true"
        else:
            try:
                v = int(val) if val.isdigit() else float(val)
            except ValueError:
                v = val
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v


def _get_entry(stem, ann_index):
    return ann_index.get(stem) or ann_index.get(str(int(stem)) if stem.isdigit() else stem)


def _get_gt_segs(stem, ann_index) -> list:
    entry = _get_entry(stem, ann_index)
    if entry is None or not entry.get("fake_periods"):
        return []
    return [(float(s), float(e)) for s, e in entry["fake_periods"]]


def _get_duration(stem, ann_index) -> float | None:
    """Per-clip duration (seconds) so the segment decoder can map the seq_len
    frames to real time as t = frame * duration / seq_len — matching how the GT
    frame labels are built. Without this, segment-AP collapses (LAV-DF clips are
    ~7.3 s median, not the 4 s the 48-frame@12fps stream would imply)."""
    entry = _get_entry(stem, ann_index)
    if entry is None:
        return None
    d = entry.get("duration")
    return float(d) if d is not None else None


def _fake_type(stem, ann_index) -> str:
    entry = ann_index.get(stem) or ann_index.get(str(int(stem)) if stem.isdigit() else stem)
    if entry is None:
        return "unknown"
    ma, mv = bool(entry.get("modify_audio")), bool(entry.get("modify_video"))
    if ma and mv:
        return "video+audio"
    if ma:
        return "audio_only"
    if mv:
        return "video_only"
    return "real"


# ── main ───────────────────────────────────────────────────────────────────

def run(config_path: str, split: str = "test", out_dir_override: str | None = None,
        frame_threshold: float = 0.5, limit: int = 0, ckpt_dir_override: str | None = None,
        sets: list | None = None) -> Dict:
    cfg = load_config(config_path)
    _apply_overrides(cfg, sets or [])
    seq_len = int(cfg["training"]["sequence_length"])
    fps_val = int(cfg["preprocessing"]["fps"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ann_json = cfg["dataset"].get("annotation_json")

    pack = REPO / "data" / "talvad_x" / f"fps{fps_val}_sl{seq_len}" / "packed"
    if not (pack / "features.npy").exists():
        raise FileNotFoundError(f"No packed features at {pack}. Run extract_talvad_x.py first.")

    manifest = json.loads((pack / "manifest.json").read_text())
    feats_all = np.load(pack / "features.npy").astype(np.float32)
    labels_all = np.load(pack / "labels.npy").astype(np.int64)
    ids_all = json.loads((pack / "ids.json").read_text())
    audio_dim = int(manifest["audio_dim"])
    video_dim = int(manifest["appearance_dim"] + manifest["spectral_dim"])

    # Filter to the requested split via video path
    from datasets.metadata import load_metadata
    ds = cfg["dataset"]
    meta = load_metadata(ds["metadata_file"], ds["root_dir"], ds["name"])
    id_to_path = {str(r["id"]): str(r["video_path"]) for _, r in meta.iterrows()}

    def _split_of(i):
        p = id_to_path.get(str(i), "").replace("\\", "/")
        if f"/{split}/" in p:
            return True
        if split == "train" and "/train/" not in p and "/dev/" not in p and "/test/" not in p:
            return True
        return False

    keep = [i for i, sid in enumerate(ids_all) if _split_of(sid)]
    if not keep:
        print(f"WARNING: no clips found for split='{split}'; evaluating all {len(ids_all)} clips.")
        keep = list(range(len(ids_all)))
    if limit and limit > 0:
        rng = np.random.default_rng(42)
        keep = list(rng.permutation(keep)[:limit])

    feats = feats_all[keep]
    labels = labels_all[keep]
    ids = [ids_all[i] for i in keep]
    print(f"  Evaluating {len(ids)} clips (split={split}) | audio_dim={audio_dim} video_dim={video_dim}")

    # Normalise with saved scaler. --ckpt_dir lets you point at a tagged
    # checkpoint (e.g. an ablation variant) without editing the config.
    ckpt_dir = Path(ckpt_dir_override or cfg["training"]["checkpoint_dir"])
    print(f"  Checkpoint dir: {ckpt_dir}")
    from joblib import load as jload
    scaler_path = ckpt_dir / "scaler.joblib"
    if scaler_path.exists():
        scaler = jload(scaler_path)
        feats = scaler.transform(feats.reshape(-1, feats.shape[-1])).reshape(feats.shape).astype(np.float32)

    # Load model
    model = TALVADXDetector(cfg["model"]).to(device)
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    tuned_thr = float(ckpt.get("threshold", 0.5)) if isinstance(ckpt, dict) else 0.5

    # Annotation index for GT segments
    ann_index = _load_metadata_index(ann_json) if ann_json else {}

    # ── Inference ─────────────────────────────────────────────────────────
    video_scores, video_labels = [], []
    frame_scores_all, frame_labels_all = [], []
    cmis_scores_all = []
    all_gts_seg: List[Dict] = []
    all_preds_seg: List[Dict] = []
    stems = []
    durations = []

    bs = 64
    with torch.no_grad():
        for i in range(0, len(feats), bs):
            xb = torch.from_numpy(feats[i:i+bs]).float().to(device)
            ax, vx = xb[..., :audio_dim], xb[..., audio_dim:]
            logits, frame_logits, attn = model(ax, vx)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            for j, (vscore, lbl, sid) in enumerate(zip(probs, labels[i:i+bs], ids[i:i+bs])):
                stem = str(sid)
                video_scores.append(float(vscore))
                video_labels.append(int(lbl))
                stems.append(stem)

                # Frame-level scores
                if frame_logits is not None:
                    fs = torch.sigmoid(frame_logits[j]).squeeze(-1).cpu().numpy()
                else:
                    fs = np.full(seq_len, float(vscore), dtype=np.float32)
                frame_scores_all.append(fs)

                # CMIS-2 inconsistency map
                ci = attn.get("cmis_incon")
                cmis_scores_all.append(ci[j].cpu().numpy() if ci is not None else None)

                # Frame GT labels
                if ann_index:
                    from datasets.annotation_loader import get_frame_labels
                    frame_labels_all.append(get_frame_labels(stem, seq_len, ann_json, fps=fps_val))
                else:
                    frame_labels_all.append(np.zeros(seq_len, dtype=np.float32))

                # Per-clip duration → correct frame→time mapping for decoding.
                dur = _get_duration(stem, ann_index)
                durations.append(dur)

                # GT + predicted segments
                for ts, te in _get_gt_segs(stem, ann_index):
                    all_gts_seg.append({"video_id": stem, "t_start": ts, "t_end": te})
                for ts, te, cf in frame_scores_to_segments(fs, frame_threshold, fps_val, duration=dur):
                    all_preds_seg.append({"video_id": stem, "t_start": ts, "t_end": te, "score": cf})

    # ── Metrics ────────────────────────────────────────────────────────────
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    )
    y = np.array(video_labels); s = np.array(video_scores)
    auc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")

    def _thresh_metrics(thr):
        p = (s >= thr).astype(int)
        return {"thr": thr, "auc": auc,
                "f1": float(f1_score(y, p, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y, p)),
                "accuracy": float(accuracy_score(y, p))}

    print(f"\n{'='*64}\n  VIDEO-LEVEL\n{'='*64}")
    m05 = _thresh_metrics(0.5); print(f"  [thr=0.50] AUC={auc:.4f} F1={m05['f1']:.4f} BalAcc={m05['balanced_accuracy']:.4f}")
    mtu = _thresh_metrics(tuned_thr); print(f"  [thr={tuned_thr:.2f}] AUC={auc:.4f} F1={mtu['f1']:.4f} BalAcc={mtu['balanced_accuracy']:.4f}")

    # Frame AUC
    fl = np.concatenate(frame_labels_all); fs_all = np.concatenate(frame_scores_all)
    frame_auc = float(roc_auc_score(fl, fs_all)) if 0 < int(fl.sum()) < len(fl) else float("nan")
    print(f"\n{'='*64}\n  FRAME-LEVEL  AUC={frame_auc:.4f}\n{'='*64}")

    # Full TAD protocol
    full_loc = compute_full_localization_metrics(all_preds_seg, all_gts_seg)
    print(f"\n{'='*64}\n  TEMPORAL LOCALIZATION (TAD protocol)\n{'='*64}")
    print(f"  {'Metric':<16} {'Value':>8}")
    for k, v in full_loc.items():
        print(f"  {k:<16} {v:>8.4f}")

    # Threshold-swept TFLH + CMIS-2 localisation
    tflh_loc = localization_from_scores(frame_scores_all, stems, frame_labels_all, all_gts_seg, fps_val,
                                        per_sample_durations=durations)
    print(f"\n  [TFLH] frame_AUC={tflh_loc['frame_auc']}  best_mAP={tflh_loc['best_map']:.4f}@{tflh_loc['best_map_threshold']:.2f}")
    cmis_loc = None
    if all(c is not None for c in cmis_scores_all):
        cmis_loc = localization_from_scores(cmis_scores_all, stems, frame_labels_all, all_gts_seg, fps_val,
                                            per_sample_durations=durations)
        print(f"  [CMIS-2] frame_AUC={cmis_loc['frame_auc']}  best_mAP={cmis_loc['best_map']:.4f}@{cmis_loc['best_map_threshold']:.2f}")

    # Save
    out_dir = Path(out_dir_override) if out_dir_override else (REPO / "logs" / cfg["experiment_name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"config": config_path, "split": split, "n": len(ids),
              "video_metrics_05": m05, "video_metrics_tuned": mtu,
              "frame_auc": frame_auc, "localization": full_loc,
              "tflh_localization": tflh_loc, "cmis_localization": cmis_loc}
    (out_dir / f"eval_{split}_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n  -> {out_dir / f'eval_{split}_results.json'}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--frame_threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ckpt_dir", default=None,
                    help="override checkpoint dir (e.g. checkpoints/talvad_x_A4_full_s42)")
    ap.add_argument("--set", action="append", default=[],
                    help="dotted model/config override key=value (repeatable); match the variant's training flags")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    run(args.config, args.split, args.out_dir, args.frame_threshold, args.limit, args.ckpt_dir, args.set)
