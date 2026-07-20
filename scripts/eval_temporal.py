"""
scripts/eval_temporal.py

Temporal forgery localisation evaluation for TALVAD v5.

Metrics computed:
  1. Video-level  — AUC, F1, Accuracy, Balanced Accuracy, Confusion Matrix
  2. Frame-level  — AUC per timestep (binary GT from fake_periods)
  3. Segment AP   — Standard temporal detection AP at IoU >= 0.1, 0.3, 0.5
  4. Subtype breakdown — audio_only / video_only / video+audio fakes

Split:
  - Honours dataset.use_official_split — when true, evaluates on the official
    LAV-DF test partition (no source-video leakage). Falls back to the custom
    random 70/15/15 split otherwise.

Notes:
  - If the test GFC cache is missing, zeros are used for the 8 GFC dims unless
    --compute_gfc is passed (computes on demand from raw video).

Usage:
  python scripts/eval_temporal.py --config configs/lavdf_talvad_v5.yaml
  python scripts/eval_temporal.py --config configs/lavdf_talvad_v5.yaml --compute_gfc
  python scripts/eval_temporal.py --config configs/lavdf_talvad_v5.yaml --frame_threshold 0.4
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Repo root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.builder import build_dataset_splits
from features.cache import build_cache_signature
from models.bigru_detector import BiGRUSoftplusDetector
from utils.config import load_config


# ── Utility: segment IoU ──────────────────────────────────────────────────────

def segment_iou(p_start: float, p_end: float, g_start: float, g_end: float) -> float:
    inter = max(0.0, min(p_end, g_end) - max(p_start, g_start))
    union = (p_end - p_start) + (g_end - g_start) - inter
    return inter / union if union > 1e-9 else 0.0


def frame_scores_to_segments(
    frame_scores: np.ndarray,
    threshold: float,
    fps: float,
    duration: float | None = None,
) -> List[Tuple[float, float, float]]:
    """
    Convert per-frame sigmoid scores to predicted time segments.
    Returns list of (t_start_sec, t_end_sec, confidence) sorted by confidence desc.

    Time mapping:
      - If ``duration`` is given, the T frames are assumed to span the whole
        clip, so seconds-per-frame = duration / T. This MUST match how the GT
        frame labels are built (datasets.annotation_loader.get_frame_labels:
        ``t -> t * duration / seq_len``); otherwise predicted segments land in
        the wrong time window and segment-AP collapses to ~0. LAV-DF clips have
        a median duration of ~7.3 s, so the 48-frame sequence is NOT a real-time
        12 fps stream — duration scaling is required.
      - If ``duration`` is None, falls back to the legacy real-time assumption
        seconds-per-frame = 1 / fps (correct only when T/fps == duration).
    """
    segments = []
    T = len(frame_scores)
    spf = (duration / T) if (duration and T > 0) else (1.0 / fps)
    in_seg = False
    seg_start = 0
    for i in range(T + 1):
        score = frame_scores[i] if i < T else -1.0
        if score >= threshold and not in_seg:
            in_seg = True
            seg_start = i
        elif (score < threshold or i == T) and in_seg:
            in_seg = False
            conf = float(np.max(frame_scores[seg_start:i]))
            t_start = seg_start * spf
            t_end = i * spf
            segments.append((t_start, t_end, conf))
    segments.sort(key=lambda s: -s[2])
    return segments


def compute_segment_ap(
    all_preds: List[Dict],
    all_gts: List[Dict],
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute segment-level Average Precision across all videos.
    Standard temporal action detection protocol.

    all_preds: list of {"video_id": str, "t_start": float, "t_end": float, "score": float}
    all_gts:   list of {"video_id": str, "t_start": float, "t_end": float}
    """
    if not all_gts or not all_preds:
        return 0.0

    n_gt = len(all_gts)

    # Sort predictions by descending confidence
    all_preds = sorted(all_preds, key=lambda p: -p["score"])

    # Build GT lookup: video_id -> list of (t_start, t_end, matched)
    gt_by_video: Dict[str, List] = {}
    for g in all_gts:
        gt_by_video.setdefault(g["video_id"], []).append(
            [g["t_start"], g["t_end"], False]
        )

    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    for i, pred in enumerate(all_preds):
        vid = pred["video_id"]
        gts = gt_by_video.get(vid, [])
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gts):
            iou = segment_iou(pred["t_start"], pred["t_end"], gt[0], gt[1])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold and best_j >= 0 and not gts[best_j][2]:
            tp[i] = 1.0
            gts[best_j][2] = True
        else:
            fp[i] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall    = cum_tp / n_gt
    precision = cum_tp / (cum_tp + cum_fp + 1e-9)

    # Add sentinel points
    recall    = np.concatenate([[0.0], recall,    [recall[-1]]])
    precision = np.concatenate([[1.0], precision, [0.0]])

    # Area under PR curve (trapezoidal)
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    ap = float(trapz(precision, recall))
    return ap


def compute_ar_at_n(
    all_preds: List[Dict],
    all_gts: List[Dict],
    n_proposals: int,
    iou_thresholds: tuple = (0.5, 0.75, 0.95),
) -> float:
    """
    Average Recall @ N proposals (AR@N), averaged over iou_thresholds.
    Standard ActivityNet / TAD benchmark protocol used by UMMAFormer/AuViRe.

    For each video, keep only the top-n_proposals predictions (by score).
    A GT segment is recalled if any surviving prediction achieves IoU >= threshold.
    AR@N = mean over thresholds of (n_recalled_gt / n_total_gt).
    """
    if not all_gts or not all_preds:
        return 0.0

    # Top-N predictions per video
    preds_by_vid: Dict[str, List] = {}
    for p in all_preds:
        preds_by_vid.setdefault(p["video_id"], []).append(p)
    top_preds = []
    for vid, ps in preds_by_vid.items():
        ps.sort(key=lambda x: -x["score"])
        top_preds.extend(ps[:n_proposals])

    gt_by_vid: Dict[str, List] = {}
    for g in all_gts:
        gt_by_vid.setdefault(g["video_id"], []).append(g)

    recalls = []
    for iou_thr in iou_thresholds:
        n_recalled = 0
        for vid, gts in gt_by_vid.items():
            vid_preds = [p for p in top_preds if p["video_id"] == vid]
            for g in gts:
                for p in vid_preds:
                    if segment_iou(p["t_start"], p["t_end"], g["t_start"], g["t_end"]) >= iou_thr:
                        n_recalled += 1
                        break
        recalls.append(n_recalled / max(len(all_gts), 1))
    return float(np.mean(recalls))


def compute_full_localization_metrics(
    all_preds: List[Dict],
    all_gts: List[Dict],
    iou_thresholds: tuple = (0.5, 0.75, 0.9, 0.95),
    ar_n_values: tuple = (100, 50, 30, 20, 10, 5),
) -> Dict:
    """
    Full TAD benchmark suite: AP@IoU thresholds + AR@N values.
    Matches the protocol used by UMMAFormer, AuViRe, AV-Deepfake1M.
    """
    result: Dict = {}
    for iou in iou_thresholds:
        key = f"AP@{int(iou*100)}"
        result[key] = compute_segment_ap(all_preds, all_gts, iou_threshold=iou)
    result["mAP"] = float(np.mean([result[f"AP@{int(iou*100)}"] for iou in iou_thresholds]))
    for n in ar_n_values:
        result[f"AR@{n}"] = compute_ar_at_n(all_preds, all_gts, n_proposals=n)
    return result


def localization_from_scores(
    per_sample_scores: List[np.ndarray],
    per_sample_stems: List[str],
    frame_labels_all: List[np.ndarray],
    all_gts_seg: List[Dict],
    fps: float,
    sweep: Optional[np.ndarray] = None,
    per_sample_durations: Optional[List[float]] = None,
) -> Dict:
    """
    Threshold-free + threshold-tuned localisation metrics for an arbitrary
    per-timestep score source — either the TFLH frame head OR the CMIS
    inconsistency map. This is the apples-to-apples way to ask "does CMIS's
    *own* signal localise forgeries?", which the legacy report never measured
    (it scored TFLH only, then decoded segments at a fixed 0.5 threshold that
    collapses a low-positive-rate / label-smoothed head to ~0 predictions).

      - frame_auc       : ranking quality, threshold-independent.
      - best_frame_f1   : best frame F1 over a threshold sweep.
      - best_map        : mAP (segment AP @ IoU 0.1/0.3/0.5) at the decode
                          threshold that maximises it.

    NOTE: the threshold is tuned on the eval set, so best_map is a diagnostic
    *upper bound* on achievable localisation, not a held-out number. For a
    paper, re-tune the chosen threshold on dev. Its purpose here is to answer
    "is the localisation signal present at all?" without a bad fixed threshold
    masking it.
    """
    from sklearn.metrics import roc_auc_score, f1_score

    fs = np.concatenate(per_sample_scores)
    fl = np.concatenate(frame_labels_all)
    has_pos = 0 < int(fl.sum()) < len(fl)
    frame_auc = float(roc_auc_score(fl, fs)) if has_pos else None

    if sweep is None:
        sweep = np.round(np.arange(0.05, 0.96, 0.05), 2)

    best_f1, best_f1_thr = -1.0, 0.5
    for thr in sweep:
        f1 = f1_score(fl, (fs >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_f1_thr = float(f1), float(thr)

    durs = per_sample_durations if per_sample_durations is not None else [None] * len(per_sample_stems)
    best_map, best_map_thr, best_seg_ap, best_npred = -1.0, 0.5, {}, 0
    for thr in sweep:
        preds: List[Dict] = []
        for sc, stem, dur in zip(per_sample_scores, per_sample_stems, durs):
            for (ts, te, cf) in frame_scores_to_segments(sc, float(thr), fps, duration=dur):
                preds.append({"video_id": stem, "t_start": ts, "t_end": te, "score": cf})
        seg_ap = {
            f"ap_{int(i*10):02d}": compute_segment_ap(preds, all_gts_seg, i)
            for i in (0.1, 0.3, 0.5)
        }
        m = float(np.mean(list(seg_ap.values())))
        if m > best_map:
            best_map, best_map_thr, best_seg_ap, best_npred = m, float(thr), seg_ap, len(preds)

    return {
        "frame_auc":               frame_auc,
        "best_frame_f1":           best_f1,
        "best_frame_f1_threshold": best_f1_thr,
        "best_map":                best_map,
        "best_map_threshold":      best_map_thr,
        "best_map_segment_ap":     best_seg_ap,
        "best_map_n_pred_segments": best_npred,
    }


# ── Feature loading ───────────────────────────────────────────────────────────

def load_features_for_sample(
    sample_id: str,
    row: dict,
    feature_dir: Path,
    gfc_cache_dir: Path,
    sequence_length: int,
    n_gfc_bands: int,
    audio_dim: int,
    compute_gfc: bool = False,
    fps: float = 2.0,
) -> Optional[np.ndarray]:
    """
    Load [sequence_length, audio_dim + video_dim + n_gfc_bands] feature array.
    Returns None if the base audio+video cache file is missing.
    """
    seq_path = feature_dir / f"{sample_id}_sequence.npy"
    if not seq_path.exists():
        return None

    seq = np.load(seq_path)   # [T, D] — may already include GFC if from extract_all.py

    full_dim = audio_dim + (n_gfc_bands if n_gfc_bands > 0 else 0) + 512
    if seq.shape[-1] >= full_dim:
        # GFC already embedded — pad/trim to sequence_length and return as-is
        if seq.shape[0] != sequence_length:
            if seq.shape[0] > sequence_length:
                seq = seq[:sequence_length]
            else:
                pad = np.repeat(seq[-1:], sequence_length - seq.shape[0], axis=0)
                seq = np.concatenate([seq, pad], axis=0)
        return seq.astype(np.float32)

    # ── GFC (8 bands) — append to legacy 632-d cache ─────────────────────────
    gfc_path = gfc_cache_dir / f"{sample_id}_gfc.npy"
    if gfc_path.exists():
        raw_gfc = np.load(gfc_path)  # [T_frames, 8] — may differ from sequence_length
        if len(raw_gfc) == 0:
            gfc = np.zeros((sequence_length, n_gfc_bands), dtype=np.float32)
        elif len(raw_gfc) != sequence_length:
            indices = np.linspace(0, len(raw_gfc) - 1, sequence_length)
            lo = np.floor(indices).astype(int)
            hi = np.minimum(lo + 1, len(raw_gfc) - 1)
            frac = (indices - lo)[:, None]
            gfc = (raw_gfc[lo] * (1 - frac) + raw_gfc[hi] * frac).astype(np.float32)
        else:
            gfc = raw_gfc.astype(np.float32)
    elif compute_gfc:
        try:
            from features.gfc import extract_gfc_features
            from preprocessing.video import extract_frames_in_memory
            frames = extract_frames_in_memory(row["video_path"], fps=fps,
                                               frame_size=(224, 224))
            raw_gfc = extract_gfc_features(frames, n_bands=n_gfc_bands)
            np.save(gfc_path, raw_gfc)
            if len(raw_gfc) == 0:
                gfc = np.zeros((sequence_length, n_gfc_bands), dtype=np.float32)
            elif len(raw_gfc) != sequence_length:
                indices = np.linspace(0, len(raw_gfc) - 1, sequence_length)
                lo = np.floor(indices).astype(int)
                hi = np.minimum(lo + 1, len(raw_gfc) - 1)
                frac = (indices - lo)[:, None]
                gfc = (raw_gfc[lo] * (1 - frac) + raw_gfc[hi] * frac).astype(np.float32)
            else:
                gfc = raw_gfc.astype(np.float32)
        except Exception:
            gfc = np.zeros((sequence_length, n_gfc_bands), dtype=np.float32)
    else:
        # Test set lacks GFC cache — use zeros (see eval_temporal limitations)
        gfc = np.zeros((sequence_length, n_gfc_bands), dtype=np.float32)

    if seq.shape[0] != sequence_length:
        if seq.shape[0] > sequence_length:
            seq = seq[:sequence_length]
        else:
            pad = np.repeat(seq[-1:], sequence_length - seq.shape[0], axis=0)
            seq = np.concatenate([seq, pad], axis=0)

    return np.concatenate([seq, gfc], axis=-1)   # [T, 640]


# ── Annotation helpers ────────────────────────────────────────────────────────

def load_annotation_index(json_path: str) -> Dict[str, dict]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return {Path(e["file"]).stem: e for e in data}


def make_frame_gt(stem: str, sequence_length: int, annotation_index: Dict) -> np.ndarray:
    """
    Per-timestep binary ground truth from fake_periods.
    Uses STEM (e.g. '000001'), NOT the numeric sample_id.
    """
    entry = annotation_index.get(stem)
    if entry is None or not entry.get("fake_periods"):
        return np.zeros(sequence_length, dtype=np.float32)
    duration = float(entry["duration"])
    labels = np.zeros(sequence_length, dtype=np.float32)
    for t in range(sequence_length):
        t_mid = (t + 0.5) * duration / sequence_length
        for start, end in entry["fake_periods"]:
            if start <= t_mid <= end:
                labels[t] = 1.0
                break
    return labels


def get_gt_segments(stem: str, annotation_index: Dict) -> Tuple[List, str]:
    """
    Returns (gt_segments_sec, fake_type).
    fake_type: 'real', 'audio_only', 'video_only', 'video+audio'
    """
    entry = annotation_index.get(stem)
    if entry is None or entry.get("n_fakes", 0) == 0:
        return [], "real"
    mv = entry.get("modify_video", False)
    ma = entry.get("modify_audio", False)
    if mv and ma:
        fake_type = "video+audio"
    elif mv:
        fake_type = "video_only"
    elif ma:
        fake_type = "audio_only"
    else:
        fake_type = "unknown"
    return entry.get("fake_periods", []), fake_type


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_eval(
    config_path: str,
    frame_threshold: float = 0.5,
    compute_gfc: bool = False,
    out_dir: Optional[str] = None,
) -> Dict:
    cfg = load_config(config_path)
    # Mirror train.py: expose the GFC band count to the model so CMIS (if enabled
    # in the checkpoint's config) can slice the spectral sub-features correctly.
    _gfc_cfg = cfg.get("features", {}).get("gfc", {})
    cfg.setdefault("model", {}).setdefault(
        "gfc_bands",
        int(_gfc_cfg.get("n_bands", 0)) if _gfc_cfg.get("enabled", False) else 0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*64}")
    print(f"  TALVAD Temporal Evaluation — {cfg['experiment_name']}")
    print(f"  Config : {config_path}")
    print(f"  Device : {device}  |  Frame threshold : {frame_threshold}")
    print(f"{'='*64}")

    # ── Paths ─────────────────────────────────────────────────────────────────
    ckpt_dir     = Path(cfg["training"]["checkpoint_dir"])
    model_path   = ckpt_dir / "best_model.pt"
    scaler_path  = ckpt_dir / "global_scaler.pkl"
    mask_path    = ckpt_dir / "feature_mask.npy"

    cache_sig    = build_cache_signature(cfg)
    # extract_all.py saves to data/features/<sig>; train.py (non-packed) uses
    # cfg["dataset"]["cache_dir"]/<sig>.  Try both so either workflow works.
    _candidate_dirs = [
        Path("data/features") / cache_sig,
        Path(cfg["dataset"]["cache_dir"]) / cache_sig,
        Path(cfg["training"]["checkpoint_dir"]) / "norm_cache",
    ]
    feature_dir = next((d for d in _candidate_dirs if d.exists() and any(d.iterdir())), _candidate_dirs[0])
    print(f"  Feature dir : {feature_dir}  ({'found' if feature_dir.exists() else 'NOT FOUND'})")
    gfc_cache_dir = Path(cfg["dataset"].get("gfc_cache_dir", "./data/cache_v5_gfc"))
    annotation_json = cfg.get("dataset", {}).get("annotation_json")

    out_path = Path(out_dir) if out_dir else Path(cfg["training"]["logs_dir"]) / cfg["experiment_name"]
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Config params ─────────────────────────────────────────────────────────
    seq_len      = cfg["training"].get("sequence_length", 24)
    fps          = cfg["preprocessing"].get("fps", 2.0)
    n_mfcc       = cfg["features"]["audio"]["n_mfcc"]
    audio_dim    = n_mfcc * 3            # MFCC + Delta + DeltaDelta
    gfc_enabled  = cfg.get("features", {}).get("gfc", {}).get("enabled", False)
    n_gfc        = int(cfg.get("features", {}).get("gfc", {}).get("n_bands", 8)) if gfc_enabled else 0
    from features.video_cnn import CNN_FEATURE_DIM
    video_dim    = CNN_FEATURE_DIM + n_gfc   # 512 + 8 = 520

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if not annotation_json or not Path(annotation_json).exists():
        raise FileNotFoundError(
            f"annotation_json not found: {annotation_json}\n"
            "Set dataset.annotation_json in your config to the metadata.min.json path."
        )

    # ── Load annotation index ──────────────────────────────────────────────────
    print(f"\n  Loading annotation index: {annotation_json}")
    ann_index = load_annotation_index(annotation_json)
    print(f"  Indexed {len(ann_index):,} videos")

    # ── Reconstruct test split (must match training seed) ─────────────────────
    print(f"\n  Reconstructing test split (seed={cfg['seed']}) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, test_df = build_dataset_splits(cfg)
    print(f"  Test split : {len(test_df):,} samples")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n  Loading model from {model_path}")
    model = BiGRUSoftplusDetector(audio_dim=audio_dim, video_dim=video_dim, cfg=cfg["model"])
    state = torch.load(model_path, map_location=device)
    model_state = state["model"] if "model" in state else state
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    saved_threshold = float(state.get("threshold", 0.5)) if isinstance(state, dict) else 0.5
    print(f"  Parameters : {n_params:,}  |  TFLH : {'on' if model.use_tflh else 'off'}")
    print(f"  Saved threshold from checkpoint : {saved_threshold}")

    # ── Load global scaler ────────────────────────────────────────────────────
    global_scaler = None
    if scaler_path.exists():
        import joblib
        from features.fusion import FeatureFusionNormalizer
        global_scaler = FeatureFusionNormalizer()
        global_scaler.load(scaler_path)
        print(f"  Global scaler loaded from {scaler_path}")
    else:
        print(f"  WARNING: Global scaler not found at {scaler_path} — features will not be normalised!")

    # ── Load feature mask ─────────────────────────────────────────────────────
    feature_mask = None
    if mask_path.exists():
        feature_mask = np.load(mask_path).astype(np.float32)
        n_active = int(feature_mask.sum())
        print(f"  Feature mask : {n_active} / {len(feature_mask)} active dims")

    # ── Inference loop ────────────────────────────────────────────────────────
    print(f"\n  Running inference on {len(test_df):,} test samples ...")
    print(f"  GFC mode: {'compute on demand' if compute_gfc else 'zeros for missing'}")

    video_scores:  List[float] = []
    video_labels:  List[int]   = []
    frame_scores_all: List[np.ndarray] = []
    cmis_scores_all:  List[Optional[np.ndarray]] = []   # CMIS inconsistency map per sample
    frame_labels_all: List[np.ndarray] = []
    sample_metas:  List[dict]  = []

    all_preds_seg:  List[dict] = []   # for segment AP
    all_gts_seg:    List[dict] = []

    missing_feat = 0
    gfc_zero_count = 0

    from tqdm import tqdm
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="  Inference", unit="sample"):
        sample_id = str(row.get("id", idx))
        video_path = row["video_path"]
        stem = Path(video_path).stem   # e.g. "000001" — correct for annotation lookup

        # Load features
        feat = load_features_for_sample(
            sample_id=sample_id,
            row=row.to_dict(),
            feature_dir=feature_dir,
            gfc_cache_dir=gfc_cache_dir,
            sequence_length=seq_len,
            n_gfc_bands=n_gfc,
            audio_dim=audio_dim,
            compute_gfc=compute_gfc,
            fps=fps,
        )
        if feat is None:
            missing_feat += 1
            continue

        # Check if GFC was zeroed
        if n_gfc > 0 and not (gfc_cache_dir / f"{sample_id}_gfc.npy").exists():
            gfc_zero_count += 1

        # Apply feature mask
        if feature_mask is not None:
            feat = feat * feature_mask

        # Normalise (global scaler)
        feat_2d = feat   # [T, D]
        if global_scaler is not None:
            feat_2d = global_scaler.transform(feat_2d)

        # Forward pass
        feat_t = torch.from_numpy(feat_2d).float().unsqueeze(0).to(device)  # [1, T, D]
        a_x = feat_t[..., :audio_dim]      # [1, T, 120]
        v_x = feat_t[..., audio_dim:]      # [1, T, 520]

        with torch.no_grad():
            logits, frame_logits, attn = model(a_x, v_x)

        # CMIS per-timestep inconsistency map (paper figure / localisation signal)
        cmis_incon = attn.get("cmis_incon") if isinstance(attn, dict) else None
        cmis_curve = (
            cmis_incon.cpu().squeeze(0).numpy() if cmis_incon is not None else None
        )

        # MAA per-modality fake scores (for modality-attribution metrics)
        _a_log = attn.get("audio_logits") if isinstance(attn, dict) else None
        _v_log = attn.get("video_logits") if isinstance(attn, dict) else None
        audio_fake_score = float(torch.sigmoid(_a_log).cpu().squeeze()) if _a_log is not None else None
        video_fake_score = float(torch.sigmoid(_v_log).cpu().squeeze()) if _v_log is not None else None

        vscore = float(torch.sigmoid(logits).cpu().squeeze())
        video_scores.append(vscore)
        video_labels.append(int(row["label"]))

        # Frame-level predictions
        if frame_logits is not None:
            fscores = torch.sigmoid(frame_logits).cpu().squeeze(0).squeeze(-1).numpy()
        else:
            fscores = np.full(seq_len, vscore, dtype=np.float32)
        frame_scores_all.append(fscores)
        # CMIS inconsistency map (its own localisation signal), aligned per-sample
        cmis_scores_all.append(
            cmis_curve.astype(np.float32) if cmis_curve is not None else None
        )

        # Frame-level GT (using STEM for correct annotation lookup)
        frame_gt = make_frame_gt(stem, seq_len, ann_index)
        frame_labels_all.append(frame_gt)

        # GT segments (from fake_periods in seconds)
        gt_segs, fake_type = get_gt_segments(stem, ann_index)
        for seg in gt_segs:
            all_gts_seg.append({
                "video_id": stem,
                "t_start": float(seg[0]),
                "t_end":   float(seg[1]),
            })

        # Predicted segments (from frame-level scores)
        pred_segs = frame_scores_to_segments(fscores, frame_threshold, fps)
        for (t_start, t_end, conf) in pred_segs:
            all_preds_seg.append({
                "video_id": stem,
                "t_start": t_start,
                "t_end":   t_end,
                "score":   conf,
            })

        sample_metas.append({
            "sample_id":   sample_id,
            "stem":        stem,
            "video_path":  str(video_path),
            "label":       int(row["label"]),
            "fake_type":   fake_type,
            "video_score": vscore,
            "audio_fake_score": audio_fake_score,
            "video_fake_score": video_fake_score,
            "frame_scores": fscores.tolist(),
            "frame_gt":     frame_gt.tolist(),
            "cmis_incon":   cmis_curve.tolist() if cmis_curve is not None else None,
            "gt_segs":      gt_segs,
        })

    n_evaluated = len(video_scores)
    print(f"\n  Evaluated  : {n_evaluated:,} samples  ({missing_feat} skipped — no cache)")
    if gfc_zero_count:
        print(f"  GFC zeros  : {gfc_zero_count:,} samples used zero-padded GFC (test set cache miss)")

    # ── Compute metrics ───────────────────────────────────────────────────────
    from sklearn.metrics import (
        roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score,
        confusion_matrix,
    )

    y_true   = np.array(video_labels)
    y_scores = np.array(video_scores)
    fixed_thresh = float(cfg.get("evaluation", {}).get("threshold", 0.5))
    y_pred   = (y_scores >= fixed_thresh).astype(int)

    # Also try tuned threshold from training (frozen from the best val checkpoint)
    tuned_thresh = saved_threshold
    y_pred_tuned = (y_scores >= tuned_thresh).astype(int)

    print(f"\n{'='*64}")
    print(f"  1. VIDEO-LEVEL METRICS  (threshold={fixed_thresh} / tuned={tuned_thresh})")
    print(f"{'='*64}")

    def _metrics(y_t, y_p, y_s, tag=""):
        auc  = roc_auc_score(y_t, y_s) if len(np.unique(y_t)) > 1 else float("nan")
        f1   = f1_score(y_t, y_p, zero_division=0)
        acc  = accuracy_score(y_t, y_p)
        bacc = balanced_accuracy_score(y_t, y_p)
        cm   = confusion_matrix(y_t, y_p)
        print(f"  {tag}AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}  BalAcc={bacc:.4f}")
        print(f"  Confusion: TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
        return {"auc": float(auc), "f1": float(f1), "accuracy": float(acc),
                "balanced_accuracy": float(bacc),
                "confusion": {"tn": int(cm[0,0]), "fp": int(cm[0,1]),
                               "fn": int(cm[1,0]), "tp": int(cm[1,1])}}

    vid_metrics_05     = _metrics(y_true, y_pred,       y_scores, tag=f"[thr={fixed_thresh:.2f}] ")
    vid_metrics_tuned  = _metrics(y_true, y_pred_tuned, y_scores, tag=f"[thr={tuned_thresh:.2f}] ")

    # ── Frame-level AUC ───────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  2. FRAME-LEVEL AUC")
    print(f"     NOTE: TFLH trained on all-zero labels (ID mismatch bug in v5)")
    print(f"     Correct stem-based GT used here for evaluation.")
    print(f"{'='*64}")

    all_frame_scores = np.concatenate(frame_scores_all)
    all_frame_labels = np.concatenate(frame_labels_all)

    n_frame_pos = int(all_frame_labels.sum())
    n_frame_tot = len(all_frame_labels)
    print(f"  Total timesteps: {n_frame_tot:,}  |  Fake timesteps: {n_frame_pos:,} ({100*n_frame_pos/n_frame_tot:.1f}%)")

    if n_frame_pos > 0 and n_frame_pos < n_frame_tot:
        frame_auc = roc_auc_score(all_frame_labels, all_frame_scores)
        frame_f1  = f1_score(all_frame_labels, (all_frame_scores >= frame_threshold).astype(int), zero_division=0)
        print(f"  Frame AUC = {frame_auc:.4f}  |  Frame F1 = {frame_f1:.4f}  (thr={frame_threshold})")
    else:
        frame_auc, frame_f1 = float("nan"), float("nan")
        print("  Frame AUC = N/A (no positive frames in test set)")

    # ── Segment AP ────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  3. SEGMENT-LEVEL AP  (AUXILIARY DIAGNOSTIC ONLY)")
    print(f"     NOTE: TALVAD is a detector, not a dedicated localiser. These")
    print(f"     numbers are reported for transparency, NOT as a localisation claim.")
    print(f"{'='*64}")
    print(f"  GT segments  : {len(all_gts_seg):,}")
    print(f"  Pred segments: {len(all_preds_seg):,}")

    # Full TAD protocol (matches UMMAFormer / AuViRe / AV-Deepfake1M)
    full_loc = compute_full_localization_metrics(all_preds_seg, all_gts_seg)
    for iou_thr in [0.1, 0.3, 0.5, 0.75, 0.9, 0.95]:
        ap = compute_segment_ap(all_preds_seg, all_gts_seg, iou_threshold=iou_thr)
        print(f"  AP @ IoU={iou_thr:.2f}: {ap:.4f}")
    print(f"  mAP (0.5,0.75,0.9,0.95): {full_loc['mAP']:.4f}")
    for n in [100, 50, 30, 20, 10, 5]:
        print(f"  AR @ {n:>3d} proposals : {full_loc[f'AR@{n}']:.4f}")

    # Legacy seg_ap dict kept for JSON output compat
    seg_ap = {f"ap_{int(t*10):02d}": compute_segment_ap(all_preds_seg, all_gts_seg, t)
              for t in [0.1, 0.3, 0.5]}
    seg_ap.update({f"AP@{int(t*100)}": full_loc[f"AP@{int(t*100)}"] for t in [0.5, 0.75, 0.9, 0.95]})
    seg_ap["mAP_sota"] = full_loc["mAP"]
    mean_ap = float(np.mean([seg_ap[k] for k in ["ap_01", "ap_03", "ap_05"]]))

    # ── Localisation, threshold-tuned + per signal source ─────────────────────
    # The fixed-0.5 decode above collapses to ~0 segments on a low-positive-rate
    # head, and it only ever scored TFLH. Here we (1) sweep the decode threshold
    # so the achievable operating point is visible, and (2) score the CMIS
    # inconsistency map as a localiser in its own right — the contribution the
    # legacy report computed every forward pass and then discarded.
    print(f"\n{'='*64}")
    print(f"  3b. LOCALISATION — threshold-tuned, by signal source (diagnostic)")
    print(f"      (decode threshold tuned on eval set -> upper bound, not held-out)")
    print(f"{'='*64}")
    per_sample_stems = [m["stem"] for m in sample_metas]

    tflh_loc = localization_from_scores(
        frame_scores_all, per_sample_stems, frame_labels_all, all_gts_seg, fps
    )
    print(f"  [TFLH frame head]  frame_AUC={tflh_loc['frame_auc']}  "
          f"best_F1={tflh_loc['best_frame_f1']:.4f}@{tflh_loc['best_frame_f1_threshold']:.2f}  "
          f"best_mAP={tflh_loc['best_map']:.4f}@{tflh_loc['best_map_threshold']:.2f}  "
          f"(npred={tflh_loc['best_map_n_pred_segments']})")

    cmis_loc = None
    if cmis_scores_all and all(c is not None for c in cmis_scores_all):
        cmis_loc = localization_from_scores(
            cmis_scores_all, per_sample_stems, frame_labels_all, all_gts_seg, fps
        )
        print(f"  [CMIS incon map ]  frame_AUC={cmis_loc['frame_auc']}  "
              f"best_F1={cmis_loc['best_frame_f1']:.4f}@{cmis_loc['best_frame_f1_threshold']:.2f}  "
              f"best_mAP={cmis_loc['best_map']:.4f}@{cmis_loc['best_map_threshold']:.2f}  "
              f"(npred={cmis_loc['best_map_n_pred_segments']})")
    else:
        print(f"  [CMIS incon map ]  not available (CMIS disabled in this checkpoint)")

    # ── Subtype breakdown ─────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  4. SUBTYPE BREAKDOWN")
    print(f"{'='*64}")

    subtypes = ["real", "audio_only", "video_only", "video+audio"]
    sub_results = {}
    for st in subtypes:
        st_meta = [m for m in sample_metas if m["fake_type"] == st]
        if not st_meta:
            continue
        st_y    = np.array([m["label"] for m in st_meta])
        st_sc   = np.array([m["video_score"] for m in st_meta])
        st_pred = (st_sc >= 0.5).astype(int)
        if st == "real":
            tn = int((st_pred == 0).sum())
            fp = int((st_pred == 1).sum())
            print(f"  real        ({len(st_meta):5d}) : TNR={tn/len(st_meta):.3f}  FPR={fp/len(st_meta):.3f}")
            sub_results[st] = {"n": len(st_meta), "tnr": tn/len(st_meta), "fpr": fp/len(st_meta)}
        else:
            # AUC requires both classes; subtype subsets are all-fake → undefined.
            # Report N/A rather than a fabricated 1.0 (previous synthetic-sample hack).
            st_auc = roc_auc_score(st_y, st_sc) if len(np.unique(st_y)) > 1 else None
            tp = int((st_pred == 1).sum())
            fn = int((st_pred == 0).sum())
            tpr = tp / len(st_meta) if st_meta else 0
            st_ap_01 = compute_segment_ap(
                [p for p in all_preds_seg
                 if any(p["video_id"] == m["stem"] for m in st_meta)],
                [g for g in all_gts_seg
                 if any(g["video_id"] == m["stem"] for m in st_meta)],
                iou_threshold=0.1,
            )
            auc_str = f"{st_auc:.3f}" if st_auc is not None else "N/A "
            print(f"  {st:<15}({len(st_meta):5d}) : AUC={auc_str}  TPR={tpr:.3f}  AP@0.1={st_ap_01:.3f}")
            sub_results[st] = {
                "n": len(st_meta), "auc": (float(st_auc) if st_auc is not None else None),
                "tpr": float(tpr), "ap_at_01": float(st_ap_01),
            }

    # ── MAA modality attribution (which modality was forged?) ─────────────────
    maa_results = None
    if any(m.get("audio_fake_score") is not None for m in sample_metas):
        print(f"\n{'='*64}")
        print(f"  5. MODALITY ATTRIBUTION  (MAA heads — which modality was forged?)")
        print(f"{'='*64}")
        _ma = {"real": 0, "audio_only": 1, "video_only": 0, "video+audio": 1}
        _mv = {"real": 0, "audio_only": 0, "video_only": 1, "video+audio": 1}
        rows = [m for m in sample_metas
                if m.get("audio_fake_score") is not None and m["fake_type"] in _ma]
        a_sc  = np.array([m["audio_fake_score"] for m in rows])
        v_sc  = np.array([m["video_fake_score"] for m in rows])
        ma_gt = np.array([_ma[m["fake_type"]] for m in rows])
        mv_gt = np.array([_mv[m["fake_type"]] for m in rows])
        audio_auc = float(roc_auc_score(ma_gt, a_sc)) if len(np.unique(ma_gt)) > 1 else None
        video_auc = float(roc_auc_score(mv_gt, v_sc)) if len(np.unique(mv_gt)) > 1 else None
        print(f"  Audio-head AUC (modify_audio): {round(audio_auc,4) if audio_auc is not None else 'N/A'}")
        print(f"  Video-head AUC (modify_video): {round(video_auc,4) if video_auc is not None else 'N/A'}")

        classes = ["audio_only", "video_only", "video+audio"]
        def _pred_label(m):
            a = m["audio_fake_score"] >= 0.5
            v = m["video_fake_score"] >= 0.5
            if a and v: return "video+audio"
            if a:       return "audio_only"
            if v:       return "video_only"
            return "none"
        fakes = [m for m in rows if m["fake_type"] != "real"]
        confusion = {gt: {p: 0 for p in classes + ["none"]} for gt in classes}
        correct = 0
        for m in fakes:
            p = _pred_label(m); confusion[m["fake_type"]][p] += 1
            correct += int(p == m["fake_type"])
        attr_acc = correct / max(len(fakes), 1)
        print(f"  3-way attribution accuracy (fakes only): {attr_acc:.4f}  (n={len(fakes)})")
        for gt in classes:
            tot = sum(confusion[gt].values())
            if tot:
                print(f"    GT {gt:<12} -> " + "  ".join(f"{p}:{confusion[gt][p]}" for p in classes + ["none"]))
        maa_results = {
            "audio_head_auc": audio_auc, "video_head_auc": video_auc,
            "attribution_accuracy": float(attr_acc), "n_fakes": len(fakes),
            "confusion": confusion,
        }

    # ── Save JSON report ──────────────────────────────────────────────────────
    report = {
        "experiment":        cfg["experiment_name"],
        "config":            config_path,
        "n_evaluated":       n_evaluated,
        "missing_features":  missing_feat,
        "gfc_zero_padded":   gfc_zero_count,
        "frame_threshold":   frame_threshold,
        "fixed_threshold":       fixed_thresh,
        "tuned_threshold":       tuned_thresh,
        "video_metrics_fixed":   vid_metrics_05,
        "video_metrics_tuned":   vid_metrics_tuned,
        "frame_auc":         frame_auc if not np.isnan(frame_auc) else None,
        "frame_f1":          frame_f1  if not np.isnan(frame_f1)  else None,
        "segment_ap":        seg_ap,
        "mean_ap":           mean_ap,
        "n_gt_segments":     len(all_gts_seg),
        "n_pred_segments":   len(all_preds_seg),
        # Threshold-tuned localisation, scored separately for the TFLH head and
        # the CMIS inconsistency map. This is the metric that actually tests
        # CMIS's localisation claim (legacy mean_ap above scores TFLH @ fixed 0.5).
        "localization_tuned": {"tflh": tflh_loc, "cmis": cmis_loc},
        "subtype_breakdown": sub_results,
        "maa_attribution":   maa_results,
        "split": "official LAV-DF train/dev/test" if cfg["dataset"].get("use_official_split") else "custom random 70/15/15",
        "known_limitations": [
            l for l in [
                "Test set GFC was zero-padded (not cached) for evaluation" if gfc_zero_count else None,
                None if cfg["dataset"].get("use_official_split")
                else "Random 70/15/15 split — not comparable to official LAV-DF train/dev/test",
            ] if l
        ],
    }

    report_path = out_path / "eval_temporal_results.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Report saved -> {report_path}")

    # ── Save per-sample results ───────────────────────────────────────────────
    samples_path = out_path / "eval_temporal_samples.json"
    # Keep only summary fields for readability
    summary_metas = [
        {k: v for k, v in m.items() if k not in ("frame_scores", "frame_gt", "cmis_incon")}
        for m in sample_metas
    ]
    samples_path.write_text(json.dumps(summary_metas, indent=2), encoding="utf-8")
    print(f"  Per-sample  -> {samples_path}")

    # ── Timeline visualisations ───────────────────────────────────────────────
    _save_timeline_plots(sample_metas, out_path, frame_threshold, fps, seq_len)

    print(f"\n{'='*64}")
    print(f"  SUMMARY")
    print(f"  Video AUC  : {vid_metrics_05['auc']:.4f}  (HEADLINE — detection, threshold-free)")
    if maa_results is not None:
        print(f"  MAA attrib : {maa_results['attribution_accuracy']:.4f}  (modality attribution accuracy, fakes)")
    print(f"  Frame AUC  : {frame_auc:.4f}  (auxiliary diagnostic — not a localisation claim)")
    print(f"  Seg AP@0.5 : {seg_ap.get('ap_05', 0):.4f}  (auxiliary diagnostic only)")
    print(f"  mAP        : {mean_ap:.4f}  (auxiliary diagnostic only)")
    print(f"{'='*64}\n")

    return report


# ── Timeline plots ────────────────────────────────────────────────────────────

def _save_timeline_plots(
    sample_metas: List[dict],
    out_path: Path,
    frame_threshold: float,
    fps: float,
    seq_len: int,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    # Pick 2 from each category: real, audio_only, video_only, video+audio
    chosen = []
    for st in ["real", "audio_only", "video_only", "video+audio"]:
        pool = [m for m in sample_metas if m["fake_type"] == st]
        chosen.extend(pool[:2])
    chosen = chosen[:8]

    if not chosen:
        return

    fig, axes = plt.subplots(len(chosen), 1, figsize=(12, 2.5 * len(chosen)))
    if len(chosen) == 1:
        axes = [axes]

    time_axis = np.arange(seq_len) / fps   # seconds

    for ax, meta in zip(axes, chosen):
        fs = np.array(meta["frame_scores"])
        fg = np.array(meta["frame_gt"])
        stem = meta["stem"]
        ftype = meta["fake_type"]
        vscore = meta["video_score"]

        ax.fill_between(time_axis, fg, alpha=0.25, color="red", label="GT fake")
        ax.plot(time_axis, fs, color="#3498db", lw=1.8, label="Frame score")
        # CMIS inconsistency map — should spike inside the GT fake region
        cmis = meta.get("cmis_incon")
        if cmis is not None:
            ax.plot(time_axis, np.array(cmis), color="#9b59b6", lw=1.6, ls="-",
                    alpha=0.9, label="CMIS inconsistency")
        ax.axhline(frame_threshold, color="orange", lw=1, ls="--", label=f"Threshold {frame_threshold}")
        ax.axhline(vscore, color="#2ecc71", lw=1, ls=":", label=f"Video score {vscore:.3f}")

        ax.set_xlim(0, seq_len / fps)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Score", fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_title(f"{stem}  [{ftype}]  video_score={vscore:.3f}", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = out_path / "eval_temporal_timelines.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Timeline plot -> {plot_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TALVAD temporal localisation evaluation")
    parser.add_argument("--config", default="configs/lavdf_talvad_v5.yaml",
                        help="Path to experiment config YAML")
    parser.add_argument("--frame_threshold", type=float, default=0.5,
                        help="Sigmoid threshold for frame-level binary prediction (default: 0.5)")
    parser.add_argument("--compute_gfc", action="store_true",
                        help="Compute GFC on demand from raw videos for samples missing cache")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Override output directory for results JSON")
    args = parser.parse_args()

    run_eval(
        config_path=args.config,
        frame_threshold=args.frame_threshold,
        compute_gfc=args.compute_gfc,
        out_dir=args.out_dir,
    )
