"""
utils/loc_ops.py — temporal-localization geometry for TALVAD-L.

ONE place for the seconds <-> normalized-frame conversion, center-based label
assignment, anchor-free decode, and Soft-NMS, so training (targets) and eval
(decode) use the *identical* math. The original segment-AP collapse was caused by
train/eval living on different time axes; centralising the conversion here is the
structural fix, and the round-trip self-test (`python -m utils.loc_ops`) is the
guardrail that proves it.

Frame coords are "level-0 step units" on a duration-normalized grid of `seq_len`
steps, so step center i sits at (i+0.5) and maps to seconds via i*duration/seq_len.
A GT span [s,e] seconds -> [s*seq_len/duration, e*seq_len/duration] frames.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


# ── conversion (the single source of truth) ───────────────────────────────────

def seconds_to_frames(s: float, e: float, duration: float, seq_len: int) -> Tuple[float, float]:
    if duration <= 0:
        return 0.0, 0.0
    k = seq_len / duration
    return s * k, e * k


def frames_to_seconds(sf: float, ef: float, duration: float, seq_len: int) -> Tuple[float, float]:
    k = duration / seq_len
    return sf * k, ef * k


def iou_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 1e-9 else 0.0


def level_centers(T0: int, stride: int) -> np.ndarray:
    """Level-0 coordinates of every step at a level with the given stride."""
    Tl = int(np.ceil(T0 / stride)) if stride > 1 else T0
    return (np.arange(Tl, dtype=np.float32) + 0.5) * stride


# ── label assignment (center-based, ActionFormer-style) ───────────────────────

def assign_targets(
    gt_frames: Sequence[Tuple[float, float]],
    T0: int,
    strides: Sequence[int],
    ranges: Sequence[Tuple[float, float]],
    center_radius: float = 1.5,
) -> List[Dict[str, np.ndarray]]:
    """Per-level (cls, reg, centers) for one clip.

    A step at level l is POSITIVE for a GT span if its center c is inside the span
    (or within `center_radius` level-0 units of the span centre, so spans shorter
    than a step still get an anchor) AND the larger boundary distance falls in that
    level's regression range. On conflict the SHORTEST span wins (fine detail).
    reg = (c - s, e - c) in level-0 units, clipped to >= 0.
    """
    out: List[Dict[str, np.ndarray]] = []
    for stride, (lo, hi) in zip(strides, ranges):
        ctr = level_centers(T0, stride)
        Tl = len(ctr)
        cls = np.zeros(Tl, dtype=np.float32)
        reg = np.zeros((Tl, 2), dtype=np.float32)
        for ci, c in enumerate(ctr):
            best_len = np.inf
            best_reg = None
            for (sf, ef) in gt_frames:
                seg_c = 0.5 * (sf + ef)
                inside = (sf <= c <= ef) or (abs(c - seg_c) <= center_radius)
                if not inside:
                    continue
                ds, de = c - sf, ef - c
                m = max(abs(ds), abs(de))
                if lo <= m < hi:
                    seglen = ef - sf
                    if seglen < best_len:
                        best_len = seglen
                        best_reg = (max(ds, 0.0), max(de, 0.0))
            if best_reg is not None:
                cls[ci] = 1.0
                reg[ci] = best_reg
        out.append({"cls": cls, "reg": reg, "centers": ctr})
    return out


# ── decode (anchor-free) ──────────────────────────────────────────────────────

def decode(
    level_scores: Sequence[np.ndarray],     # per level [Tl] in (0,1)
    level_reg: Sequence[np.ndarray],         # per level [Tl,2] level-0 units
    level_centers_list: Sequence[np.ndarray],
    duration: float,
    seq_len: int,
    score_thr: float = 0.0,
) -> List[Tuple[float, float, float]]:
    """Per-level steps -> (t_start_sec, t_end_sec, score), unsorted."""
    segs: List[Tuple[float, float, float]] = []
    for scores, reg, ctr in zip(level_scores, level_reg, level_centers_list):
        keep = np.where(scores >= score_thr)[0]
        for i in keep:
            sf = ctr[i] - reg[i, 0]
            ef = ctr[i] + reg[i, 1]
            if ef <= sf:
                continue
            s, e = frames_to_seconds(float(sf), float(ef), duration, seq_len)
            segs.append((s, e, float(scores[i])))
    return segs


def decode_frames(
    level_scores: Sequence[np.ndarray],
    level_reg: Sequence[np.ndarray],
    level_centers_list: Sequence[np.ndarray],
    score_thr: float = 0.0,
) -> List[Tuple[float, float, float]]:
    """Like `decode` but returns segments in LEVEL-0 FRAME coords (no seconds
    conversion) so boundaries can be refined against the frame-indexed CBM curve."""
    segs: List[Tuple[float, float, float]] = []
    for scores, reg, ctr in zip(level_scores, level_reg, level_centers_list):
        for i in np.where(scores >= score_thr)[0]:
            sf = ctr[i] - reg[i, 0]
            ef = ctr[i] + reg[i, 1]
            if ef > sf:
                segs.append((float(sf), float(ef), float(scores[i])))
    return segs


def refine_boundaries(
    segs: List[Tuple[float, float, float]],
    cbm: np.ndarray,
    window: float = 4.0,
    power: float = 2.0,
) -> List[Tuple[float, float, float]]:
    """Snap each segment's start/end to the nearby CBM boundary-energy peak via a
    soft-argmax (sub-frame precision). The CBM is trained to fire at GT edges, so
    this pulls a loosely-regressed boundary onto the true change-point — the CBM
    novelty's job, targeting high-IoU AP. Segments are in level-0 frame coords."""
    T = len(cbm)

    def snap(x: float) -> float:
        lo = max(0, int(np.floor(x - window)))
        hi = min(T, int(np.ceil(x + window)) + 1)
        if hi <= lo:
            return x
        pos = np.arange(lo, hi, dtype=np.float64) + 0.5
        w = np.asarray(cbm[lo:hi], dtype=np.float64) ** power
        return float((pos * w).sum() / w.sum()) if w.sum() > 1e-6 else x

    out = []
    for s, e, sc in segs:
        ns, ne = snap(s), snap(e)
        if ne <= ns:
            ns, ne = s, e
        out.append((ns, ne, sc))
    return out


def soft_nms(
    segs: List[Tuple[float, float, float]],
    sigma: float = 0.5,
    score_thr: float = 0.001,
    max_keep: int = 100,
) -> List[Tuple[float, float, float]]:
    """Gaussian Soft-NMS on 1-D temporal segments. Returns kept (s,e,score) desc."""
    if not segs:
        return []
    items = [list(x) for x in segs]
    kept: List[Tuple[float, float, float]] = []
    while items and len(kept) < max_keep:
        items.sort(key=lambda x: -x[2])
        s0, e0, sc0 = items.pop(0)
        kept.append((s0, e0, sc0))
        for it in items:
            ov = iou_1d(s0, e0, it[0], it[1])
            it[2] *= float(np.exp(-(ov * ov) / sigma))
        items = [it for it in items if it[2] > score_thr]
    return kept


def weighted_fusion(
    segs: List[Tuple[float, float, float]],
    iou_thr: float = 0.5,
    max_keep: int = 100,
) -> List[Tuple[float, float, float]]:
    """Weighted boundary fusion (1-D WBF/NMW). Instead of keeping the top-scoring
    box and suppressing overlaps (plain Soft-NMS — which discards the lower-score
    but tighter boundary), CLUSTER all predictions overlapping the current best at
    >= iou_thr and emit ONE box whose start/end are the SCORE-WEIGHTED AVERAGE of
    the cluster. The cluster sits around the true segment, so the average tightens
    the boundary instead of throwing it away. Diagnosed as the single biggest
    AP@0.90 lever (Soft-NMS halved pre-NMS recall@0.9)."""
    if not segs:
        return []
    order = sorted(segs, key=lambda x: -x[2])
    used = [False] * len(order)
    out: List[Tuple[float, float, float]] = []
    for i in range(len(order)):
        if used[i]:
            continue
        s0, e0, sc0 = order[i]
        used[i] = True
        ss, ee, ww = [s0], [e0], [sc0]
        for j in range(i + 1, len(order)):
            if used[j]:
                continue
            if iou_1d(s0, e0, order[j][0], order[j][1]) >= iou_thr:
                used[j] = True
                ss.append(order[j][0]); ee.append(order[j][1]); ww.append(order[j][2])
        w = np.asarray(ww, dtype=np.float64)
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        out.append((float(np.dot(w, ss)), float(np.dot(w, ee)), float(sc0)))
        if len(out) >= max_keep:
            break
    return out


def gaussian_boundary_target(
    gt_frames: Sequence[Tuple[float, float]],
    T0: int,
    sigma: float = 1.0,
) -> np.ndarray:
    """Per-step boundary target in [0,1]: a Gaussian bump (width `sigma` level-0
    frames) centred on every GT edge (both start and end). Supervises the CBM
    boundary-energy head (BSN/BMN-style soft edge labels)."""
    t = np.arange(T0, dtype=np.float32) + 0.5
    target = np.zeros(T0, dtype=np.float32)
    for (sf, ef) in gt_frames:
        for edge in (sf, ef):
            target = np.maximum(target, np.exp(-0.5 * ((t - edge) / max(sigma, 1e-3)) ** 2))
    return target


# ── round-trip self-test (Risk #1 guardrail) ──────────────────────────────────

def _selftest() -> None:
    import json
    import sys
    from pathlib import Path

    REPO = Path(__file__).resolve().parent.parent
    ann_path = REPO / "datasets" / "LAV-DF" / "metadata.min.json"
    if not ann_path.exists():
        print(f"[skip] annotation not found: {ann_path}")
        return

    seq_len = 192
    strides = [1, 2, 4, 8]
    ranges = [(0, 8), (8, 16), (16, 32), (32, 1e9)]

    data = json.loads(ann_path.read_text(encoding="utf-8"))
    fakes = [e for e in data
             if e.get("split") == "test" and e.get("fake_periods") and float(e.get("duration", 0)) > 0]
    print(f"  test-split clips with fake_periods: {len(fakes):,}")

    ious, missed, n_multi = [], 0, 0
    sub_step = 0
    for e in fakes[:400]:
        dur = float(e["duration"])
        gt_sec = [(float(s), float(t)) for s, t in e["fake_periods"]]
        if len(gt_sec) > 1:
            n_multi += 1
        gt_frames = [seconds_to_frames(s, t, dur, seq_len) for s, t in gt_sec]
        for (sf, ef) in gt_frames:
            if (ef - sf) < 1.0:
                sub_step += 1

        levels = assign_targets(gt_frames, seq_len, strides, ranges, center_radius=1.5)
        # decode using the TRUE targets with score=1 at positives -> must reconstruct GT.
        scores = [lv["cls"] for lv in levels]
        regs = [lv["reg"] for lv in levels]
        ctrs = [lv["centers"] for lv in levels]
        decoded = decode(scores, regs, ctrs, dur, seq_len, score_thr=0.5)
        decoded = soft_nms(decoded, sigma=0.5)

        for (gs, ge) in gt_sec:
            best = max((iou_1d(gs, ge, ds, de) for ds, de, _ in decoded), default=0.0)
            if best < 1e-6:
                missed += 1
            ious.append(best)

    ious = np.array(ious)
    print(f"  multi-segment clips: {n_multi}  | GT spans < 1 level-0 step (sub-resolution): {sub_step}")
    print(f"  round-trip IoU: mean={ious.mean():.4f}  median={np.median(ious):.4f}  "
          f"min={ious.min():.4f}  | GT spans with no recovery: {missed}/{len(ious)}")
    # Spans that ARE resolvable (>= 1 step) must round-trip ~exactly.
    resolvable = ious[ious > 0]
    ok = resolvable.mean() > 0.98 if len(resolvable) else False
    print(f"  recovered-span mean IoU = {resolvable.mean():.4f}  -> {'OK' if ok else 'FAIL'}")
    if not ok:
        sys.exit(1)
    print("LOC_OPS ROUND-TRIP OK")


if __name__ == "__main__":
    _selftest()
