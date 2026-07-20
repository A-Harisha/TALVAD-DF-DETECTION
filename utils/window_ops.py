"""
utils/window_ops.py — fixed-fps SLIDING-WINDOW geometry for TALVAD-L on
AV-Deepfake1M (DF1M).

LAV-DF used a duration-normalized seq_len grid (utils/loc_ops.py): every clip is
squashed to `seq_len` steps. That destroys sub-second, word-level boundaries on
DF1M's long, variable-length videos. Here the time axis is instead a FIXED frame
rate (fps): step i == frame i, seconds = i / fps. A clip of `duration` seconds has
`round(duration*fps)` frames; we train on fixed `win`-frame windows cropped from it
(sliding, overlapping) and, at inference, slide the window across the whole clip and
STITCH the per-window detections back together in absolute seconds.

This module owns ONLY the window<->clip transforms. Per-window label assignment,
anchor-free decode, and Soft-NMS are REUSED verbatim from utils.loc_ops, so the DF1M
path lives on the same geometry the LAV-DF localizer was validated on — the window is
just a moving [0, win) frame coordinate frame. The self-test (`python -m
utils.window_ops`) round-trips GT through window cropping + decode + stitch and asserts
recovery even for segments that straddle a window boundary (the new failure mode).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from utils import loc_ops


# ── fixed-fps frame <-> seconds ───────────────────────────────────────────────

def n_frames_of(duration: float, fps: float) -> int:
    return max(1, int(round(float(duration) * float(fps))))


def sec_to_frame(t: float, fps: float) -> float:
    return float(t) * float(fps)


def frame_to_sec(f: float, fps: float) -> float:
    return float(f) / float(fps)


# ── sliding windows over one clip ─────────────────────────────────────────────

def make_windows(n_frames: int, win: int, hop: int) -> List[int]:
    """Start frames of overlapping `win`-length windows tiling [0, n_frames).

    The final window is clamped so it ends exactly at n_frames (start =
    n_frames - win), guaranteeing full coverage with no out-of-range tail. Clips
    shorter than `win` yield a single window starting at 0 (the dataset/model pad).
    """
    if n_frames <= win:
        return [0]
    starts = list(range(0, n_frames - win + 1, hop))
    if starts[-1] != n_frames - win:
        starts.append(n_frames - win)        # clamp last window to the clip end
    return starts


def segments_to_window(
    gt_sec: Sequence[Tuple[float, float]],
    win_start_frame: int,
    win: int,
    fps: float,
    min_overlap_frames: float = 0.5,
) -> List[Tuple[float, float]]:
    """GT spans (seconds) -> window-local frame spans, clipped to [0, win].

    A span is kept for this window only if its overlap with the window is at least
    `min_overlap_frames` frames (so a span barely clipped by a boundary in THIS
    window is dropped here — an overlapping window contains it whole). Returned
    spans are in window-local level-0 frame units, ready for loc_ops.assign_targets
    with T0 = win.
    """
    out: List[Tuple[float, float]] = []
    w0, w1 = float(win_start_frame), float(win_start_frame + win)
    for (s, e) in gt_sec:
        sf, ef = sec_to_frame(s, fps), sec_to_frame(e, fps)
        cs, ce = max(sf, w0), min(ef, w1)            # clip to window (absolute frames)
        if (ce - cs) >= min_overlap_frames:
            out.append((cs - w0, ce - w0))           # -> window-local frames
    return out


def assign_window_targets(
    gt_sec: Sequence[Tuple[float, float]],
    win_start_frame: int,
    win: int,
    fps: float,
    strides: Sequence[int],
    ranges: Sequence[Tuple[float, float]],
    center_radius: float = 1.5,
    cbm_sigma: float = 1.0,
    min_overlap_frames: float = 0.5,
) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    """Per-window (levels, cbm_target[win], frame_label[win]) — all via loc_ops on
    the window-local frame axis. Returns whatever GT falls inside this window."""
    local = segments_to_window(gt_sec, win_start_frame, win, fps, min_overlap_frames)
    levels = loc_ops.assign_targets(local, win, strides, ranges, center_radius)
    cbm = loc_ops.gaussian_boundary_target(local, win, cbm_sigma)
    frame = np.zeros(win, dtype=np.float32)
    for (sf, ef) in local:
        a, b = int(np.floor(sf)), int(np.ceil(ef))
        frame[max(0, a):max(0, min(win, b))] = 1.0
    return levels, cbm, frame


# ── inference: decode one window -> ABSOLUTE seconds, then stitch ─────────────

def window_decode_to_sec(
    level_scores: Sequence[np.ndarray],       # per level [Tl] in (0,1)
    level_reg: Sequence[np.ndarray],          # per level [Tl,2] window-local frames
    level_centers_list: Sequence[np.ndarray],
    win_start_frame: int,
    fps: float,
    score_thr: float = 0.0,
) -> List[Tuple[float, float, float]]:
    """Decode a single window's pyramid outputs to (t_start_sec, t_end_sec, score)
    in ABSOLUTE clip time. Window-local frame -> absolute frame (+win_start) -> sec."""
    fsegs = loc_ops.decode_frames(level_scores, level_reg, level_centers_list, score_thr)
    out: List[Tuple[float, float, float]] = []
    for sf, ef, sc in fsegs:
        s = frame_to_sec(sf + win_start_frame, fps)
        e = frame_to_sec(ef + win_start_frame, fps)
        if e > s:
            out.append((s, e, sc))
    return out


def stitch(
    segs_sec: List[Tuple[float, float, float]],
    sigma: float = 0.5,
    score_thr: float = 0.001,
    max_keep: int = 100,
) -> List[Tuple[float, float, float]]:
    """Merge detections pooled from all windows of a clip (absolute seconds) into a
    final set via global Soft-NMS. Overlap regions produce duplicate boxes for the
    same event; Soft-NMS collapses them. (WBF available via loc_ops.weighted_fusion.)"""
    return loc_ops.soft_nms(segs_sec, sigma=sigma, score_thr=score_thr, max_keep=max_keep)


# ── round-trip self-test (Risk #1 guardrail, incl. boundary-straddling) ───────

def _selftest() -> None:
    import sys

    fps = 25.0
    win, hop = 96, 48                       # 3.84 s windows, 50% overlap
    strides = [1, 2, 4, 8]
    ranges = [(0, 8), (8, 16), (16, 32), (32, 1e9)]

    # A long, variable-length clip with several short word-level fakes, one placed
    # DELIBERATELY on a window boundary (~win/fps = 3.84 s and ~2*hop/fps).
    duration = 12.3                          # 307 frames
    gt_sec = [
        (1.10, 1.42),                        # well inside window 0
        (3.80, 4.05),                        # straddles the win0/win1 boundary (96 fr)
        (3.84 * 2 - 0.1, 3.84 * 2 + 0.2),    # near win1/win2 boundary
        (9.50, 9.74),                        # late, only in the clamped last window
        (6.40, 6.55),                        # short (0.15 s ~ 3.75 frames)
    ]
    n_frames = n_frames_of(duration, fps)
    starts = make_windows(n_frames, win, hop)
    print(f"  clip {duration}s -> {n_frames} frames | windows={len(starts)} starts={starts}")

    # Build per-window GT targets, "decode" them perfectly (score=1 at positives),
    # map back to absolute seconds, stitch, then check every GT span is recovered.
    pooled: List[Tuple[float, float, float]] = []
    for ws in starts:
        levels, _, _ = assign_window_targets(gt_sec, ws, win, fps, strides, ranges)
        scores = [lv["cls"] for lv in levels]
        regs = [lv["reg"] for lv in levels]
        ctrs = [lv["centers"] for lv in levels]
        pooled += window_decode_to_sec(scores, regs, ctrs, ws, fps, score_thr=0.5)
    final = stitch(pooled, sigma=0.5)
    print(f"  pooled {len(pooled)} window dets -> {len(final)} stitched")

    ious = []
    for (gs, ge) in gt_sec:
        best = max((loc_ops.iou_1d(gs, ge, ds, de) for ds, de, _ in final), default=0.0)
        ious.append(best)
        print(f"   GT [{gs:.2f},{ge:.2f}] (len {(ge-gs)*fps:.1f} fr)  best IoU={best:.4f}")
    ious = np.array(ious)
    ok = ious.min() > 0.98
    print(f"  min round-trip IoU = {ious.min():.4f}  mean = {ious.mean():.4f}  -> {'OK' if ok else 'FAIL'}")
    if not ok:
        print("  FAIL: a GT span (incl. boundary-straddling) did not round-trip — windowing geometry is wrong.")
        sys.exit(1)
    print("WINDOW_OPS ROUND-TRIP OK")


if __name__ == "__main__":
    _selftest()
