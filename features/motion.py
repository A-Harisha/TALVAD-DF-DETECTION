"""
features/motion.py — mouth-region optical-flow MOTION descriptor (TALVAD-X v2).

The transferable deepfake cue is articulation: how the mouth MOVES vs what the audio
says — not the speaker's appearance/identity (which the EfficientNet features encode)
nor the generator's frequency fingerprint (which inverts cross-dataset). This computes
a compact per-frame motion descriptor from dense optical flow on the mouth ROI of the
(already face-cropped) frames, to be concatenated onto the existing feature pack.

Per frame t (flow from t-1 -> t; t=0 is zeros):
  [ mean|mag|, std|mag|, p90|mag|, mean v_x, std v_x, mean v_y, std v_y,
    |mean v_y| (mouth open/close energy), 8-bin flow-direction histogram (mag-weighted) ]
= 16 dims.  Deterministic, CPU, ~Farneback.
"""
from __future__ import annotations

import numpy as np

MOTION_DIM = 16


def mouth_roi(gray: np.ndarray) -> np.ndarray:
    """Lower-center region of a square face crop ≈ the mouth."""
    h, w = gray.shape[:2]
    y0, y1 = int(0.58 * h), int(0.95 * h)
    x0, x1 = int(0.25 * w), int(0.75 * w)
    return gray[y0:y1, x0:x1]


def _frame_descriptor(flow: np.ndarray) -> np.ndarray:
    import cv2
    vx, vy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(vx, vy)            # ang in [0, 2π)
    m = mag.ravel()
    desc = [
        float(m.mean()), float(m.std()), float(np.percentile(m, 90)),
        float(vx.mean()), float(vx.std()),
        float(vy.mean()), float(vy.std()), float(abs(vy.mean())),
    ]
    # magnitude-weighted 8-bin direction histogram (normalized)
    hist, _ = np.histogram(ang.ravel(), bins=8, range=(0.0, 2 * np.pi), weights=m)
    s = hist.sum()
    hist = hist / s if s > 1e-8 else hist
    desc.extend(hist.tolist())
    return np.asarray(desc, dtype=np.float32)      # [16]


def compute_motion(crops, roi: bool = True) -> np.ndarray:
    """crops: [T,H,W,3] RGB (face crops). Returns [T, MOTION_DIM] float32.

    Frame 0 is zeros (no previous frame). Robust to tiny ROIs / decode glitches."""
    import cv2
    crops = np.asarray(crops)
    T = len(crops)
    out = np.zeros((T, MOTION_DIM), dtype=np.float32)
    if T < 2:
        return out
    grays = []
    for c in crops:
        g = cv2.cvtColor(c.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        grays.append(mouth_roi(g) if roi else g)
    for t in range(1, T):
        prev, cur = grays[t - 1], grays[t]
        if prev.shape != cur.shape or prev.size == 0:
            continue
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev, cur, None, pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            out[t] = _frame_descriptor(flow)
        except cv2.error:
            continue
    return out


if __name__ == "__main__":
    # smoke: synthetic moving block -> nonzero motion; static -> ~zero
    rng = np.random.default_rng(0)
    T = 48
    static = np.tile(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)[None], (T, 1, 1, 1))
    d_static = compute_motion(static)
    moving = static.copy()
    for t in range(T):  # shift mouth region content each frame
        moving[t, 140:200, 70:150] = np.roll(moving[t, 140:200, 70:150], t * 2, axis=0)
    d_move = compute_motion(moving)
    print(f"[OK] motion desc shape={d_move.shape} dim={MOTION_DIM}")
    print(f"  static mean|mag| = {d_static[:,0].mean():.4f}  (~0)")
    print(f"  moving mean|mag| = {d_move[:,0].mean():.4f}  (> static)")
    assert d_move.shape == (T, MOTION_DIM)
    assert d_move[:, 0].mean() >= d_static[:, 0].mean()
    print("MOTION SMOKE OK")
