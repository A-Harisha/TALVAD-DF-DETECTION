"""
Global Frequency Consistency (GFC) feature extractor.

Deepfake synthesis introduces high-frequency artifacts at synthesis boundaries,
GAN grid patterns, and codec inconsistencies that spatial CNNs (ResNet18) trained
on ImageNet cannot reliably detect — ImageNet training suppresses high-freq signal.

For each video frame we:
  1. Convert to grayscale
  2. Apply 2-D FFT and shift DC to centre
  3. Compute log-energy in N annular frequency bands (low → high)
  4. Return a compact band-energy vector per frame

The TDFS layer then learns which frequency bands are temporally discriminative,
making this the key PhD contribution: frequency-aware temporal feature selection.
"""
from __future__ import annotations

import numpy as np


def extract_gfc_features(
    frames: list,
    n_bands: int = 8,
) -> np.ndarray:
    """
    Args:
        frames: list of H×W×3 uint8 or float32 arrays (one per video frame).
        n_bands: number of concentric frequency bands to measure energy in.

    Returns:
        Array of shape [T, n_bands] — one row per frame.
    """
    out = []
    for frame in frames:
        frame = np.asarray(frame, dtype=np.float32)

        # ── Grayscale conversion ─────────────────────────────────────────
        if frame.ndim == 3 and frame.shape[2] >= 3:
            gray = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
        elif frame.ndim == 3:
            gray = frame[..., 0]
        else:
            gray = frame

        H, W = gray.shape
        cy, cx = H // 2, W // 2
        max_r = min(cy, cx)

        # ── 2-D FFT → log-magnitude spectrum ────────────────────────────
        F = np.fft.fftshift(np.fft.fft2(gray))
        mag = np.log1p(np.abs(F))

        # ── Distance from centre for each pixel ──────────────────────────
        Y_idx, X_idx = np.ogrid[:H, :W]
        r_map = np.sqrt((Y_idx - cy) ** 2 + (X_idx - cx) ** 2)

        # ── Band energies ─────────────────────────────────────────────────
        band_feats = np.zeros(n_bands, dtype=np.float32)
        band_width = max_r / n_bands
        for i in range(n_bands):
            r_lo = i * band_width
            r_hi = (i + 1) * band_width
            mask = (r_map >= r_lo) & (r_map < r_hi)
            if mask.any():
                band_feats[i] = float(mag[mask].mean())

        out.append(band_feats)

    return np.array(out, dtype=np.float32)   # [T, n_bands]
