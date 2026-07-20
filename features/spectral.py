"""
Enriched spectral descriptor for TALVAD-X (CMIS-2 frequency branch).

Generalises the original 8-band radial GFC (`features/gfc.py`) into a richer,
structure-bearing descriptor so that the inconsistency-gated spectral attention
in CMIS-2 has something meaningful to attend over. For each frame we compute:

  1. radial x angular log-magnitude FFT bins   (n_radial * n_angular features)
  2. high-frequency residual ratio             (1 feature)  — outer-ring energy
                                                              relative to all
  3. high-frequency phase coherence            (1 feature)  — circular variance
                                                              of the phase in the
                                                              high-frequency region

Why: neural synthesis (GAN/diffusion upsampling, blending, codec) leaves
amplitude *and* phase fingerprints in the 2-D Fourier spectrum (Frank 2020;
Durall 2020; FreqNet/AAAI 2024). The original GFC kept only radial amplitude;
adding angular structure, an explicit HF-energy ratio, and a phase-coherence
term gives a descriptor that is both more discriminative in-domain and (paired
with the cross-modal gate) more transferable.

Backward-compatible: with n_angular=1, include_hf_residual=False,
include_phase=False this reduces to the legacy radial GFC, so the
spectral-richness ablation toggles cleanly.

Pure numpy, no learnable parameters, no GPU. Output: float32 [T, out_dim].
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np


def spectral_feature_dim(
    n_radial: int = 8,
    n_angular: int = 4,
    include_hf_residual: bool = True,
    include_phase: bool = True,
) -> int:
    """Return the output feature dimension for the given settings."""
    return n_radial * n_angular + int(include_hf_residual) + int(include_phase)


@lru_cache(maxsize=16)
def _geometry(h: int, w: int, n_radial: int, n_angular: int):
    """Precompute (cached per shape/binning) the per-pixel bin assignments.

    Returns flat-index arrays restricted to the inscribed disc (r <= r_max) so
    the spectrum is binned without corner bias, plus masks needed for the HF
    residual and phase terms.
    """
    # After np.fft.fftshift the DC component lands at index (h//2, w//2); centre
    # the radial/angular geometry there so bin 0 is the true low-frequency core
    # (matches the legacy GFC convention in features/gfc.py).
    cy, cx = float(h // 2), float(w // 2)
    y, x = np.ogrid[:h, :w]
    dy, dx = (y - cy), (x - cx)
    r = np.sqrt(dy * dy + dx * dx)
    r_max = float(min(cy, cx))
    r_norm = r / (r_max + 1e-8)

    valid = r_norm <= 1.0  # inscribed disc
    # radial bin in [0, n_radial-1]
    rad_idx = np.clip((r_norm * n_radial).astype(np.int64), 0, n_radial - 1)
    # angular bin in [0, n_angular-1]; theta mapped to [0,1)
    theta = (np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)
    theta = np.broadcast_to(theta, (h, w))
    ang_idx = np.clip((theta * n_angular).astype(np.int64), 0, n_angular - 1)

    n_bins = n_radial * n_angular
    bin_id = rad_idx * n_angular + ang_idx  # [h, w]

    valid_flat = valid.reshape(-1)
    bin_id_valid = bin_id.reshape(-1)[valid_flat]
    counts = np.bincount(bin_id_valid, minlength=n_bins).astype(np.float32)
    counts[counts == 0] = 1.0  # avoid /0 for empty bins

    outer_mask = (rad_idx == (n_radial - 1)) & valid  # high-freq ring
    non_dc = (rad_idx >= 1) & valid                   # all but the DC/low core ring
    hf_region = (r_norm > 0.5) & valid                # high-freq half for phase

    return {
        "valid_flat": valid_flat,
        "bin_id_valid": bin_id_valid,
        "counts": counts,
        "n_bins": n_bins,
        "outer_flat": outer_mask.reshape(-1),
        "non_dc_flat": non_dc.reshape(-1),
        "hf_flat": hf_region.reshape(-1),
    }


def _to_gray(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim == 3 and frame.shape[2] >= 3:
        return 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
    if frame.ndim == 3:
        return frame[..., 0]
    return frame


def extract_spectral_features(
    frames: Sequence[np.ndarray],
    n_radial: int = 8,
    n_angular: int = 4,
    include_hf_residual: bool = True,
    include_phase: bool = True,
) -> np.ndarray:
    """
    Args:
        frames: sequence of H x W x C (or H x W) arrays, one per video frame.
        n_radial, n_angular: spectrum partitioning (angular=1 -> legacy GFC).
        include_hf_residual: append outer-ring / all energy ratio.
        include_phase: append high-frequency phase circular variance.

    Returns:
        float32 array [T, spectral_feature_dim(...)].
    """
    out_dim = spectral_feature_dim(n_radial, n_angular, include_hf_residual, include_phase)
    if len(frames) == 0:
        return np.zeros((0, out_dim), dtype=np.float32)

    rows = []
    geom = None
    for frame in frames:
        gray = _to_gray(frame)
        h, w = gray.shape
        if geom is None or geom_shape != (h, w):  # noqa: F821 (defined below)
            geom = _geometry(h, w, n_radial, n_angular)
            geom_shape = (h, w)

        F = np.fft.fftshift(np.fft.fft2(gray))
        mag = np.log1p(np.abs(F)).reshape(-1)

        valid_flat = geom["valid_flat"]
        sums = np.bincount(
            geom["bin_id_valid"], weights=mag[valid_flat], minlength=geom["n_bins"]
        ).astype(np.float32)
        band = sums / geom["counts"]  # mean log-magnitude per (radial,angular) bin

        feats = [band]

        if include_hf_residual:
            # Outer-ring power as a fraction of all NON-DC power. Excluding the
            # DC/low-frequency core ring (which otherwise dwarfs everything) makes
            # this a variance-rich high-vs-mid frequency ratio in [0,1]: small for
            # smooth frames, large when synthesis injects high-frequency detail.
            power = (np.abs(F) ** 2).reshape(-1)
            denom = float(power[geom["non_dc_flat"]].sum()) + 1e-8
            outer = float(power[geom["outer_flat"]].sum())
            feats.append(np.array([outer / denom], dtype=np.float32))

        if include_phase:
            phase = np.angle(F).reshape(-1)
            hf = geom["hf_flat"]
            if hf.any():
                z = np.exp(1j * phase[hf])
                circ_var = 1.0 - np.abs(z.mean())  # 0 = coherent, 1 = uniform/random
            else:
                circ_var = 0.0
            feats.append(np.array([circ_var], dtype=np.float32))

        rows.append(np.concatenate(feats).astype(np.float32))

    return np.stack(rows, axis=0).astype(np.float32)


if __name__ == "__main__":
    # Self-test: shapes, determinism, finiteness, legacy-GFC reduction, and that
    # genuine high-frequency content raises the HF-power term.
    T, H, W = 5, 224, 224
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # Low-frequency base: smooth 2-D sinusoid (energy concentrated near DC).
    lowf = np.sin(2 * np.pi * xx / W) + np.sin(2 * np.pi * yy / H)
    lowf = (lowf - lowf.min()) / (lowf.max() - lowf.min())
    low_img = (lowf * 255).astype(np.uint8)
    low_frames = [np.stack([low_img] * 3, axis=-1)] * T

    # Two clean sub-Nyquist gratings whose energy lands INSIDE the inscribed disc:
    # a mid-frequency one (peaks in a middle ring) and a high-frequency one (peaks
    # in the outer ring). The outer one must show a much larger HF-power fraction.
    def _grating(freq_cycles):
        g = np.sin(2 * np.pi * freq_cycles * xx / W)
        g = (g - g.min()) / (g.max() - g.min())
        return (g * 255).astype(np.uint8)
    mid_frames = [np.stack([_grating(40)] * 3, axis=-1)] * T    # ring ~2 of 8
    high_frames = [np.stack([_grating(100)] * 3, axis=-1)] * T  # ring ~7 of 8 (outer)

    for nr, na, hf, ph in [(8, 1, False, False), (8, 4, True, True), (12, 6, True, True)]:
        d = spectral_feature_dim(nr, na, hf, ph)
        f1 = extract_spectral_features(low_frames, nr, na, hf, ph)
        f2 = extract_spectral_features(low_frames, nr, na, hf, ph)
        assert f1.shape == (T, d), (f1.shape, d)
        assert np.allclose(f1, f2), "non-deterministic"
        assert np.isfinite(f1).all(), "non-finite output"
        print(f"[OK] radial={nr} angular={na} hf={hf} phase={ph} -> dim={d} shape={f1.shape}")

    # Legacy-GFC reduction: radial=8/angular=1/no-extras must match features/gfc.py
    try:
        from features.gfc import extract_gfc_features
    except Exception:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from features.gfc import extract_gfc_features
    legacy = extract_gfc_features(low_frames, n_bands=8)
    ours = extract_spectral_features(low_frames, 8, 1, False, False)
    # Not bit-exact: legacy uses half-open radial ranges, we use clipped indices,
    # so boundary pixels differ slightly. They should agree to ~1e-2.
    print(f"[CHECK] legacy-GFC max|diff|={np.abs(legacy - ours).max():.6f}")
    assert np.allclose(legacy, ours, atol=2e-2), "radial-only mode departs from legacy GFC"

    # HF power fraction must be much larger for outer-ring than mid-ring energy.
    hf_idx = 8 * 4  # position of the hf_residual term in the (8,4,True,True) layout
    lo = extract_spectral_features(mid_frames, 8, 4, True, True)[:, hf_idx].mean()
    hi = extract_spectral_features(high_frames, 8, 4, True, True)[:, hf_idx].mean()
    print(f"[CHECK] HF-power fraction (non-DC)  mid-freq={lo:.5f}  high-freq={hi:.5f}")
    assert hi > lo, "HF power did not respond to higher-frequency content"
    assert hi > 0.5, f"HF feature looks degenerate (hi={hi:.5f}) — should be large for outer-ring energy"
    print("[OK] spectral.py self-test passed")
