from __future__ import annotations

from typing import Dict, List

import numpy as np
from skimage.feature import graycomatrix, graycoprops


PROPS = ["contrast", "homogeneity", "energy", "correlation", "ASM", "dissimilarity"]
GLCM_FEATURE_NAMES = PROPS + ["entropy"]


def _glcm_entropy(glcm: np.ndarray) -> float:
    p = glcm.astype(np.float64)
    p = p / (p.sum() + 1e-12)
    return float(-(p * np.log2(p + 1e-12)).sum())


def _extract_frame_glcm(frame: np.ndarray, distances: List[int], angles: List[float], levels: int) -> np.ndarray:
    glcm = graycomatrix(
        frame,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    vals = []
    for prop in PROPS:
        vals.append(float(graycoprops(glcm, prop).mean()))
    vals.append(_glcm_entropy(glcm))
    return np.array(vals, dtype=np.float32)


def extract_temporal_glcm_features(
    frames: np.ndarray,
    distances: List[int],
    angles: List[float],
    levels: int = 256,
    temporal_window: int = 3,
) -> np.ndarray:
    """Temporal texture enhancement preserving GLCM-based visual forgery analysis."""
    n = frames.shape[0]
    half = temporal_window // 2
    seq = []

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        local = frames[start:end]
        local_features = [_extract_frame_glcm(f, distances, angles, levels) for f in local]
        seq.append(np.mean(np.stack(local_features, axis=0), axis=0))

    return np.stack(seq, axis=0)


def glcm_feature_dim() -> int:
    return len(GLCM_FEATURE_NAMES)
