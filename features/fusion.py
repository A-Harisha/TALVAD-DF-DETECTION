from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureFusionNormalizer:
    def __init__(self) -> None:
        self.scaler = StandardScaler()

    def fuse(self, audio_seq: np.ndarray, video_seq: np.ndarray) -> np.ndarray:
        n = min(audio_seq.shape[0], video_seq.shape[0])
        return np.concatenate([audio_seq[:n], video_seq[:n]], axis=1)

    def _reshape(self, x: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        if x.ndim == 2:
            return x, x.shape
        if x.ndim == 3:
            shape = x.shape
            return x.reshape(-1, shape[-1]), shape
        raise ValueError(f"Expected 2D or 3D array, received shape {x.shape}")

    def fit(self, x: np.ndarray) -> "FeatureFusionNormalizer":
        flat, _ = self._reshape(x)
        self.scaler.fit(flat)
        return self

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        flat, shape = self._reshape(x)
        scaled = self.scaler.fit_transform(flat)
        if len(shape) == 3:
            return scaled.reshape(shape)
        return scaled

    def transform(self, x: np.ndarray) -> np.ndarray:
        flat, shape = self._reshape(x)
        scaled = self.scaler.transform(flat)
        if len(shape) == 3:
            return scaled.reshape(shape)
        return scaled

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)

    def load(self, path: str | Path) -> None:
        self.scaler = joblib.load(path)


def align_and_pack_sequences(
    fused_seq: np.ndarray,
    sequence_length: int,
    stride: Optional[int] = None,
) -> np.ndarray:
    if stride is None:
        stride = sequence_length // 2

    if fused_seq.shape[0] == 0:
        return np.zeros((1, sequence_length, fused_seq.shape[-1]), dtype=fused_seq.dtype)

    windows = []
    if fused_seq.shape[0] < sequence_length:
        pad = np.repeat(fused_seq[-1:, :], sequence_length - fused_seq.shape[0], axis=0)
        windows.append(np.concatenate([fused_seq, pad], axis=0))
        return np.stack(windows, axis=0)

    for s in range(0, fused_seq.shape[0] - sequence_length + 1, max(stride, 1)):
        windows.append(fused_seq[s : s + sequence_length])

    return np.stack(windows, axis=0)


def apply_feature_mask(fused_seq: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return fused_seq
    mask = np.asarray(mask, dtype=fused_seq.dtype)
    if mask.shape[-1] != fused_seq.shape[-1]:
        raise ValueError(f"Mask dimension {mask.shape[-1]} does not match feature dimension {fused_seq.shape[-1]}")
    return fused_seq * mask
