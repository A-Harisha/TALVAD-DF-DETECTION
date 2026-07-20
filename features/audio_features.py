from __future__ import annotations

import numpy as np
import librosa

AUDIO_FEATURE_MULTIPLIER = 3


def extract_mfcc_with_deltas(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    ).T
    delta = librosa.feature.delta(mfcc.T).T
    delta2 = librosa.feature.delta(mfcc.T, order=2).T

    feat = np.concatenate([mfcc, delta, delta2], axis=1)
    mean = feat.mean(axis=0, keepdims=True)
    std = feat.std(axis=0, keepdims=True) + 1e-8
    return (feat - mean) / std


def audio_feature_dim(n_mfcc: int) -> int:
    return n_mfcc * AUDIO_FEATURE_MULTIPLIER
