"""
Self-supervised audio features for TALVAD-X (frozen wav2vec2).

MFCCs capture short-term spectral envelope but discard the phonetic / prosodic
structure that a self-supervised speech encoder learns. wav2vec2 representations
are the de-facto strong front-end for audio anti-spoofing and audio-visual
deepfake work, and — crucially for CMIS-2 — they give a temporally dense,
content-bearing audio embedding that the cross-modal correspondence head can
align against the lip-region visual stream.

We run wav2vec2 **frozen, inference-only** (no fine-tuning → fits 8 GB, fast,
reproducible) via `torchaudio.pipelines` (no fairseq dependency — clean on
Windows). The ~50 Hz wav2vec2 frames are resampled to the model's video
sequence length so every per-frame feature row lines up across modalities.

Output: float32 [target_len, feature_dim] (feature_dim = 768 for WAV2VEC2_BASE).
Kept complementary to MFCC+Δ+Δ² rather than replacing it.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torchaudio


def wav2vec2_feature_dim(model_name: str = "WAV2VEC2_BASE") -> int:
    """Embedding width for a given torchaudio wav2vec2 bundle (BASE = 768)."""
    return 768 if "LARGE" not in model_name.upper() and "XLSR" not in model_name.upper() else 1024


@lru_cache(maxsize=2)
def _load_bundle(model_name: str, device: str):
    """Load + cache a frozen wav2vec2 model (shared across threads/calls)."""
    bundle = getattr(torchaudio.pipelines, model_name)
    model = bundle.get_model().to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return bundle, model


def _resample_time(feat: torch.Tensor, target_len: int) -> torch.Tensor:
    """Linearly resample a [T, C] sequence to [target_len, C] along time."""
    t = feat.shape[0]
    if t == target_len:
        return feat
    if t == 0:
        return feat.new_zeros((target_len, feat.shape[1]))
    x = feat.t().unsqueeze(0)  # [1, C, T]
    x = torch.nn.functional.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return x.squeeze(0).t().contiguous()  # [target_len, C]


@torch.inference_mode()
def extract_wav2vec2_features(
    y: np.ndarray,
    sr: int,
    target_len: int,
    model_name: str = "WAV2VEC2_BASE",
    layer: int = -1,
    device: str = "cpu",
    max_seconds: Optional[float] = 20.0,
) -> np.ndarray:
    """
    Args:
        y: 1-D waveform (mono), float in roughly [-1, 1].
        sr: sample rate of y.
        target_len: number of output frames (== model sequence_length).
        model_name: a torchaudio.pipelines bundle name (e.g. WAV2VEC2_BASE).
        layer: which transformer layer's output to use (-1 = last). Mid layers
               carry more phonetic detail; exposed for ablation.
        device: 'cuda' or 'cpu'.
        max_seconds: hard cap on audio length (memory guard); None = no cap.

    Returns:
        float32 [target_len, wav2vec2_feature_dim(model_name)].
    """
    dim = wav2vec2_feature_dim(model_name)
    bundle, model = _load_bundle(model_name, device)
    target_sr = bundle.sample_rate  # 16000 for BASE

    wav = torch.as_tensor(np.asarray(y, dtype=np.float32))
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)            # [1, n]
    elif wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # force mono

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if max_seconds is not None:
        cap = int(max_seconds * target_sr)
        if wav.shape[1] > cap:
            wav = wav[:, :cap]
    if wav.shape[1] < 400:  # too short for the conv front-end; pad
        wav = torch.nn.functional.pad(wav, (0, 400 - wav.shape[1]))

    wav = wav.to(device)
    feats, _ = model.extract_features(wav)  # list of [1, T_w, dim]
    f = feats[layer][0]                     # [T_w, dim]
    f = _resample_time(f, target_len)       # [target_len, dim]
    out = f.float().cpu().numpy()
    assert out.shape == (target_len, dim), (out.shape, target_len, dim)
    return out


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sr = 16000
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    # a 220 Hz tone + a little noise — just needs to be a valid waveform
    y = (0.6 * np.sin(2 * np.pi * 220 * t) + 0.02 * np.random.randn(t.size)).astype(np.float32)

    f1 = extract_wav2vec2_features(y, sr, target_len=48, device=dev)
    f2 = extract_wav2vec2_features(y, sr, target_len=48, device=dev)
    print(f"[OK] wav2vec2 out shape={f1.shape} finite={np.isfinite(f1).all()} dim={wav2vec2_feature_dim()}")
    assert f1.shape == (48, 768)
    assert np.isfinite(f1).all()
    assert np.allclose(f1, f2, atol=1e-4), "non-deterministic"

    # resample path (8 kHz input) + short-clip path
    f_lo = extract_wav2vec2_features(y[: sr // 2].copy(), sr=sr, target_len=24, device=dev)
    print(f"[OK] short-clip out shape={f_lo.shape}")
    assert f_lo.shape == (24, 768)

    f8 = extract_wav2vec2_features(np.interp(np.linspace(0, len(y), len(y) // 2), np.arange(len(y)), y).astype(np.float32), sr=8000, target_len=48, device=dev)
    print(f"[OK] 8kHz-resample out shape={f8.shape}")
    assert f8.shape == (48, 768)
    print("[OK] audio_ssl.py self-test passed")
