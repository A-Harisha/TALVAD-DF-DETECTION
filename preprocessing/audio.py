from __future__ import annotations

import subprocess
from shutil import which
from pathlib import Path
from typing import Tuple
import warnings

import librosa
import numpy as np

try:
    import pywt
except Exception:  # pragma: no cover - optional dependency
    pywt = None


def extract_audio_wav(video_path: str | Path, out_wav_path: str | Path, sample_rate: int = 16000) -> Path:
    out_wav_path = Path(out_wav_path)
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = which("ffmpeg")
    if ffmpeg_exe is None:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe

            ffmpeg_exe = get_ffmpeg_exe()
        except Exception as exc:
            raise FileNotFoundError(
                "ffmpeg executable not found. Install ffmpeg or imageio-ffmpeg to extract audio from video files."
            ) from exc

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-vn",
        "-loglevel",
        "error",
        str(out_wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_wav_path


def preemphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def _optional_wavelet_denoise(signal: np.ndarray) -> np.ndarray:
    if pywt is None:
        return signal

    wavelet = pywt.Wavelet("db4")
    level = min(4, pywt.dwt_max_level(len(signal), wavelet.dec_len))
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) else 0.0
    threshold = sigma * np.sqrt(2.0 * np.log(max(len(signal), 2)))
    coeffs = [coeffs[0]] + [pywt.threshold(c, value=threshold, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)[: len(signal)]


def preprocess_audio_waveform(
    wav_path: str | Path,
    sample_rate: int = 16000,
    preemph: float = 0.97,
    wavelet_denoise: bool = False,
) -> Tuple[np.ndarray, int]:
    try:
        y, sr = librosa.load(wav_path, sr=sample_rate, mono=True)
    except Exception as exc:
        warnings.warn(
            f"Could not load audio from {wav_path}; using a silent fallback waveform.",
            RuntimeWarning,
            stacklevel=2,
        )
        y = np.zeros(sample_rate, dtype=np.float32)
        sr = sample_rate

    if y.size == 0:
        warnings.warn(
            f"Audio from {wav_path} was empty; using a silent fallback waveform.",
            RuntimeWarning,
            stacklevel=2,
        )
        y = np.zeros(sample_rate, dtype=np.float32)

    y = preemphasis(y, coeff=preemph)
    if wavelet_denoise:
        y = _optional_wavelet_denoise(y)

    y = librosa.util.normalize(y)
    return y.astype(np.float32, copy=False), sr
