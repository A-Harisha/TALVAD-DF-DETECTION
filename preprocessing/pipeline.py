from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from preprocessing.audio import extract_audio_wav, preprocess_audio_waveform
from preprocessing.video import extract_frames, preprocess_frames


def process_video_sample(video_path: str | Path, work_dir: str | Path, config: Dict) -> Tuple[np.ndarray, int, np.ndarray]:
    work_dir = Path(work_dir)
    audio_dir = work_dir / "audio"
    frame_dir = work_dir / "frames"

    pre = config["preprocessing"]
    audio_cfg = pre["audio"]

    wav_path = extract_audio_wav(video_path, audio_dir / (Path(video_path).stem + ".wav"), audio_cfg["sample_rate"])
    y, sr = preprocess_audio_waveform(
        wav_path,
        sample_rate=audio_cfg["sample_rate"],
        preemph=audio_cfg["preemphasis"],
        wavelet_denoise=audio_cfg.get("wavelet_denoise", False),
    )

    fpaths = extract_frames(video_path, frame_dir / Path(video_path).stem, fps=pre["fps"])
    frames = preprocess_frames(fpaths, tuple(pre["frame_size"]), pre.get("grayscale", True))

    return y, sr, frames
