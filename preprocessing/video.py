from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np


def extract_frames(video_path: str | Path, out_dir: str | Path, fps: int = 1) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    interval = max(int(round(original_fps / max(fps, 1))), 1)

    frame_paths: List[Path] = []
    idx = 0
    saved = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            path = out_dir / f"frame_{saved:05d}.png"
            cv2.imwrite(str(path), frame)
            frame_paths.append(path)
            saved += 1
        idx += 1

    cap.release()
    return frame_paths


def extract_frames_in_memory(video_path: str | Path, fps: int = 1, frame_size=(224, 224)) -> np.ndarray:
    """Decode video frames directly into a numpy array — no disk writes, no audio."""
    cap = cv2.VideoCapture(str(video_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    interval = max(int(round(original_fps / max(fps, 1))), 1)
    frames = []
    idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    if not frames:
        return np.zeros((1, frame_size[1], frame_size[0], 3), dtype=np.uint8)
    return np.stack(frames, axis=0)


def preprocess_frames(frame_paths: List[Path], frame_size=(128, 128), grayscale: bool = True) -> np.ndarray:
    processed = []
    for p in frame_paths:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.medianBlur(frame, 3)
        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
        processed.append(frame)

    if not processed:
        return np.zeros((1, frame_size[1], frame_size[0]), dtype=np.uint8)

    return np.stack(processed, axis=0)
