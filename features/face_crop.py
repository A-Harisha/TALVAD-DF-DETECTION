"""
Face-region cropping for TALVAD-X (visual front-end).

The legacy pipeline fed the **whole 224x224 frame** to a frozen ImageNet
ResNet-18. For talking-head forgeries (LAV-DF, FakeAVCeleb) that dilutes the
manipulated mouth/face region with background and lets the model latch onto
background/codec shortcuts (a likely contributor to the cross-dataset
inversion). Cropping to the face concentrates both the appearance encoder and
the spectral descriptor on exactly where lip-sync and blending artifacts live.

Detector: facenet-pytorch MTCNN (pure-torch, GPU-capable). We use it for the
bounding box only and crop/resize ourselves so we control the margin. Robust
fallbacks: forward/back-fill boxes across a clip to bridge missed frames, and a
centered square crop when no face is found anywhere (so extraction never fails).

Output: uint8 [T, size, size, 3] RGB. No learnable parameters (detector frozen).
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

import numpy as np


@lru_cache(maxsize=2)
def _load_mtcnn(device: str):
    from facenet_pytorch import MTCNN
    # select_largest -> the main speaker's face; post_process=False -> we crop raw.
    return MTCNN(keep_all=False, select_largest=True, post_process=False, device=device)


class FaceCropper:
    def __init__(
        self,
        size: int = 224,
        margin: float = 0.3,
        device: str = "cpu",
        min_conf: float = 0.90,
        detect_every: int = 1,
    ) -> None:
        self.size = int(size)
        self.margin = float(margin)
        self.device = device
        self.min_conf = float(min_conf)
        self.detect_every = max(1, int(detect_every))

    # -- detection -----------------------------------------------------------
    def _detect(self, frames: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        import torch
        from PIL import Image

        mtcnn = _load_mtcnn(self.device)
        boxes_out: List[Optional[np.ndarray]] = [None] * len(frames)
        idxs = list(range(0, len(frames), self.detect_every))
        pil = [Image.fromarray(np.asarray(frames[i], dtype=np.uint8)) for i in idxs]
        with torch.inference_mode():
            boxes, probs = mtcnn.detect(pil)
        for j, i in enumerate(idxs):
            b = boxes[j] if boxes is not None else None
            p = probs[j] if probs is not None else None
            if b is None or p is None:
                continue
            b = np.asarray(b)
            p = np.asarray(p, dtype=np.float32)
            if b.size >= 4 and np.isfinite(p[0]) and float(p[0]) >= self.min_conf:
                boxes_out[i] = b.reshape(-1, 4)[0]
        return boxes_out

    @staticmethod
    def _fill(boxes: List[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
        """Forward-then-backward fill so missed frames borrow the nearest box."""
        T = len(boxes)
        filled: List[Optional[np.ndarray]] = list(boxes)
        last = None
        for i in range(T):
            if filled[i] is not None:
                last = filled[i]
            elif last is not None:
                filled[i] = last
        nxt = None
        for i in range(T - 1, -1, -1):
            if filled[i] is not None:
                nxt = filled[i]
            elif nxt is not None:
                filled[i] = nxt
        return filled

    # -- cropping ------------------------------------------------------------
    def _crop_one(self, frame: np.ndarray, box: Optional[np.ndarray]) -> np.ndarray:
        import cv2

        h, w = frame.shape[:2]
        if box is None:
            s = min(h, w)
            y0, x0 = (h - s) // 2, (w - s) // 2
            crop = frame[y0:y0 + s, x0:x0 + s]
        else:
            x1, y1, x2, y2 = [float(v) for v in box]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            half = max(x2 - x1, y2 - y1) * (1.0 + self.margin) / 2.0
            xa, ya = int(max(0, cx - half)), int(max(0, cy - half))
            xb, yb = int(min(w, cx + half)), int(min(h, cy + half))
            if xb <= xa or yb <= ya:
                s = min(h, w)
                y0, x0 = (h - s) // 2, (w - s) // 2
                crop = frame[y0:y0 + s, x0:x0 + s]
            else:
                crop = frame[ya:yb, xa:xb]
        return cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_AREA)

    def crop_frames(self, frames) -> np.ndarray:
        """frames: sequence of H x W x 3 uint8 RGB -> [T, size, size, 3] uint8."""
        frames = [np.asarray(f, dtype=np.uint8) for f in frames]
        if len(frames) == 0:
            return np.zeros((0, self.size, self.size, 3), dtype=np.uint8)
        boxes = self._fill(self._detect(frames))
        out = np.stack([self._crop_one(frames[i], boxes[i]) for i in range(len(frames))], axis=0)
        return out.astype(np.uint8)


if __name__ == "__main__":
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    H, W, T = 360, 480, 6

    # Plumbing + fallback: random frames almost certainly contain no face ->
    # exercises the centered-square-crop fallback. Output shape/dtype must hold.
    rng = np.random.default_rng(0)
    rand_frames = [rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(T)]
    fc = FaceCropper(size=224, margin=0.3, device=dev)
    out = fc.crop_frames(rand_frames)
    print(f"[OK] crop_frames out shape={out.shape} dtype={out.dtype}")
    assert out.shape == (T, 224, 224, 3) and out.dtype == np.uint8

    # Box-crop math: a known box on a gradient frame -> correct size, right region.
    grad = np.tile(np.linspace(0, 255, W, dtype=np.uint8)[None, :, None], (H, 1, 3))
    box = np.array([100, 50, 200, 150], dtype=np.float32)  # x1,y1,x2,y2
    c = fc._crop_one(grad, box)
    assert c.shape == (224, 224, 3)
    # centre-crop fallback path
    c2 = fc._crop_one(grad, None)
    assert c2.shape == (224, 224, 3)
    print(f"[OK] _crop_one box+fallback shapes OK")

    # Fill logic: a single detected box should propagate to all frames.
    boxes = [None, np.array([10, 10, 50, 50]), None, None]
    filled = FaceCropper._fill(boxes)
    assert all(b is not None for b in filled), filled
    print("[OK] face_crop.py self-test passed (real-face detection validated later on LAV-DF frames)")
