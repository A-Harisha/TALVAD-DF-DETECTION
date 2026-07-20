"""
datasets/windowed_localization_dataset.py — fixed-fps SLIDING-WINDOW localization
dataset for TALVAD-L on AV-Deepfake1M (DF1M).

LAV-DF packed every clip into one [N, seq_len, D] array (duration-normalized).
DF1M clips are long and variable-length, so features are stored PER CLIP
(feats/<clip_id>.npy, shape [T_clip, D] at a fixed fps) and this dataset crops
`win`-frame windows on the fly. Boundary targets are re-computed in window-local
frame coords via utils.window_ops (which reuses utils.loc_ops), so the SAME geometry
that the LAV-DF localizer was validated on drives DF1M — the window is just a moving
coordinate frame.

DF1M fakes are word-level (tiny) and sparse, so most windows contain no forgery.
Training therefore samples POSITIVE-biased: every window overlapping a fake segment,
plus `neg_ratio` times as many randomly-chosen empty windows. The returned dict
matches LocalizationPackedDataset exactly, so training.trainer_loc.TrainerL consumes
it with no change; extra keys (clip_id, win_start) are ignored by the trainer and used
by the sliding-window evaluator for stitching.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import window_ops


def load_clip_feats(feats_dir: Path, clip_id: str) -> np.ndarray:
    """Per-clip features [T, D] as float32 (mmap-friendly single-clip read)."""
    return np.asarray(np.load(Path(feats_dir) / f"{clip_id}.npy"), dtype=np.float32)


def crop_window(feats: np.ndarray, win_start: int, win: int) -> np.ndarray:
    """Crop [win_start, win_start+win); zero-pad the tail if the clip is shorter."""
    T, D = feats.shape
    end = win_start + win
    if end <= T:
        return feats[win_start:end]
    out = np.zeros((win, D), dtype=feats.dtype)
    avail = max(0, T - win_start)
    if avail > 0:
        out[:avail] = feats[win_start:T]
    return out


def _windows_for_clip(n_frames: int, win: int, hop: int) -> List[int]:
    return window_ops.make_windows(n_frames, win, hop)


class WindowedLocalizationDataset(Dataset):
    def __init__(
        self,
        feats_dir: str | Path,
        index: Dict[str, dict],            # DF1M annotation index (id -> {fake_periods,...})
        ids: List[str],                    # clip ids in this split
        frames: Dict[str, int],            # id -> n_frames (from extractor manifest)
        fps: float,
        win: int,
        hop: int,
        strides: List[int],
        ranges: List[Tuple[float, float]],
        center_radius: float = 1.5,
        cbm_sigma: float = 1.0,
        scaler=None,
        mode: str = "train",               # 'train' (subsample) | 'all' (every window)
        neg_ratio: float = 1.0,            # empty windows kept per positive window (train)
        min_overlap_frames: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.feats_dir = Path(feats_dir)
        self.index = index
        self.fps = float(fps)
        self.win = int(win)
        self.hop = int(hop)
        self.strides = list(strides)
        self.ranges = [tuple(r) for r in ranges]
        self.center_radius = float(center_radius)
        self.cbm_sigma = float(cbm_sigma)
        self.scaler = scaler
        self.mode = mode
        self.min_overlap = float(min_overlap_frames)

        rng = np.random.default_rng(seed)
        pos: List[Tuple[str, int]] = []
        neg: List[Tuple[str, int]] = []
        self.gt: Dict[str, List[Tuple[float, float]]] = {}
        self.nframes: Dict[str, int] = {}
        for cid in ids:
            n = int(frames.get(cid, 0))
            if n <= 0:
                continue
            gt_sec = list(self.index.get(cid, {}).get("fake_periods", []))
            self.gt[cid] = gt_sec
            self.nframes[cid] = n
            for ws in _windows_for_clip(n, self.win, self.hop):
                has_fake = len(window_ops.segments_to_window(
                    gt_sec, ws, self.win, self.fps, self.min_overlap)) > 0
                (pos if has_fake else neg).append((cid, ws))

        if mode == "all":
            self.samples = pos + neg
        else:
            keep_neg = int(round(neg_ratio * len(pos))) if pos else min(len(neg), 4096)
            if len(neg) > keep_neg:
                sel = rng.choice(len(neg), size=keep_neg, replace=False)
                neg = [neg[i] for i in sel]
            self.samples = pos + neg
            rng.shuffle(self.samples)
        self.n_pos, self.n_neg = len(pos), len(self.samples) - len(pos)

    def __len__(self) -> int:
        return len(self.samples)

    def windows_of(self, clip_id: str) -> List[int]:
        return _windows_for_clip(self.nframes[clip_id], self.win, self.hop)

    def get_window_tensor(self, clip_id: str, win_start: int) -> torch.Tensor:
        x = crop_window(load_clip_feats(self.feats_dir, clip_id), win_start, self.win)
        if self.scaler is not None:
            x = self.scaler.transform(x).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)).float()

    def __getitem__(self, i: int):
        clip_id, ws = self.samples[i]
        x = self.get_window_tensor(clip_id, ws)
        gt_sec = self.gt[clip_id]
        levels, cbm, frame = window_ops.assign_window_targets(
            gt_sec, ws, self.win, self.fps, self.strides, self.ranges,
            self.center_radius, self.cbm_sigma, self.min_overlap)
        y = 1.0 if len(window_ops.segments_to_window(gt_sec, ws, self.win, self.fps, self.min_overlap)) else 0.0
        return {
            "x": x,
            "y": torch.tensor(y).float(),
            "duration": torch.tensor(self.nframes[clip_id] / self.fps).float(),
            "cls": [torch.from_numpy(lv["cls"]).float() for lv in levels],
            "reg": [torch.from_numpy(lv["reg"]).float() for lv in levels],
            "cbm_target": torch.from_numpy(cbm).float(),
            "frame_label": torch.from_numpy(frame).float(),
            "idx": torch.tensor(i, dtype=torch.long),
            "clip_id": clip_id,
            "win_start": torch.tensor(ws, dtype=torch.long),
        }


def load_pack(pack_dir: str | Path):
    """Read a DF1M fixed-fps pack's manifest + split id lists. Returns
    (manifest, ids_by_split, frames, labels)."""
    pack = Path(pack_dir)
    manifest = json.loads((pack / "manifest.json").read_text())
    frames = {k: int(v) for k, v in manifest["frames"].items()}
    labels = {k: int(v) for k, v in manifest.get("labels", {}).items()}
    ids_by_split = manifest.get("splits", {"all": list(frames.keys())})
    return manifest, ids_by_split, frames, labels
