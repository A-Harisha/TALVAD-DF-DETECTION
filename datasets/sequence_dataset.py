from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.annotation_loader import (
    _load_metadata_index,
    get_frame_labels,
    get_modality_labels,
)
from utils.loc_ops import assign_targets, gaussian_boundary_target, seconds_to_frames


class MultiModalSequenceDataset(Dataset):
    def __init__(
        self,
        rows,
        feature_cache_dir: str | Path,
        sequence_length: int = 12,
        feature_mask: np.ndarray | str | Path | None = None,
        annotation_json: str | Path | None = None,
        fps: float = 1.0,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.feature_cache_dir = Path(feature_cache_dir)
        self.sequence_length = sequence_length
        self.annotation_json = str(annotation_json) if annotation_json else None
        self.fps = fps
        if isinstance(feature_mask, (str, Path)):
            self.feature_mask = np.load(feature_mask)
        else:
            self.feature_mask = feature_mask

    def __len__(self) -> int:
        return len(self.rows)

    def _pad_or_trim(self, arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] == self.sequence_length:
            return arr
        if arr.shape[0] > self.sequence_length:
            return arr[: self.sequence_length]
        pad = np.repeat(arr[-1:, :], self.sequence_length - arr.shape[0], axis=0)
        return np.concatenate([arr, pad], axis=0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[idx]
        sample_id = str(row.get("id", idx))

        x_path = self.feature_cache_dir / f"{sample_id}_sequence.npy"
        if not x_path.exists():
            raise FileNotFoundError(f"Missing cached feature sequence: {x_path}")

        x = np.load(x_path)
        x = self._pad_or_trim(x)
        if self.feature_mask is not None:
            mask = np.asarray(self.feature_mask, dtype=x.dtype)
            if mask.shape[-1] != x.shape[-1]:
                raise ValueError(
                    f"Feature mask dimension {mask.shape[-1]} does not match "
                    f"cached feature dimension {x.shape[-1]}"
                )
            x = x * mask

        y = np.array(row["label"], dtype=np.float32)

        # Frame-level forgery labels from LAV-DF fake_periods annotations.
        # Returns all-zeros for real videos or when annotation_json not provided.
        if self.annotation_json:
            frame_y = get_frame_labels(
                sample_id,
                self.sequence_length,
                self.annotation_json,
                fps=self.fps,
            )
            audio_y, video_y = get_modality_labels(sample_id, self.annotation_json)
        else:
            frame_y = np.zeros(self.sequence_length, dtype=np.float32)
            audio_y, video_y = 0.0, 0.0

        return (
            torch.from_numpy(x).float(),
            torch.tensor(y).float(),
            torch.from_numpy(frame_y).float(),
            torch.tensor(audio_y, dtype=torch.float32),
            torch.tensor(video_y, dtype=torch.float32),
        )


class PackedSequenceDataset(Dataset):
    """
    In-memory dataset backed by pre-loaded numpy arrays.

    Zero disk reads during training — __getitem__ is pure array indexing.
    Created by training/train.py when data/packed/<sig>/features.npy exists.
    Frame labels are pre-loaded at __init__ so there is no per-sample I/O.
    """

    def __init__(
        self,
        features: np.ndarray,               # [N, T, D] float32
        labels: np.ndarray,                 # [N]
        ids: List[str],                     # sample IDs for annotation lookup
        sequence_length: int,
        feature_mask: Optional[np.ndarray] = None,
        annotation_json: Optional[str] = None,
        fps: float = 1.0,
    ) -> None:
        assert len(features) == len(labels) == len(ids), (
            f"Length mismatch: features={len(features)} labels={len(labels)} ids={len(ids)}"
        )
        self.features        = features
        self.labels          = labels.astype(np.float32)
        self.ids             = list(ids)
        self.sequence_length = sequence_length
        self.feature_mask    = feature_mask

        # Pre-load all frame labels at init (N × T floats — cheap, eliminates
        # per-sample JSON parsing during DataLoader iteration).
        ann = str(annotation_json) if annotation_json else None
        if ann:
            self._frame_labels = np.stack(
                [get_frame_labels(sid, sequence_length, ann, fps=fps) for sid in ids],
                axis=0,
            )  # [N, T]
            modality_pairs = [get_modality_labels(sid, ann) for sid in ids]
            self._modality_labels = np.array(modality_pairs, dtype=np.float32)  # [N, 2]
        else:
            self._frame_labels = np.zeros(
                (len(ids), sequence_length), dtype=np.float32
            )
            self._modality_labels = np.zeros((len(ids), 2), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.features[idx].copy()   # [T, D]  — copy avoids mmap aliasing

        if self.feature_mask is not None:
            x = x * self.feature_mask   # broadcasts [D] over [T, D]

        audio_y, video_y = self._modality_labels[idx]
        return (
            torch.from_numpy(x).float(),
            torch.tensor(self.labels[idx]).float(),
            torch.from_numpy(self._frame_labels[idx]).float(),
            torch.tensor(audio_y, dtype=torch.float32),
            torch.tensor(video_y, dtype=torch.float32),
        )


class LocalizationPackedDataset(Dataset):
    """In-memory dataset for TALVAD-L localization.

    Pre-computes, per clip: the per-level center-based targets (cls / reg) and the
    CBM Gaussian boundary target, so the DataLoader does pure array indexing. GT
    segments (seconds) + duration are returned too so the eval decoder can score
    against them. All targets are built through utils.loc_ops — the SAME geometry
    the decoder uses — which is what keeps train and eval on one time axis.
    """

    def __init__(
        self,
        features: np.ndarray,               # [N, T, D] float32
        labels: np.ndarray,                 # [N]
        ids: List[str],
        sequence_length: int,
        annotation_json: str,
        strides: List[int],
        ranges: List[Tuple[float, float]],
        center_radius: float = 1.5,
        cbm_sigma: float = 1.0,
        scaler=None,
        index_map=None,
    ) -> None:
        # `features` may be the FULL memmap; `index_map` selects this split's global
        # rows so we never materialize a subset (fancy-indexing a memmap copies it).
        self.features = features          # memmap (np.load mmap_mode='r') or array
        self.index_map = (np.arange(len(labels)) if index_map is None
                          else np.asarray(index_map, dtype=np.int64))
        assert len(self.index_map) == len(labels) == len(ids)
        self.labels = labels.astype(np.float32)
        self.ids = list(ids)
        self.T = sequence_length
        self.strides = strides
        self.scaler = scaler              # applied PER-SAMPLE so full-data never loads whole

        index = _load_metadata_index(str(annotation_json))
        N = len(ids)
        self.durations = np.zeros(N, dtype=np.float32)
        self.gt_frames: List[List[Tuple[float, float]]] = []
        self.gt_seconds: List[List[Tuple[float, float]]] = []
        self._cls: List[np.ndarray] = [
            np.zeros((N, len(level_centers_len(sequence_length, s))), dtype=np.float32) for s in strides
        ]
        self._reg: List[np.ndarray] = [
            np.zeros((N, len(level_centers_len(sequence_length, s)), 2), dtype=np.float32) for s in strides
        ]
        self._cbm = np.zeros((N, sequence_length), dtype=np.float32)
        self._frame = np.zeros((N, sequence_length), dtype=np.float32)

        for n, sid in enumerate(ids):
            entry = index.get(sid)
            dur = float(entry["duration"]) if entry else 0.0
            periods = entry.get("fake_periods", []) if entry else []
            self.durations[n] = dur
            gsec = [(float(s), float(e)) for s, e in periods]
            gfr = [seconds_to_frames(s, e, dur, sequence_length) for s, e in gsec] if dur > 0 else []
            self.gt_seconds.append(gsec)
            self.gt_frames.append(gfr)
            if gfr:
                lv = assign_targets(gfr, sequence_length, strides, ranges, center_radius)
                for li in range(len(strides)):
                    self._cls[li][n] = lv[li]["cls"]
                    self._reg[li][n] = lv[li]["reg"]
                self._cbm[n] = gaussian_boundary_target(gfr, sequence_length, cbm_sigma)
                self._frame[n] = get_frame_labels(sid, sequence_length, str(annotation_json))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        g = int(self.index_map[idx])
        x = np.asarray(self.features[g], dtype=np.float32)     # materialize ONE clip (mmap-safe)
        if self.scaler is not None:
            x = self.scaler.transform(x).astype(np.float32)    # per-sample standardization
        return {
            "x": torch.from_numpy(x).float(),
            "y": torch.tensor(self.labels[idx]).float(),
            "duration": torch.tensor(self.durations[idx]).float(),
            "cls": [torch.from_numpy(self._cls[li][idx]).float() for li in range(len(self.strides))],
            "reg": [torch.from_numpy(self._reg[li][idx]).float() for li in range(len(self.strides))],
            "cbm_target": torch.from_numpy(self._cbm[idx]).float(),
            "frame_label": torch.from_numpy(self._frame[idx]).float(),
            "idx": torch.tensor(idx, dtype=torch.long),
        }


def level_centers_len(T0: int, stride: int) -> np.ndarray:
    import math
    Tl = math.ceil(T0 / stride) if stride > 1 else T0
    return np.arange(Tl)
