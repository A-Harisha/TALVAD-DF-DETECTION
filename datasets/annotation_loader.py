from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@lru_cache(maxsize=4)
def _load_metadata_index(json_path: str) -> Dict[str, dict]:
    """
    Load LAV-DF metadata.min.json and index by the bare filename stem
    (e.g. '000002' from 'test/000002.mp4') for O(1) lookup.

    Also indexes by the integer-stripped form (e.g. '2') so that numeric
    sample IDs from the metadata CSV (the 'id' column stores plain integers
    like 2, not zero-padded '000002') resolve correctly.  Both keys point
    to the same entry, so existing stem-based callers are unaffected.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    index: Dict[str, dict] = {}
    for entry in data:
        stem = Path(entry["file"]).stem        # '000002'
        index[stem] = entry
        # Integer-stripped form: '000002' -> '2', '100000' -> '100000' (no-op)
        try:
            int_key = str(int(stem))
            if int_key not in index:           # never overwrite a shorter stem that exists
                index[int_key] = entry
        except ValueError:
            pass
    return index


def get_frame_labels(
    sample_id: str,
    sequence_length: int,
    annotation_json: str,
    fps: float = 1.0,
) -> np.ndarray:
    """
    Convert LAV-DF fake_periods annotations to a per-timestep binary label
    vector of length sequence_length.

    Mapping:  timestep t  →  time  t * duration / sequence_length
    Any timestep whose midpoint falls inside a fake_period is labelled 1.

    Returns np.ndarray of shape (sequence_length,) dtype float32.
    Real videos and samples missing from the index return all-zeros.
    """
    index = _load_metadata_index(annotation_json)
    entry = index.get(sample_id)
    if entry is None or not entry.get("fake_periods"):
        return np.zeros(sequence_length, dtype=np.float32)

    duration: float  = float(entry["duration"])
    fake_periods: List[List[float]] = entry["fake_periods"]

    labels = np.zeros(sequence_length, dtype=np.float32)
    for t in range(sequence_length):
        # Midpoint time of this timestep
        t_mid = (t + 0.5) * duration / sequence_length
        for start, end in fake_periods:
            if start <= t_mid <= end:
                labels[t] = 1.0
                break

    return labels


def get_modality_labels(
    sample_id: str,
    annotation_json: str,
) -> Tuple[float, float]:
    """
    Return (audio_label, video_label) for a sample based on LAV-DF
    modify_audio / modify_video boolean fields.

    Used to supervise the Modality Attribution Heads (MAA) independently
    from the global real/fake label, enabling the model to attribute a
    detected forgery to its source modality.
    """
    index = _load_metadata_index(annotation_json)
    entry = index.get(sample_id)
    if entry is None:
        return 0.0, 0.0
    return float(entry.get("modify_audio", False)), float(entry.get("modify_video", False))
