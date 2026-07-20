from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np


def load_cached_feature(cache_path: str | Path) -> Optional[np.ndarray]:
    cache_path = Path(cache_path)
    if cache_path.exists():
        return np.load(cache_path)
    return None


def save_cached_feature(cache_path: str | Path, arr: np.ndarray) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Use a uniquely named temporary file in the same directory to avoid
    # Windows file-lock races when trying to unlink an existing .tmp file.
    with tempfile.NamedTemporaryFile(dir=cache_path.parent, delete=False, suffix=".tmp") as tmp_file:
        np.save(tmp_file, arr)
        tmp_name = tmp_file.name
    tmp_path = Path(tmp_name)
    os.replace(tmp_path, cache_path)


def build_cache_signature(config: dict) -> str:
    payload = {
        "preprocessing": config.get("preprocessing", {}),
        "features": config.get("features", {}),
        "sequence_length": config.get("training", {}).get("sequence_length", 12),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return digest[:12]
