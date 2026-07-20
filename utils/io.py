from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)
