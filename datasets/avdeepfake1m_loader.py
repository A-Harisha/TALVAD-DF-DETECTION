"""
datasets/avdeepfake1m_loader.py — localization-grade annotation loader for
AV-Deepfake1M (DF1M).

The existing scripts/build_avdeepfake1m_csv.py is DETECTION-only: it maps each clip
to a (label, modify_video, modify_audio) row and DROPS the temporal `fake_segments`.
Localization needs those segments. This module reads the official metadata.json and
builds an index keyed by a canonical clip id:

    {clip_id: {file, label, modify_type, fake_periods, audio_periods, visual_periods}}

`fake_periods` are [start,end] pairs in SECONDS (absolute clip time). Clip DURATION
is intentionally NOT taken from the metadata (it is often absent / unreliable); the
fixed-fps extractor records the true frame count from the decoded video, and the
windowing in utils.window_ops works directly in seconds -> frames. So this loader is
the single source of GT *segments*, the extractor is the single source of *duration*.

The canonical id (`df1m_clip_id`) is derived from the relative file path and is the
SAME id the extractor uses to name per-clip features — this is the guard against the
LAV-DF-style "numeric id != padded stem -> all-zero labels" bug. Field names mirror
the official README and are overridable (FIELD_*).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── metadata field names (override if a real file differs) ────────────────────
FIELD_FILE = "file"
FIELD_MODIFY = "modify_type"
FIELD_FAKE_SEGMENTS = "fake_segments"
FIELD_AUDIO_SEGMENTS = "audio_fake_segments"
FIELD_VISUAL_SEGMENTS = "visual_fake_segments"

# modify_type -> (clip label, modify_video, modify_audio)
MODIFY_MAP = {
    "real": (0, 0, 0),
    "visual_modified": (1, 1, 0),
    "audio_modified": (1, 0, 1),
    "both_modified": (1, 1, 1),
}


def df1m_clip_id(file_rel: str) -> str:
    """Canonical, collision-free clip id from a relative file path.

    DF1M reuses filenames across identity folders (00123.mp4 appears under many
    ids), so the stem alone is NOT unique — we hash the normalized relative path.
    The stem prefix is kept for human readability / debugging.
    """
    rel = str(file_rel).replace("\\", "/").lstrip("/")
    stem = Path(rel).stem
    return f"{stem}_{hashlib.md5(rel.encode('utf-8')).hexdigest()[:12]}"


def _segs(rec: dict, key: str) -> List[Tuple[float, float]]:
    raw = rec.get(key) or []
    out: List[Tuple[float, float]] = []
    for pair in raw:
        try:
            s, e = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        if e > s:
            out.append((s, e))
    return out


def load_records(meta_path: Path) -> List[dict]:
    """DF1M metadata may be a JSON list, or a dict keyed by id."""
    data = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        recs = []
        for k, v in data.items():
            if isinstance(v, dict):
                v.setdefault(FIELD_FILE, k)
                recs.append(v)
        return recs
    return list(data)


def build_index(meta_path: str | Path) -> Dict[str, dict]:
    """metadata.json -> {clip_id: {...}}. Real clips get empty period lists."""
    index: Dict[str, dict] = {}
    for rec in load_records(Path(meta_path)):
        rel = str(rec.get(FIELD_FILE, "")).replace("\\", "/").lstrip("/")
        if not rel:
            continue
        raw = str(rec.get(FIELD_MODIFY, "")).strip().lower()
        if raw not in MODIFY_MAP:
            continue
        label, mv, ma = MODIFY_MAP[raw]
        cid = df1m_clip_id(rel)
        index[cid] = {
            "file": rel,
            "label": label,
            "modify_type": raw,
            "modify_video": mv,
            "modify_audio": ma,
            "fake_periods": _segs(rec, FIELD_FAKE_SEGMENTS),
            "audio_periods": _segs(rec, FIELD_AUDIO_SEGMENTS),
            "visual_periods": _segs(rec, FIELD_VISUAL_SEGMENTS),
        }
    return index


def write_min_index(index: Dict[str, dict], out_path: str | Path) -> None:
    """Persist a compact id->segments index for fast dataset/eval loading
    (the DF1M analogue of LAV-DF's metadata.min.json)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index), encoding="utf-8")


def get_periods(index: Dict[str, dict], clip_id: str,
                modality: str = "fake") -> List[Tuple[float, float]]:
    """Segments (seconds) for a clip. modality in {fake, audio, visual}."""
    entry = index.get(clip_id)
    if not entry:
        return []
    return entry.get({"fake": "fake_periods", "audio": "audio_periods",
                      "visual": "visual_periods"}[modality], [])


# ── self-test on a fabricated record (no real data required) ──────────────────
def _selftest() -> None:
    import tempfile

    sample = [
        {"file": "train/id00001/00007.mp4", "modify_type": "both_modified",
         "fake_segments": [[1.2, 1.5], [3.0, 3.25]],
         "audio_fake_segments": [[1.2, 1.5]], "visual_fake_segments": [[3.0, 3.25]]},
        {"file": "train/id00002/00007.mp4", "modify_type": "real"},   # same stem, diff id
        {"file": "train/id00003/00012.mp4", "modify_type": "audio_modified",
         "fake_segments": [[0.4, 0.7]]},
    ]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "meta.json"
        p.write_text(json.dumps(sample), encoding="utf-8")
        idx = build_index(p)
    assert len(idx) == 3, f"expected 3 unique ids, got {len(idx)}"
    ids = list(idx)
    # the two 00007.mp4 under different folders must NOT collide
    assert len({df1m_clip_id(s["file"]) for s in sample}) == 3
    both = next(v for v in idx.values() if v["modify_type"] == "both_modified")
    assert both["label"] == 1 and both["modify_video"] and both["modify_audio"]
    assert both["fake_periods"] == [(1.2, 1.5), (3.0, 3.25)]
    assert both["audio_periods"] == [(1.2, 1.5)] and both["visual_periods"] == [(3.0, 3.25)]
    real = next(v for v in idx.values() if v["modify_type"] == "real")
    assert real["label"] == 0 and real["fake_periods"] == []
    print(f"  parsed {len(idx)} clips; ids unique; segments + modality split OK")
    print("AVDEEPFAKE1M_LOADER OK")


if __name__ == "__main__":
    _selftest()
