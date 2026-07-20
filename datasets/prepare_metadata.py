from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
FAKE_TOKENS = {"fake", "forged", "manipulated", "deepfake"}
REAL_TOKENS = {"real", "authentic", "original"}


def _iter_videos(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            yield p


def _infer_label(path: Path) -> int | None:
    parts = [s.lower() for s in path.parts]
    for token in FAKE_TOKENS:
        if token in parts:
            return 1
    for token in REAL_TOKENS:
        if token in parts:
            return 0
    return None


def _normalize_video_path(value: str, root: Path) -> str:
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((root / p).resolve())


def _from_split_json(split_json: Path, root: Path, dataset_name: str) -> pd.DataFrame:
    payload = json.loads(split_json.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("rows", payload))

    if not isinstance(payload, list):
        raise ValueError(f"Unsupported JSON structure in {split_json}")

    rows = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            continue

        video_path = item.get("video_path") or item.get("file") or item.get("path")
        if video_path is None:
            continue
        normalized_video_path = _normalize_video_path(str(video_path), root)
        if not Path(normalized_video_path).exists() or Path(normalized_video_path).name.startswith("._"):
            continue

        label = item.get("label")
        if label is None:
            label = item.get("is_fake")
        if label is None and "split" in item:
            label = 0 if str(item["split"]).lower() == "real" else item.get("n_fakes", 1)
        if label is None:
            label = _infer_label(Path(str(video_path)))
        if label is None:
            label = 0 if str(item.get("category", "")).lower().startswith("real") else 1

        if isinstance(label, bool):
            label = int(label)

        if isinstance(label, str):
            label = label.lower()
            label = 0 if label in {"0", "real", "authentic", "original"} else 1
        elif isinstance(label, (int, float)):
            label = 0 if int(label) == 0 else 1

        rows.append(
            {
                "id": item.get("id", Path(str(video_path)).stem if video_path else f"sample_{i}"),
                "video_path": normalized_video_path,
                "label": int(label),
                "dataset": item.get("dataset", dataset_name),
            }
        )

    if not rows:
        raise RuntimeError(f"No usable rows found in {split_json}")

    return pd.DataFrame(rows)


def build_metadata(root_dir: str | Path, output_csv: str | Path, dataset_name: str) -> Path:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    for candidate in [root / "metadata.min.json", root / "train.json", root / "val.json", root / "test.json"]:
        if candidate.exists() and candidate.suffix == ".json":
            try:
                df = _from_split_json(candidate, root, dataset_name)
                out = Path(output_csv)
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out, index=False)
                print(f"Saved metadata to {out} from {candidate} with {len(df)} samples.")
                return out
            except Exception:
                pass

    rows = []
    skipped = 0
    for vp in _iter_videos(root):
        label = _infer_label(vp)
        if label is None:
            skipped += 1
            continue

        rows.append(
            {
                "id": vp.stem,
                "video_path": str(vp.relative_to(root)).replace("\\", "/"),
                "label": int(label),
                "dataset": dataset_name,
            }
        )

    if not rows:
        raise RuntimeError(
            "No labeled videos found. Expected folder names containing real/authentic/original or fake/forged/manipulated/deepfake."
        )

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    print(
        f"Saved metadata to {out} with {len(rows)} samples. "
        f"Skipped {skipped} videos without inferable label tokens."
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--output", required=True, help="Output metadata CSV path")
    parser.add_argument("--dataset", default="custom", help="Dataset name field value")
    args = parser.parse_args()

    build_metadata(args.root, args.output, args.dataset)
