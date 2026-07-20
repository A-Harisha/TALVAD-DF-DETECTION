"""
scripts/build_deepfaketimit_csv.py

Build a metadata CSV for DeepfakeTIMIT that is compatible with
eval_crossdataset.py / extract_talvad_x.py.

DeepfakeTIMIT structure (after downloading + unzipping):
  datasets/DeepfakeTIMIT/
    higher_quality/   (HQ, 128x128 face swap)
      id00076/
        <video>.avi
    lower_quality/    (LQ, 64x64 face swap)
      id00076/
        <video>.avi
    real/             (VidTIMIT originals, from zenodo.org/records/158963)
      id00076/
        <video>.avi   or  <video>.mp4

Columns in the output CSV:
  video_path, label (0=real / 1=fake), quality (HQ/LQ/real), subject_id

Downloads needed:
  1. Fake videos: zenodo.org/records/4068245  (~226 MB, HQ+LQ)
  2. Real videos: zenodo.org/records/158963   (VidTIMIT, ~30 MB audio, video separate)
     OR use the real videos bundled in some versions of DeepfakeTIMIT.

Usage:
  python scripts/build_deepfaketimit_csv.py \\
      --root datasets/DeepfakeTIMIT \\
      --output data/deepfaketimit_metadata.csv
  python scripts/build_deepfaketimit_csv.py \\
      --root datasets/DeepfakeTIMIT --quality HQ \\
      --output data/deepfaketimit_hq_metadata.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_csv(root: Path, output: Path, quality: str | None = None) -> pd.DataFrame:
    rows = []
    root = root.expanduser().resolve()

    fake_dirs = {
        "HQ": root / "higher_quality",
        "LQ": root / "lower_quality",
    }
    real_candidates = [root / "real", root / "RealVideos", root / "real_videos"]

    # Fake videos (HQ and/or LQ)
    for q, d in fake_dirs.items():
        if quality and q != quality:
            continue
        if not d.exists():
            print(f"  SKIP: {d} not found")
            continue
        vids = list(d.rglob("*.avi")) + list(d.rglob("*.mp4")) + list(d.rglob("*.mpg"))
        for v in vids:
            rows.append({
                "video_path": str(v),
                "label": 1,
                "quality": q,
                "subject_id": v.parent.name,
                "modify_audio": 0,
                "modify_video": 1,
            })
        print(f"  {q}: {len(vids)} fake videos")

    # Real videos (VidTIMIT originals)
    real_dir = None
    for c in real_candidates:
        if c.exists():
            real_dir = c; break
    if real_dir:
        rvids = list(real_dir.rglob("*.avi")) + list(real_dir.rglob("*.mp4")) + list(real_dir.rglob("*.mpg"))
        for v in rvids:
            rows.append({
                "video_path": str(v),
                "label": 0,
                "quality": "real",
                "subject_id": v.parent.name,
                "modify_audio": 0,
                "modify_video": 0,
            })
        print(f"  real: {len(rvids)} real videos")
    else:
        print("  WARNING: no 'real/' directory found — add VidTIMIT videos to datasets/DeepfakeTIMIT/real/")

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"  ERROR: no videos found under {root}")
        return df

    exists = df.video_path.map(lambda p: Path(p).exists())
    missing = int((~exists).sum())
    if missing:
        print(f"  WARNING: {missing} paths don't exist — dropped")
    df = df[exists].reset_index(drop=True)

    df["id"] = df.video_path.apply(lambda p: Path(p).stem)
    # Deduplicate ids if any
    if df.id.duplicated().any():
        df["id"] = df.apply(lambda r: f"{r.quality}_{r.subject_id}_{Path(r.video_path).stem}", axis=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    n_real = int((df.label == 0).sum()); n_fake = int((df.label == 1).sum())
    print(f"  Saved {len(df)} rows -> {output}  (real={n_real} fake={n_fake})")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/DeepfakeTIMIT")
    ap.add_argument("--output", default="data/deepfaketimit_metadata.csv")
    ap.add_argument("--quality", default=None, choices=["HQ", "LQ"],
                    help="HQ=128x128 only; LQ=64x64 only; omit for both")
    args = ap.parse_args()
    build_csv(Path(args.root), Path(args.output), args.quality)
