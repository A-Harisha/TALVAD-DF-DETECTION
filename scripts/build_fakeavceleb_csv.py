"""
scripts/build_fakeavceleb_csv.py

Build a metadata CSV for FakeAVCeleb that scripts/eval_crossdataset.py can read.

FakeAVCeleb organises clips under four category folders:
    RealVideo-RealAudio   -> real          (label 0)
    FakeVideo-RealAudio   -> fake (video)   (label 1, modify_video=1)
    RealVideo-FakeAudio   -> fake (audio)   (label 1, modify_audio=1)
    FakeVideo-FakeAudio   -> fake (both)    (label 1, modify_video=1, modify_audio=1)

The label/modality is inferred from substrings in each file's path, so it is
robust to the exact nesting (identity subfolders, etc.). Paths are written
relative to --root, so pass the same --root as --root_dir to eval_crossdataset.

Usage:
  python scripts/build_fakeavceleb_csv.py --root /path/to/FakeAVCeleb --out data/fakeavceleb_metadata.csv
  # then:
  python scripts/eval_crossdataset.py --config configs/lavdf_talvad_v8gen.yaml \
      --data_csv data/fakeavceleb_metadata.csv --root_dir /path/to/FakeAVCeleb --name fakeavceleb
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def classify(path_str: str) -> tuple[int, int, int]:
    """Return (label, modify_video, modify_audio) from path substrings."""
    p = path_str.lower().replace("_", "").replace("-", "")
    fake_video = "fakevideo" in p
    fake_audio = "fakeaudio" in p
    label = 1 if (fake_video or fake_audio) else 0
    return label, int(fake_video), int(fake_audio)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FakeAVCeleb metadata CSV")
    ap.add_argument("--root", required=True, help="FakeAVCeleb dataset root")
    ap.add_argument("--out", default="data/fakeavceleb_metadata.csv")
    ap.add_argument("--exts", nargs="+", default=[".mp4", ".avi", ".mov"],
                    help="Video extensions to include")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        ap.error(f"root not found: {root}")

    exts = {e.lower() for e in args.exts}
    rows = []
    counts = {"real": 0, "fake_video": 0, "fake_audio": 0, "fake_both": 0, "unknown": 0}
    for f in root.rglob("*"):
        if f.suffix.lower() not in exts or not f.is_file():
            continue
        rel = f.relative_to(root).as_posix()
        label, mv, ma = classify(str(f))
        rows.append((rel, label, mv, ma))
        if label == 0:
            counts["real"] += 1
        elif mv and ma:
            counts["fake_both"] += 1
        elif mv:
            counts["fake_video"] += 1
        elif ma:
            counts["fake_audio"] += 1
        else:
            counts["unknown"] += 1

    if not rows:
        ap.error(f"No videos with extensions {sorted(exts)} found under {root}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video_path", "label", "modify_video", "modify_audio"])
        w.writerows(rows)

    n = len(rows)
    n_real = counts["real"]
    n_fake = n - n_real
    print(f"Wrote {n:,} rows -> {out}")
    print(f"  real          : {n_real:,}")
    print(f"  fake (video)  : {counts['fake_video']:,}")
    print(f"  fake (audio)  : {counts['fake_audio']:,}")
    print(f"  fake (both)   : {counts['fake_both']:,}")
    print(f"  fake total    : {n_fake:,}  ({100*n_fake/n:.1f}%)")
    if counts["unknown"]:
        print(f"  WARNING: {counts['unknown']:,} files matched no Real/Fake category — "
              f"check the folder layout (expected RealVideo-RealAudio / FakeVideo-* etc.)")


if __name__ == "__main__":
    main()
