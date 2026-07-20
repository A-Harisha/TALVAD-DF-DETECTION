"""
scripts/build_dfdc_csv.py

Build a metadata CSV for DFDC (Deepfake Detection Challenge) that
scripts/eval_crossdataset.py can read.

DFDC ships as .mp4 clips grouped in part folders, each part carrying a
`metadata.json` of the form:

    {
      "aagfhgtpmv.mp4": {"label": "FAKE", "split": "train", "original": "vudstovrck.mp4"},
      "abarnvbtwb.mp4": {"label": "REAL", "split": "train"}
    }

This script walks --root, reads EVERY metadata.json it finds, builds a
filename -> label map, then emits one CSV row per matching .mp4 with a path
relative to --root (so pass the same --root as --root_dir to eval_crossdataset).

DFDC does not annotate which modality was manipulated, so modify_video /
modify_audio are left out (the AV model still gets real audio + real video).

Usage:
  python scripts/build_dfdc_csv.py --root datasets/DFDC_videos --out data/dfdc_metadata.csv
  # then:
  python scripts/eval_crossdataset.py --config configs/_ablation_generated/lavdf_talvad_v8gen__b2_cmis_couple__s42.yaml \
      --data_csv data/dfdc_metadata.csv --root_dir datasets/DFDC_videos --name dfdc
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_label_map(root: Path) -> dict[str, int]:
    """filename (lowercased) -> label (0 real / 1 fake), merged across all metadata.json."""
    label_map: dict[str, int] = {}
    n_files = 0
    for meta in root.rglob("metadata.json"):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: could not parse {meta}: {exc}")
            continue
        n_files += 1
        for fname, info in data.items():
            lbl = str(info.get("label", "")).upper()
            if lbl in ("FAKE", "REAL"):
                label_map[fname.lower()] = 1 if lbl == "FAKE" else 0
    print(f"  Read {n_files} metadata.json file(s) -> {len(label_map):,} labelled clips")
    return label_map


def main() -> None:
    ap = argparse.ArgumentParser(description="Build DFDC metadata CSV")
    ap.add_argument("--root", required=True, help="DFDC video root (contains part folders w/ metadata.json)")
    ap.add_argument("--out", default="data/dfdc_metadata.csv")
    ap.add_argument("--exts", nargs="+", default=[".mp4", ".avi", ".mov"])
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        ap.error(f"root not found: {root}")

    label_map = _load_label_map(root)
    if not label_map:
        ap.error(f"No usable metadata.json found under {root} "
                 f"(expected DFDC label files mapping filename -> REAL/FAKE).")

    exts = {e.lower() for e in args.exts}
    rows = []
    n_unlabelled = 0
    for f in root.rglob("*"):
        if f.suffix.lower() not in exts or not f.is_file():
            continue
        lbl = label_map.get(f.name.lower())
        if lbl is None:
            n_unlabelled += 1
            continue
        rows.append((f.relative_to(root).as_posix(), lbl))

    if not rows:
        ap.error(f"No videos matched a label. Checked exts {sorted(exts)} under {root}.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video_path", "label"])
        w.writerows(rows)

    n = len(rows)
    n_fake = sum(l for _, l in rows)
    n_real = n - n_fake
    print(f"Wrote {n:,} rows -> {out}")
    print(f"  real : {n_real:,}")
    print(f"  fake : {n_fake:,}  ({100*n_fake/n:.1f}%)")
    if n_unlabelled:
        print(f"  NOTE: {n_unlabelled:,} video file(s) had no entry in any metadata.json "
              f"(skipped) — check that the part's metadata.json is present.")


if __name__ == "__main__":
    main()
