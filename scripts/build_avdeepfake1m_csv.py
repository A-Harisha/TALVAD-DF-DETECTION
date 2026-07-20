"""
scripts/build_avdeepfake1m_csv.py

Build a metadata CSV for AV-Deepfake1M (a.k.a. "DeepFake 1M" / 1M-Deepfakes
Detection Challenge) that the existing cross-dataset pipeline can consume:

    build_avdeepfake1m_csv.py  ->  <video_path,label,modify_video,modify_audio> CSV
    extract_talvad_x.py --target_csv ...  ->  feature pack
    eval_crossdataset_x.py --target_name avdeepfake1m  ->  AUC / EER

Unlike FakeAVCeleb (labels encoded in folder names), AV-Deepfake1M ships labels
in a JSON metadata file (train_metadata.json / val_metadata.json), one record per
clip. Relevant fields (verbatim from the official repo):

    file                 : video file path (relative to dataset root)
    modify_type          : "real" | "visual_modified" | "audio_modified" | "both_modified"
    fake_segments        : [[start,end], ...]  (combined fake intervals; may be absent for real)
    audio_fake_segments  : audio-modality intervals
    visual_fake_segments  : visual-modality intervals
    audio_model          : generator id (kept for optional stratification)

This script reads that JSON and writes the SAME 4-column CSV schema that
build_fakeavceleb_csv.py emits, so nothing downstream changes. Because the full
set is multiple TB, the default behaviour is to sample a BALANCED subset:
--n_per_class reals + an equal number of fakes, with fakes stratified across the
three fake types (visual / audio / both) so the dub-vs-swap-vs-both breakdown is
preserved for per-type analysis.

NOTE: This is staged ahead of having the data. The metadata field names are taken
from the official README; if a real metadata file differs, only the FIELD_* names
below need adjusting.

Usage (once the val subset is downloaded under <root>):
  python scripts/build_avdeepfake1m_csv.py \
      --metadata /path/to/AV-Deepfake1M/val_metadata.json \
      --root /path/to/AV-Deepfake1M \
      --out data/avdeepfake1m_metadata.csv \
      --n_per_class 1500 --seed 42
  # then:
  python scripts/extract_talvad_x.py --config configs/lavdf_talvad_x.yaml \
      --target_csv data/avdeepfake1m_metadata.csv --target_name avdeepfake1m \
      --root_dir /path/to/AV-Deepfake1M
  python scripts/eval_crossdataset_x.py --config configs/lavdf_talvad_x.yaml \
      --target_name avdeepfake1m
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# --- metadata field names (adjust here if a real file differs) ---------------
FIELD_FILE = "file"
FIELD_MODIFY = "modify_type"
FIELD_FAKE_SEGMENTS = "fake_segments"

# modify_type -> (label, modify_video, modify_audio)
MODIFY_MAP = {
    "real": (0, 0, 0),
    "visual_modified": (1, 1, 0),
    "audio_modified": (1, 0, 1),
    "both_modified": (1, 1, 1),
}
# per-row fake "type bucket" used only for stratified subset sampling
TYPE_BUCKET = {
    "visual_modified": "fake_video",
    "audio_modified": "fake_audio",
    "both_modified": "fake_both",
}


def load_records(meta_path: Path) -> list[dict]:
    """AV-Deepfake1M metadata may be a JSON list, or a dict keyed by id."""
    with open(meta_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        # dict-of-records -> list, injecting the key as 'file' fallback
        recs = []
        for k, v in data.items():
            if isinstance(v, dict):
                v.setdefault(FIELD_FILE, k)
                recs.append(v)
        return recs
    return list(data)


def classify(rec: dict) -> tuple[int, int, int, str]:
    raw = str(rec.get(FIELD_MODIFY, "")).strip().lower()
    if raw not in MODIFY_MAP:
        return -1, 0, 0, "unknown"
    label, mv, ma = MODIFY_MAP[raw]
    bucket = "real" if label == 0 else TYPE_BUCKET[raw]
    return label, mv, ma, bucket


def main() -> None:
    ap = argparse.ArgumentParser(description="Build AV-Deepfake1M metadata CSV (balanced subset)")
    ap.add_argument("--metadata", required=True, help="path to train/val_metadata.json")
    ap.add_argument("--root", required=True, help="dataset root (paths written relative to this)")
    ap.add_argument("--out", default="data/avdeepfake1m_metadata.csv")
    ap.add_argument("--n_per_class", type=int, default=1500,
                    help="reals to keep; an equal number of fakes is sampled (stratified). "
                         "Use 0 to keep ALL rows (no subsetting).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require_exists", action="store_true",
                    help="only keep rows whose video file is present under --root")
    args = ap.parse_args()

    meta_path = Path(args.metadata)
    root = Path(args.root)
    if not meta_path.exists():
        ap.error(f"metadata not found: {meta_path}")

    rng = random.Random(args.seed)
    recs = load_records(meta_path)

    by_bucket: dict[str, list[tuple[str, int, int, int]]] = defaultdict(list)
    unknown = 0
    for rec in recs:
        rel = str(rec.get(FIELD_FILE, "")).replace("\\", "/").lstrip("/")
        if not rel:
            continue
        label, mv, ma, bucket = classify(rec)
        if label < 0:
            unknown += 1
            continue
        if args.require_exists and not (root / rel).exists():
            continue
        by_bucket[bucket].append((rel, label, mv, ma))

    n_real_avail = len(by_bucket["real"])
    fake_buckets = ["fake_video", "fake_audio", "fake_both"]
    n_fake_avail = sum(len(by_bucket[b]) for b in fake_buckets)
    if n_real_avail == 0 or n_fake_avail == 0:
        ap.error(f"need both classes; got real={n_real_avail} fake={n_fake_avail} "
                 f"(unknown modify_type rows={unknown}). Check FIELD_* / MODIFY_MAP.")

    # ---- select rows ----
    if args.n_per_class and args.n_per_class > 0:
        target = min(args.n_per_class, n_real_avail, n_fake_avail)
        reals = by_bucket["real"][:]
        rng.shuffle(reals)
        reals = reals[:target]

        # stratify fakes across the three buckets as evenly as availability allows
        fakes: list[tuple[str, int, int, int]] = []
        per = target // 3
        for b in fake_buckets:
            rows = by_bucket[b][:]
            rng.shuffle(rows)
            fakes.extend(rows[:per])
        # top up to exactly `target` from whatever remains
        if len(fakes) < target:
            remaining = [r for b in fake_buckets for r in by_bucket[b] if r not in set(fakes)]
            rng.shuffle(remaining)
            fakes.extend(remaining[: target - len(fakes)])
        selected = reals + fakes
    else:
        selected = [r for rows in by_bucket.values() for r in rows]

    rng.shuffle(selected)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video_path", "label", "modify_video", "modify_audio"])
        w.writerows(selected)

    # ---- report ----
    cnt = Counter()
    for rel, label, mv, ma in selected:
        if label == 0:
            cnt["real"] += 1
        elif mv and ma:
            cnt["fake_both"] += 1
        elif mv:
            cnt["fake_video"] += 1
        elif ma:
            cnt["fake_audio"] += 1
    n = len(selected)
    print(f"Wrote {n:,} rows -> {out}")
    print(f"  available: real={n_real_avail:,}  fake={n_fake_avail:,}  unknown_type={unknown:,}")
    print(f"  selected real         : {cnt['real']:,}")
    print(f"  selected fake (video) : {cnt['fake_video']:,}")
    print(f"  selected fake (audio) : {cnt['fake_audio']:,}  <- the dub-type the prior targets")
    print(f"  selected fake (both)  : {cnt['fake_both']:,}")
    print(f"  fake total            : {n - cnt['real']:,}")


if __name__ == "__main__":
    main()
