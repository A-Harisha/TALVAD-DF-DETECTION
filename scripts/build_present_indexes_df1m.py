"""scripts/build_present_indexes_df1m.py — wire the DF1++ 50k "present" training scope.

The downloaded/unzipped subset is 50,424 train clips (train_extracted/train, matched by
df1m_present_index.min.json) + 1,500 dev clips (val_extracted/val). This script:

  1. builds a DEV index (df1m_present_val.min.json) = val-split entries whose mp4 is
     actually on disk, each tagged split="val" (-> "dev" via extractor _split_of),
  2. writes a COMBINED present-all index (df1m_present_all.min.json) = the 50,424 train
     entries (split="train") + the dev entries, used by the `pack` stage to assign
     manifest splits and by train_talvad_l_df1m.py as the GT-segments source.

cids are df1m_clip_id(relative_path) — identical to the ids the extractor uses to name
per-clip features, so train/pack/eval all line up.

  .venv\\Scripts\\python.exe scripts\\build_present_indexes_df1m.py [--df-root F:\\AV-Deepfake1M-PlusPlus]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from datasets.avdeepfake1m_loader import df1m_clip_id  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--df-root", default=r"F:\AV-Deepfake1M-PlusPlus",
                    help="DF1++ dataset root (holds train_extracted/ and val_extracted/)")
    ap.add_argument("--data", default=str(REPO / "data"), help="repo data/ dir with the index jsons")
    args = ap.parse_args()

    data = Path(args.data)
    val_root = Path(args.df_root) / "val_extracted" / "val"

    train_present = json.loads((data / "df1m_present_index.min.json").read_text())  # 50,424 (train)
    val_full = json.loads((data / "df1m_val_index.min.json").read_text())            # 77,326 (val)

    # DEV = val-index entries whose mp4 is on disk under val_extracted/val
    on_disk = list(val_root.rglob("*.mp4"))
    val_present, missing = {}, 0
    for p in on_disk:
        rel = p.relative_to(val_root).as_posix()
        e = val_full.get(df1m_clip_id(rel))
        if e is None:
            missing += 1
            continue
        e = dict(e); e["split"] = "val"          # -> "dev"
        val_present[df1m_clip_id(rel)] = e
    print(f"val on-disk mp4: {len(on_disk)} | matched in val index: {len(val_present)} | unmatched: {missing}")

    for e in train_present.values():
        e["split"] = "train"

    collisions = set(train_present) & set(val_present)
    print(f"train cids: {len(train_present)} | dev cids: {len(val_present)} | collisions: {len(collisions)}")
    if collisions:
        print("  WARNING: dropping colliding cids from dev to keep ids unique")
        for c in collisions:
            val_present.pop(c, None)

    (data / "df1m_present_val.min.json").write_text(json.dumps(val_present))
    combined = dict(train_present); combined.update(val_present)
    (data / "df1m_present_all.min.json").write_text(json.dumps(combined))

    labels = [e["label"] for e in combined.values()]
    print(f"dev index           -> {data/'df1m_present_val.min.json'}  ({len(val_present)} clips)")
    print(f"combined present-all -> {data/'df1m_present_all.min.json'}  "
          f"({len(combined)} clips: real={labels.count(0)} fake={labels.count(1)})")


if __name__ == "__main__":
    main()
