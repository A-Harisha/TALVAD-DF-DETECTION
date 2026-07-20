"""scripts/build_df1m_val_zeroshot_index.py — annotation index for the ORIGINAL
AV-Deepfake1M (DF1M) validation split, for a zero-shot eval of the ++-trained
TALVAD-L checkpoint.

DF1M's val_metadata.json uses the exact schema datasets.avdeepfake1m_loader already
parses (file / modify_type / *_segments). We build the id->segments index, keep only
clips whose .mp4 is actually on disk under <val_root>, and stamp split="val" so the
extractor/eval route them to the "dev" split (build_index drops the metadata's own
`split` field, and the DF1M `file` paths have no "val/" prefix for _split_of to catch).

  .venv\\Scripts\\python.exe scripts\\build_df1m_val_zeroshot_index.py \\
      --meta E:\\AV-Deepfake1M\\val_metadata.json \\
      --val-root F:\\AV-Deepfake1M\\val_data\\val \\
      --out data\\df1m_val_zeroshot.min.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from datasets.avdeepfake1m_loader import build_index, df1m_clip_id  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=r"E:\AV-Deepfake1M\val_metadata.json")
    ap.add_argument("--val-root", default=r"F:\AV-Deepfake1M\val_data\val")
    ap.add_argument("--out", default=str(REPO / "data" / "df1m_val_zeroshot.min.json"))
    args = ap.parse_args()

    val_root = Path(args.val_root)
    full = build_index(args.meta)                       # {cid: {...}} for every val record
    print(f"metadata records parsed: {len(full)}")

    # cids present on disk (one rglob, then match by canonical id)
    on_disk = {df1m_clip_id(p.relative_to(val_root).as_posix()): p
               for p in val_root.rglob("*.mp4")}
    print(f"mp4 files on disk under val-root: {len(on_disk)}")

    present, missing_meta = {}, 0
    for cid in on_disk:
        e = full.get(cid)
        if e is None:
            missing_meta += 1
            continue
        e = dict(e); e["split"] = "val"                 # -> "dev" downstream
        present[cid] = e

    labels = [e["label"] for e in present.values()]
    n_fake_seg = sum(1 for e in present.values() if e["fake_periods"])
    print(f"indexed present clips : {len(present)}  "
          f"(real={labels.count(0)} fake={labels.count(1)}; {n_fake_seg} have fake_periods)")
    print(f"on-disk without metadata match: {missing_meta}")
    if len(full) - len(present) > 0:
        print(f"metadata clips NOT on disk (skipped): {len(full) - (len(present))}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(present), encoding="utf-8")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
