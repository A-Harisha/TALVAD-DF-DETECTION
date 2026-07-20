"""scripts/split_index.py — split a min-index json into N DISJOINT shards so the
`main` extraction stage can run as N parallel workers (main is CPU/IO-bound: video
decode + face detect starve the GPU, so more CPU workers = faster). Shards are
round-robin over sorted clip ids => each shard gets a balanced clip-length mix.
All workers write to the SAME raw_main dir; clip ids are unique so no collision,
and already-cached clips are skipped on restart.

  .venv\\Scripts\\python.exe scripts\\split_index.py --index data\\df1m_val_zeroshot.min.json --n 2
  -> data\\df1m_val_zeroshot.part1of2.json , ...part2of2.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--n", type=int, required=True, help="number of shards")
    args = ap.parse_args()

    src = Path(args.index)
    idx = json.loads(src.read_text(encoding="utf-8"))
    cids = sorted(idx)                                   # deterministic order
    shards = [dict() for _ in range(args.n)]
    for i, cid in enumerate(cids):
        shards[i % args.n][cid] = idx[cid]              # round-robin

    stem = src.stem.replace(".min", "")
    for k, sh in enumerate(shards, 1):
        out = src.with_name(f"{stem}.part{k}of{args.n}.json")
        out.write_text(json.dumps(sh), encoding="utf-8")
        print(f"  part {k}/{args.n}: {len(sh):>6} clips -> {out}")
    print(f"total {len(cids)} clips split into {args.n} disjoint shards")


if __name__ == "__main__":
    main()
