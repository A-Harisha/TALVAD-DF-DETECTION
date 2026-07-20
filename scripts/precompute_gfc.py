"""
Pre-extract GFC (high-frequency spectral) features for all videos in the dataset.
Run this ONCE before training v5 so training doesn't stall extracting GFC on the fly.

Usage:
    python scripts/precompute_gfc.py --config configs/lavdf_talvad_v5.yaml
    python scripts/precompute_gfc.py --config configs/lavdf_talvad_v5.yaml --workers 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from datasets.builder import build_dataset_splits
from features.gfc import extract_gfc_features
from preprocessing.video import extract_frames_in_memory
from utils.config import load_config


def run(config_path: str, workers: int = 8):
    cfg = load_config(config_path)

    gfc_cfg = cfg.get("features", {}).get("gfc", {})
    if not gfc_cfg.get("enabled", False):
        print("GFC is not enabled in this config — nothing to do.")
        return

    n_bands = int(gfc_cfg.get("n_bands", 8))
    gfc_cache_dir = Path(cfg["dataset"].get("gfc_cache_dir", "./data/cache_v5_gfc"))
    gfc_cache_dir.mkdir(parents=True, exist_ok=True)
    fps = cfg.get("preprocessing", {}).get("fps", 2)
    frame_size = tuple(cfg.get("preprocessing", {}).get("frame_size", [224, 224]))

    train_df, val_df, test_df = build_dataset_splits(cfg)
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    total = len(full_df)

    rows = []
    for i, row in full_df.iterrows():
        sid = str(row.get("id", i))
        rows.append((sid, row.to_dict()))

    already_done = sum(1 for sid, _ in rows if (gfc_cache_dir / f"{sid}_gfc.npy").exists())

    print(f"GFC cache dir  : {gfc_cache_dir}")
    print(f"Bands          : {n_bands}")
    print(f"Total videos   : {total:,}")
    print(f"Already cached : {already_done:,}")
    print(f"To extract     : {total - already_done:,}")
    print(f"Workers        : {workers}")
    print()

    def extract_one(sid_row):
        sid, row = sid_row
        out_path = gfc_cache_dir / f"{sid}_gfc.npy"
        if out_path.exists():
            return "skipped"
        try:
            frames = extract_frames_in_memory(row["video_path"], fps=fps, frame_size=frame_size)
            gfc = extract_gfc_features(frames, n_bands=n_bands)
            np.save(out_path, gfc)
            return "ok"
        except Exception as e:
            return f"error: {e}"

    done, skipped, errors = 0, 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_one, r): r[0] for r in rows}
        with tqdm(total=total, desc="GFC extraction", unit="video") as pbar:
            for fut in as_completed(futures):
                status = fut.result()
                if status == "skipped":
                    skipped += 1
                elif status == "ok":
                    done += 1
                else:
                    errors += 1
                    tqdm.write(f"  [WARN] {futures[fut]}: {status}")
                pbar.update(1)
                pbar.set_postfix(done=done, skip=skipped, err=errors)

    print(f"\nExtracted: {done}  |  Skipped (cached): {skipped}  |  Errors: {errors}")
    print(f"Cache ready at: {gfc_cache_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lavdf_talvad_v5.yaml")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers (default 8, you have 28 CPU cores)")
    args = parser.parse_args()
    run(args.config, workers=args.workers)
