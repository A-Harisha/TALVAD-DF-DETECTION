"""
Pack individual .npy feature files into a single array for fast training startup.

Run after extract_all.py:
    python scripts/pack_features.py --config configs/lavdf_talvad_v5.yaml

Creates in data/packed/<signature>/:
    features.npy   — float32 [N, seq_len, feature_dim]  (~7 GB for LAV-DF)
    labels.npy     — int32   [N]
    ids.json       — {"sample_id": packed_index}  O(1) lookup at training time
    manifest.json  — provenance metadata

At training time, train.py detects this directory and loads the entire feature
array in one shot (~5–10 s from NVMe), skipping the per-sample .npy loop entirely.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from datasets.metadata import load_metadata
from features.cache import build_cache_signature
from utils.config import load_config


def run(config_path: str, force: bool = False) -> None:
    cfg = load_config(config_path)
    sig = build_cache_signature(cfg)

    src_dir = Path("data/features") / sig
    out_dir = Path("data/packed")  / sig
    out_dir.mkdir(parents=True, exist_ok=True)

    features_path = out_dir / "features.npy"
    labels_path   = out_dir / "labels.npy"
    ids_path      = out_dir / "ids.json"
    manifest_path = out_dir / "manifest.json"

    if features_path.exists() and not force:
        print(f"Packed features already exist at {out_dir}")
        print("Pass --force to rebuild.")
        return

    if not src_dir.exists():
        print(f"ERROR: source dir {src_dir} does not exist.")
        print(f"Run:  python scripts/extract_all.py --config {config_path}")
        return

    # ── Load metadata to get labels and canonical ordering ───────────────
    ds_cfg = cfg["dataset"]
    df = load_metadata(ds_cfg["metadata_file"], ds_cfg["root_dir"], ds_cfg["name"])

    # ── Find which samples have been extracted ───────────────────────────
    available: list[tuple[str, int, Path]] = []  # (id_str, label, npy_path)
    missing:   list[str]                   = []

    for i, row in df.iterrows():
        sid  = str(row.get("id", i))
        path = src_dir / f"{sid}_sequence.npy"
        if path.exists():
            available.append((sid, int(row["label"]), path))
        else:
            missing.append(sid)

    total = len(available)

    W = 60
    print("=" * W)
    print(f"  pack_features.py")
    print(f"  Config    : {config_path}")
    print(f"  Signature : {sig}")
    print(f"  Source    : {src_dir}")
    print(f"  Output    : {out_dir}")
    print(f"  Available : {total:,} samples")
    if missing:
        print(f"  Missing   : {len(missing):,} samples  (run extract_all.py to fill gaps)")
    print("=" * W)

    if total == 0:
        print("ERROR: no feature files found in source dir.")
        print(f"Run:  python scripts/extract_all.py --config {config_path}")
        return

    # Probe shape from first file
    probe               = np.load(available[0][2])
    seq_len, feat_dim   = probe.shape
    size_gb             = total * seq_len * feat_dim * 4 / 1e9

    print(f"  Feature shape : ({seq_len}, {feat_dim})  per sample")
    print(f"  Packed size   : {size_gb:.2f} GB  (float32)")

    # ── Allocate output arrays ───────────────────────────────────────────
    features   = np.empty((total, seq_len, feat_dim), dtype=np.float32)
    labels     = np.empty(total, dtype=np.int32)
    id_to_idx: dict[str, int] = {}

    # ── Load and pack ────────────────────────────────────────────────────
    print(f"\n  Loading {total:,} files …")
    t0 = time.perf_counter()

    for idx, (sid, label, npy_path) in enumerate(
        tqdm(available, unit="file", dynamic_ncols=True)
    ):
        features[idx] = np.load(npy_path)
        labels[idx]   = label
        id_to_idx[sid] = idx

    load_elapsed = time.perf_counter() - t0
    print(f"  Loaded in {load_elapsed:.1f}s  ({total / load_elapsed:.0f} files/sec)")

    # ── Save atomically (write to .tmp then rename) ──────────────────────
    print(f"\n  Saving features.npy ({size_gb:.2f} GB) …", flush=True)
    t0   = time.perf_counter()
    tmp  = out_dir / "features_tmp.npy"
    np.save(tmp, features)
    tmp.replace(features_path)
    save_elapsed = time.perf_counter() - t0
    print(f"  Saved in {save_elapsed:.1f}s  ({size_gb / save_elapsed:.2f} GB/s)")

    print("  Saving labels.npy …")
    np.save(labels_path, labels)

    print("  Saving ids.json …")
    ids_path.write_text(json.dumps(id_to_idx), encoding="utf-8")

    # ── Manifest ─────────────────────────────────────────────────────────
    unique, counts   = np.unique(labels, return_counts=True)
    label_counts     = {str(int(k)): int(v) for k, v in zip(unique, counts)}
    manifest = {
        "config":          config_path,
        "signature":       sig,
        "total_samples":   total,
        "missing_samples": len(missing),
        "seq_len":         seq_len,
        "feature_dim":     feat_dim,
        "label_counts":    label_counts,
        "size_gb":         round(size_gb, 3),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print()
    print("=" * W)
    print(f"  Pack complete")
    print(f"  features.npy : {features_path.stat().st_size / 1e9:.2f} GB")
    print(f"  Samples      : {total:,}   labels: {label_counts}")
    print(f"  Output dir   : {out_dir}")
    print("=" * W)
    print(f"\n  Training will now use packed features automatically.")
    print(f"  Run:  python -m training.train --config {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pack individual .npy feature files into a single array"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild packed file even if it already exists",
    )
    args = parser.parse_args()
    run(args.config, force=args.force)
