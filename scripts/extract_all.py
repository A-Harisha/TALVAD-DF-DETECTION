"""
One-time GPU-accelerated feature extractor.

Extracts MFCC+delta+delta2 (audio) + ResNet18 + GFC (visual) for every sample in
the metadata and saves the fused sequence [seq_len, feature_dim] as an individual
.npy file to data/features/<config_signature>/.

Run once before training:
    python scripts/extract_all.py --config configs/lavdf_talvad_v5.yaml

Re-running is safe — already-extracted files are skipped (idempotent).
After this completes, run pack_features.py to consolidate into one array.
"""
from __future__ import annotations

import os
# Set before numpy/librosa are imported — OpenBLAS reads these at first call.
# With our own ThreadPoolExecutor (28 threads), we don't want OpenBLAS to spawn
# additional internal threads. Setting to 1 makes each Python thread run BLAS
# operations inline (no internal thread pool needed → warning gone, no overhead).
os.environ["OPENBLAS_NUM_THREADS"] = "18"
os.environ["OMP_NUM_THREADS"]      = "18"
os.environ["MKL_NUM_THREADS"]      = "18"

import argparse
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from datasets.metadata import load_metadata
from features.audio_features import extract_mfcc_with_deltas
from features.cache import build_cache_signature, save_cached_feature
from features.fusion import FeatureFusionNormalizer, align_and_pack_sequences
from features.gfc import extract_gfc_features
from features.video_cnn import (
    _load_resnet18_feature_extractor,
    extract_resnet18_frame_features,
)
from preprocessing.audio import extract_audio_wav, preprocess_audio_waveform
from preprocessing.video import extract_frames_in_memory
from utils.config import load_config


# ──────────────────────────────────────────────────────────────────────────────
# Worker (runs in each thread)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_one(
    sample_id: str,
    video_path: str,
    out_dir: Path,
    cfg: dict,
    sequence_length: int,
    device: str,
) -> tuple[str, bool, str]:
    """
    Extract all features for one video and save to out_dir/sample_id_sequence.npy.
    Returns (sample_id, success, message).
    Skips if the output file already exists.
    """
    out_path = out_dir / f"{sample_id}_sequence.npy"
    if out_path.exists():
        return sample_id, True, "cached"

    try:
        pre      = cfg["preprocessing"]
        fps      = int(pre.get("fps", 2))
        frame_sz = tuple(pre.get("frame_size", [224, 224]))
        a_pre    = pre["audio"]
        a_feat   = cfg["features"]["audio"]
        cnn_cfg  = cfg["features"].get("cnn", {})
        gfc_cfg  = cfg["features"].get("gfc", {})

        # ── Video frames (in-memory — no disk writes for frames) ────────
        frames = extract_frames_in_memory(video_path, fps=fps, frame_size=frame_sz)

        # ── Audio: ffmpeg → temp wav → librosa → delete ─────────────────
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / f"{sample_id}.wav"
            extract_audio_wav(video_path, wav_path, a_pre["sample_rate"])
            y, sr = preprocess_audio_waveform(
                wav_path,
                sample_rate=a_pre["sample_rate"],
                preemph=a_pre.get("preemphasis", 0.97),
                wavelet_denoise=a_pre.get("wavelet_denoise", False),
            )

        # ── MFCC + delta + delta² ────────────────────────────────────────
        a_seq = extract_mfcc_with_deltas(y, sr, **a_feat)  # [T_a, 120]

        # ── ResNet18 frame embeddings (GPU) ──────────────────────────────
        v_seq = extract_resnet18_frame_features(
            frames,
            image_size=cnn_cfg.get("image_size", 224),
            batch_size=cnn_cfg.get("batch_size", 16),
            pretrained=cnn_cfg.get("pretrained", True),
            device=device,
        )  # [T_v, 512]

        # ── GFC frequency bands ──────────────────────────────────────────
        if gfc_cfg.get("enabled", False):
            n_bands = int(gfc_cfg.get("n_bands", 8))
            g_seq   = extract_gfc_features(frames, n_bands=n_bands)  # [T_v, 8]
            n_v     = min(len(v_seq), len(g_seq))
            v_seq   = np.concatenate([v_seq[:n_v], g_seq[:n_v]], axis=-1)  # [T_v, 520]

        # ── Fuse audio + video, pack to fixed sequence length ────────────
        per_sample_norm = cfg.get("features", {}).get("per_sample_norm", False)
        if per_sample_norm:
            fuser = FeatureFusionNormalizer()
            fused = fuser.fit_transform(fuser.fuse(a_seq, v_seq))
        else:
            fused = FeatureFusionNormalizer().fuse(a_seq, v_seq)  # [T, D]

        windows = align_and_pack_sequences(fused, sequence_length=sequence_length)
        seq = windows.mean(axis=0)  # [seq_len, feature_dim]

        save_cached_feature(out_path, seq)
        return sample_id, True, "extracted"

    except Exception:
        return sample_id, False, traceback.format_exc()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(config_path: str, workers: int = 0, cnn_batch_size: int = 0) -> None:
    """
    workers=0        → use all logical CPU cores (os.cpu_count())
    cnn_batch_size=0 → use config value (fall back to 64 if unset)
    """
    import os
    n_cpu   = os.cpu_count() or 1
    # Default to 12: enough to keep the GPU stream full without flooding it.
    # With a single CUDA stream, >12 threads just queue up and wait — adding
    # disk/CPU pressure with no GPU gain. Tune up/down with --workers.
    workers = workers if workers > 0 else min(12, n_cpu)

    cfg     = load_config(config_path)
    sig     = build_cache_signature(cfg)
    seq_len = int(cfg["training"].get("sequence_length", 12))
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Unlock GPU throughput ────────────────────────────────────────────
    # cuDNN auto-tunes kernels for fixed input sizes (all frames → 224×224).
    torch.backends.cudnn.benchmark = True
    # TF32 matmul on Ampere+ GPUs (RTX 30xx/40xx, A100) — ~2x faster,
    # negligible precision loss for feature extraction.
    torch.set_float32_matmul_precision("high")

    # Override CNN batch size for extraction (larger = better GPU utilisation)
    cnn_cfg = cfg["features"].get("cnn", {})
    effective_cnn_batch = cnn_batch_size if cnn_batch_size > 0 else cnn_cfg.get("batch_size", 64)
    # Patch into cfg so _extract_one picks it up
    cfg["features"].setdefault("cnn", {})["batch_size"] = effective_cnn_batch

    out_dir = Path("data/features") / sig
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load full metadata (all splits combined)
    ds_cfg = cfg["dataset"]
    df = load_metadata(ds_cfg["metadata_file"], ds_cfg["root_dir"], ds_cfg["name"])
    total  = len(df)

    # Count already-extracted files
    already = sum(
        1 for _, row in df.iterrows()
        if (out_dir / f"{str(row.get('id', ''))}_sequence.npy").exists()
    )
    pending = total - already

    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    W = 60
    print("=" * W)
    print(f"  extract_all.py")
    print(f"  Config    : {config_path}")
    print(f"  Signature : {sig}")
    print(f"  Output    : {out_dir}")
    print(f"  Device    : {device}  ({gpu_name})")
    print(f"  Workers   : {workers} threads  (all {n_cpu} logical cores)")
    print(f"  CNN batch : {effective_cnn_batch}  per thread")
    print(f"  cudnn     : benchmark=True  tf32={torch.backends.cuda.matmul.allow_tf32}")
    print(f"  Total     : {total:,} samples")
    print(f"  Cached    : {already:,}")
    print(f"  Pending   : {pending:,}")
    print("=" * W)

    if pending == 0:
        print("  All features already extracted — nothing to do.")
        print(f"  Next step: python scripts/pack_features.py --config {config_path}")
        return

    # Pre-warm ResNet in the main thread so lru_cache loads it once.
    # All worker threads share the same GPU-resident model via the cache.
    print(f"  Loading ResNet18 onto {device} …", flush=True)
    _load_resnet18_feature_extractor(cnn_cfg.get("pretrained", True)).to(device)
    print("  ResNet18 ready — starting extraction.\n")

    rows_list = [
        (str(row.get("id", i)), str(row["video_path"]))
        for i, row in df.iterrows()
    ]

    t_start  = time.perf_counter()
    n_done   = 0
    n_errors = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_extract_one, sid, vpath, out_dir, cfg, seq_len, device): sid
            for sid, vpath in rows_list
        }
        with tqdm(
            total=total,
            unit="video",
            desc="  Extracting",
            dynamic_ncols=True,
            smoothing=0.05,
        ) as pbar:
            for fut in as_completed(futures):
                sid, ok, msg = fut.result()
                pbar.update(1)

                if not ok:
                    n_errors += 1
                elif msg == "extracted":
                    n_done += 1

                elapsed = time.perf_counter() - t_start
                if n_done > 0 and elapsed > 0:
                    pbar.set_postfix(
                        speed=f"{n_done / elapsed:.1f}/s",
                        errors=n_errors,
                        refresh=False,
                    )

    elapsed = time.perf_counter() - t_start

    print()
    print("=" * W)
    print(f"  Extraction complete")
    print(f"  Already cached  : {already:,}")
    print(f"  Newly extracted : {n_done:,}")
    print(f"  Errors          : {n_errors}")
    print(f"  Throughput      : {max(n_done, 1) / max(elapsed, 1):.1f} videos/sec")
    print(f"  Wall time       : {int(elapsed // 3600)}h {int(elapsed % 3600 // 60):02d}m {int(elapsed % 60):02d}s")
    print(f"  Output dir      : {out_dir}")
    print("=" * W)

    if n_errors:
        err_log = out_dir / "extraction_errors.log"
        failed_sids = [
            futures[fut] for fut in futures
            if not fut.exception() and fut.result()[1] is False
        ]
        err_log.write_text(
            "\n".join(f"{sid}" for sid in failed_sids),
            encoding="utf-8",
        )
        print(f"\n  Error list saved to {err_log}")
        print("  Re-run this script to retry failed samples (idempotent).")

    print(f"\n  Next step: python scripts/pack_features.py --config {config_path}")


if __name__ == "__main__":
    import os as _os
    parser = argparse.ArgumentParser(
        description="One-time GPU-accelerated feature extraction for all dataset samples"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=f"Parallel threads (0 = all CPU cores, currently {_os.cpu_count()})",
    )
    parser.add_argument(
        "--cnn_batch_size",
        type=int,
        default=0,
        help="ResNet batch size per thread (0 = use config value, default 64)",
    )
    args = parser.parse_args()
    run(args.config, workers=args.workers, cnn_batch_size=args.cnn_batch_size)
