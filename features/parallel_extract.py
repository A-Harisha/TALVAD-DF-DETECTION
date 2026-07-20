from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm


def get_optimal_workers(reserve: int = 2) -> int:
    """Returns cpu_count - reserve, minimum 1. No upper cap."""
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - reserve)


def _sample_id_from_row(row_dict: dict) -> str:
    return str(row_dict.get("id", Path(row_dict["video_path"]).stem))


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _extract_worker(args: tuple) -> tuple[str, bool, str]:
    """
    Top-level worker. args = (sample_id, video_path, work_dir_str, cache_dir_str, cfg_dict, sequence_length)
    Returns (sample_id, success: bool, error_message: str)
    Must import everything it needs locally (spawn safety).
    Writes atomically: tmp_path -> os.replace -> final_path
    Skips if cache file already exists (idempotent).
    """
    sample_id, video_path, work_dir_str, cache_dir_str, cfg_dict, sequence_length = args
    try:
        from pathlib import Path

        import numpy as np

        from features.audio_features import extract_mfcc_with_deltas
        from features.fusion import FeatureFusionNormalizer, align_and_pack_sequences
        from features.video_cnn import extract_resnet18_frame_features
        from preprocessing.pipeline import process_video_sample

        cache_dir = Path(cache_dir_str)
        work_dir = Path(work_dir_str)
        final_path = cache_dir / f"{sample_id}_sequence.npy"
        if final_path.exists():
            return sample_id, True, ""

        final_path.parent.mkdir(parents=True, exist_ok=True)
        preprocessed_dir = work_dir / "preprocessed" / sample_id
        y, sr, frames = process_video_sample(video_path, preprocessed_dir, cfg_dict)

        audio_cfg = cfg_dict["features"]["audio"]
        a_seq = extract_mfcc_with_deltas(y, sr, **audio_cfg)

        cnn_cfg = cfg_dict["features"].get("cnn", {})
        v_seq = extract_resnet18_frame_features(
            frames,
            image_size=cnn_cfg.get("image_size", 224),
            batch_size=cnn_cfg.get("batch_size", 16),
            pretrained=cnn_cfg.get("pretrained", True),
            device="cpu",
        )

        per_sample_norm = cfg_dict.get("features", {}).get("per_sample_norm", True)
        if per_sample_norm:
            fuser = FeatureFusionNormalizer()
            fused = fuser.fit_transform(fuser.fuse(a_seq, v_seq))
        else:
            fused = FeatureFusionNormalizer().fuse(a_seq, v_seq)
        windows = align_and_pack_sequences(fused, sequence_length=sequence_length)
        seq = windows.mean(axis=0)

        tmp_path = final_path.with_name(final_path.name + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        with tmp_path.open("wb") as tmp_file:
            np.save(tmp_file, seq)
        os.replace(tmp_path, final_path)
        return sample_id, True, ""
    except Exception as exc:
        return sample_id, False, traceback.format_exc()


def extract_single(row: dict, cache_dir, cfg: dict, work_dir, sequence_length: int) -> bool:
    """Single-process extraction for one row. Returns True on success."""
    sample_id = _sample_id_from_row(row)
    _, success, _ = _extract_worker((sample_id, row["video_path"], str(work_dir), str(cache_dir), cfg, sequence_length))
    return success


def parallel_extract_features(
    df,
    cfg: dict,
    work_dir,
    cache_dir,
    sequence_length: int,
    n_workers: int = 0,
    checkpoint_every: int = 500,
    show_progress: bool = True,
    log_dir: Path = Path("logs"),
) -> dict:
    """
    Extract features for all rows using ProcessPoolExecutor.
    Returns {total, completed, skipped, failed, elapsed_seconds, workers_used}
    """
    cache_dir = Path(cache_dir)
    work_dir = Path(work_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    status_log_path = log_dir / "extraction_status.log"
    error_log_path = log_dir / "extraction_errors.log"

    if n_workers <= 0:
        n_workers = get_optimal_workers()
    workers_used = max(1, min(int(n_workers), os.cpu_count() or 1))

    records = [dict(row) for row in df.to_dict(orient="records")]
    total = len(records)
    already_cached: list[tuple[str, str]] = []
    pending: list[tuple[str, str]] = []
    for row in records:
        sample_id = _sample_id_from_row(row)
        video_path = row["video_path"]
        final_path = cache_dir / f"{sample_id}_sequence.npy"
        if final_path.exists():
            already_cached.append((sample_id, video_path))
        else:
            pending.append((sample_id, video_path))

    to_extract = len(pending)
    print(f"[Extraction] {len(already_cached)} already cached, {to_extract} to extract")
    print(f"using {workers_used} workers (i7-14700: 16 P-core threads)")

    start = time.perf_counter()
    errors = 0
    completed_run = 0
    cache_hits = len(already_cached)
    # Open log files so worker loop can write status/errors
    with status_log_path.open("a", encoding="utf-8") as status_log, error_log_path.open("a", encoding="utf-8") as error_log:
        with ProcessPoolExecutor(max_workers=workers_used) as executor:
            future_map = {
                executor.submit(_extract_worker, (sample_id, video_path, str(work_dir), str(cache_dir), cfg, sequence_length)): sample_id
                for sample_id, video_path in pending
            }
            progress = tqdm(total=to_extract, unit="video", desc="Extracting features") if show_progress else None
            try:
                for future in as_completed(future_map):
                    sample_id = future_map[future]
                    try:
                        returned_sample_id, success, error_message = future.result()
                    except Exception as exc:
                        returned_sample_id, success, error_message = sample_id, False, str(exc)

                    completed_run += 1
                    if progress is not None:
                        progress.update(1)
                    if not success:
                        errors += 1
                        line = f"[{datetime.now().strftime('%H:%M:%S')}] FAILED sample_id={returned_sample_id} | {error_message}"
                        print(line)
                        error_log.write(line + "\n")
                        error_log.flush()

                    if completed_run % checkpoint_every == 0 or completed_run == to_extract:
                        elapsed = time.perf_counter() - start
                        speed = completed_run / elapsed if elapsed > 0 else 0.0
                        remaining = max(to_extract - completed_run, 0)
                        eta_seconds = remaining / speed if speed > 0 else 0.0
                        status_line = (
                            f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {completed_run}/{to_extract} | "
                            f"speed={speed:.1f} vid/s | ETA={_format_duration(eta_seconds)} | errors={errors} | cache_hits={cache_hits}"
                        )
                        print(status_line)
                        status_log.write(status_line + "\n")
                        status_log.flush()
            finally:
                if progress is not None:
                    progress.close()

    elapsed_seconds = time.perf_counter() - start
    extracted = completed_run
    speed = extracted / elapsed_seconds if elapsed_seconds > 0 else 0.0

    print("╔══════════════════════════════════════════════════════╗")
    print("║  EXTRACTION COMPLETE                                 ║")
    print(f"║  Already cached : {len(already_cached):>8}                          ║")
    print(f"║  Newly extracted: {extracted:>8}                           ║")
    print(f"║  Errors         : {errors:>8}  (see logs/ if any)       ║")
    print(f"║  Workers used   : {workers_used:>8}  (i7-14700 P-cores)       ║")
    print(f"║  Throughput     : {speed:>8.1f} videos/sec              ║")
    print(f"║  Wall time      : {_format_duration(elapsed_seconds):>12}                       ║")
    print("╚══════════════════════════════════════════════════════╝")

    status_path = log_dir / "extraction_status.json"
    status_payload = {
        "status": "complete",
        "total": total,
        "completed": total,
        "cache_hits": len(already_cached),
        "extracted": extracted,
        "errors": errors,
        "elapsed_seconds": int(round(elapsed_seconds)),
        "workers_used": workers_used,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    status_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")

    return {
        "total": total,
        "cache_hits": len(already_cached),
        "extracted": extracted,
        "errors": errors,
        "elapsed_seconds": int(round(elapsed_seconds)),
        "throughput": speed,
        "workers_used": workers_used,
    }