from __future__ import annotations

import os
# Must be set before numpy/scipy/librosa are imported.
# OpenBLAS on this machine is compiled with NUM_THREADS=24; cap it there.
os.environ["OPENBLAS_NUM_THREADS"] = "18"
os.environ["OMP_NUM_THREADS"]      = "18"
os.environ["MKL_NUM_THREADS"]      = "18"

import argparse
from pathlib import Path
import json
import sys
import time

# Add repo root to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets.builder import build_dataset_splits
from datasets.sequence_dataset import MultiModalSequenceDataset, PackedSequenceDataset
from features.audio_features import audio_feature_dim
from features.audio_features import extract_mfcc_with_deltas
from features.cache import build_cache_signature, load_cached_feature, save_cached_feature
from features.fusion import FeatureFusionNormalizer, align_and_pack_sequences
from features.gfc import extract_gfc_features
from features.parallel_extract import get_optimal_workers, parallel_extract_features
from features.video_cnn import (
    CNN_FEATURE_DIM,
    CNN_FEATURE_DIM_MULTISCALE,
    extract_multiscale_resnet50_frame_features,
    extract_resnet18_frame_features,
)
from models.bigru_detector import BiGRUSoftplusDetector
from optimization.fitness import FitnessEvaluator
from optimization.gbwso import GBWSOFeatureSelector
from preprocessing.pipeline import process_video_sample
from preprocessing.video import extract_frames_in_memory
from training.trainer import Trainer
from utils.config import load_config
from utils.reproducibility import seed_everything


def _load_or_compute_gfc(
    sample_id: str,
    row: dict,
    cfg: dict,
    work_dir: Path,
    sequence_length: int,
    n_frames_in_base: int,
) -> np.ndarray | None:
    """Load GFC features from cache or compute from raw frames (CPU-only, fast)."""
    gfc_cfg = cfg.get("features", {}).get("gfc", {})
    if not gfc_cfg.get("enabled", False):
        return None

    n_bands = int(gfc_cfg.get("n_bands", 8))
    gfc_cache_dir = Path(cfg["dataset"].get("gfc_cache_dir", str(work_dir / "gfc_cache")))
    gfc_cache_dir.mkdir(parents=True, exist_ok=True)
    gfc_path = gfc_cache_dir / f"{sample_id}_gfc.npy"

    if gfc_path.exists():
        gfc = np.load(gfc_path)
    else:
        fps = cfg.get("preprocessing", {}).get("fps", 2)
        frame_size = tuple(cfg.get("preprocessing", {}).get("frame_size", [224, 224]))
        frames = extract_frames_in_memory(row["video_path"], fps=fps, frame_size=frame_size)
        gfc = extract_gfc_features(frames, n_bands=n_bands)  # [T_frames, n_bands]
        np.save(gfc_path, gfc)

    # Align to the same number of timesteps as the base feature sequence
    if len(gfc) == 0:
        return np.zeros((sequence_length, n_bands), dtype=np.float32)

    # Resample GFC to match sequence_length via linear interpolation
    if len(gfc) != sequence_length:
        indices = np.linspace(0, len(gfc) - 1, sequence_length)
        lo = np.floor(indices).astype(int)
        hi = np.minimum(lo + 1, len(gfc) - 1)
        frac = (indices - lo)[:, None]
        gfc = gfc[lo] * (1 - frac) + gfc[hi] * frac

    return gfc.astype(np.float32)   # [sequence_length, n_bands]


def _ensure_gfc_cache(df, cfg, workers: int = 8):
    """Auto-precompute any missing GFC files before training. Skips if GFC is disabled."""
    gfc_cfg = cfg.get("features", {}).get("gfc", {})
    if not gfc_cfg.get("enabled", False):
        return

    n_bands = int(gfc_cfg.get("n_bands", 8))
    gfc_cache_dir = Path(cfg["dataset"].get("gfc_cache_dir", "./data/cache_v5_gfc"))
    gfc_cache_dir.mkdir(parents=True, exist_ok=True)

    fps = cfg.get("preprocessing", {}).get("fps", 2)
    frame_size = tuple(cfg.get("preprocessing", {}).get("frame_size", [224, 224]))

    pending = []
    for i, row in df.iterrows():
        sid = str(row.get("id", i))
        if not (gfc_cache_dir / f"{sid}_gfc.npy").exists():
            pending.append((sid, row.to_dict()))

    total = len(df)
    cached = total - len(pending)
    if not pending:
        print(f"  GFC cache   : all {total:,} files ready — skipping extraction")
        return

    print(f"  GFC cache   : {cached:,} / {total:,} ready — extracting {len(pending):,} missing files ({workers} threads) …")

    def extract_one(sid_row):
        sid, row = sid_row
        out_path = gfc_cache_dir / f"{sid}_gfc.npy"
        if out_path.exists():
            return True
        try:
            frames = extract_frames_in_memory(row["video_path"], fps=fps, frame_size=frame_size)
            gfc = extract_gfc_features(frames, n_bands=n_bands)
            np.save(out_path, gfc)
            return True
        except Exception:
            return False

    from tqdm import tqdm
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_one, r): r[0] for r in pending}
        with tqdm(total=len(pending), desc="  GFC extract", unit="video") as pbar:
            for fut in as_completed(futures):
                if fut.result():
                    done += 1
                pbar.update(1)
    print(f"  GFC extract : complete ({done:,} extracted)")


def _build_features_for_row(row, cfg, work_dir: Path, sequence_length: int, cache_dir: Path | None = None):
    sample_id = str(row.get("id", Path(row["video_path"]).stem))
    seq_cache = work_dir / "feature_cache" / f"{sample_id}_sequence.npy"
    fused_cache = load_cached_feature(seq_cache)
    if fused_cache is None and cache_dir is not None:
        fused_cache = load_cached_feature(cache_dir / f"{sample_id}_sequence.npy")

    if fused_cache is not None:
        gfc_cfg = cfg.get("features", {}).get("gfc", {})
        if gfc_cfg.get("enabled", False):
            n_bands = int(gfc_cfg.get("n_bands", 8))
            backbone = cfg["features"].get("cnn", {}).get("backbone", "resnet18")
            base_v   = CNN_FEATURE_DIM_MULTISCALE if backbone == "resnet50_multiscale" else CNN_FEATURE_DIM
            full_dim = audio_feature_dim(cfg["features"]["audio"]["n_mfcc"]) + base_v + n_bands
            if fused_cache.shape[-1] < full_dim:
                # Legacy cache that pre-dates GFC — append it now.
                gfc = _load_or_compute_gfc(
                    sample_id, row, cfg, work_dir, sequence_length,
                    n_frames_in_base=fused_cache.shape[0],
                )
                if gfc is not None:
                    return np.concatenate([fused_cache, gfc], axis=-1)
            # GFC already embedded in cached features — return as-is.
        return fused_cache

    # No cache — extract everything from scratch
    y, sr, frames = process_video_sample(row["video_path"], work_dir / "preprocessed" / sample_id, cfg)

    audio_cfg = cfg["features"]["audio"]
    a_seq = extract_mfcc_with_deltas(y, sr, **audio_cfg)

    cnn_cfg = cfg["features"].get("cnn", {})
    backbone = cnn_cfg.get("backbone", "resnet18")
    if backbone == "resnet50_multiscale":
        v_seq = extract_multiscale_resnet50_frame_features(
            frames,
            image_size=cnn_cfg.get("image_size", 224),
            batch_size=cnn_cfg.get("batch_size", 8),
            pretrained=cnn_cfg.get("pretrained", True),
            device=cnn_cfg.get("device", "cpu"),
        )
    else:
        v_seq = extract_resnet18_frame_features(
            frames,
            image_size=cnn_cfg.get("image_size", 224),
            batch_size=cnn_cfg.get("batch_size", 16),
            pretrained=cnn_cfg.get("pretrained", True),
            device=cnn_cfg.get("device", "cpu"),
        )

    # Optionally append GFC to video features before fusion
    gfc_cfg = cfg.get("features", {}).get("gfc", {})
    if gfc_cfg.get("enabled", False):
        n_bands = int(gfc_cfg.get("n_bands", 8))
        g_seq = extract_gfc_features(frames, n_bands=n_bands)  # [T_frames, n_bands]
        n_v = min(len(v_seq), len(g_seq))
        v_seq = np.concatenate([v_seq[:n_v], g_seq[:n_v]], axis=-1)

    per_sample_norm = cfg.get("features", {}).get("per_sample_norm", True)
    if per_sample_norm:
        fuser = FeatureFusionNormalizer()
        fused = fuser.fit_transform(fuser.fuse(a_seq, v_seq))
    else:
        fused = FeatureFusionNormalizer().fuse(a_seq, v_seq)
    windows = align_and_pack_sequences(fused, sequence_length=sequence_length)
    seq = windows.mean(axis=0)

    save_cached_feature(seq_cache, seq)
    return seq


def prepare_sequences(df, cfg, work_dir: Path, sequence_length: int, cache_dir: Path | None = None, split_name: str = ""):
    from tqdm import tqdm
    rows = list(df.iterrows())
    n = len(rows)
    seqs: list = [None] * n
    labels: list = [None] * n
    ids: list = [None] * n

    label = f"  Loading {split_name} features" if split_name else "  Loading features"

    def load_one(args):
        idx, (i, row) = args
        seq = _build_features_for_row(row, cfg, work_dir, sequence_length, cache_dir=cache_dir)
        return idx, seq, int(row["label"]), str(row.get("id", i))

    import os as _os
    with ThreadPoolExecutor(max_workers=_os.cpu_count() or 8) as pool:
        futures = {pool.submit(load_one, (idx, item)): idx for idx, item in enumerate(rows)}
        with tqdm(total=n, desc=label, unit="sample") as pbar:
            for fut in as_completed(futures):
                idx, seq, lbl, sid = fut.result()
                seqs[idx] = seq
                labels[idx] = lbl
                ids[idx] = sid
                pbar.update(1)

    return np.array(seqs), np.array(labels, dtype=int), ids


def _load_packed_splits(
    packed_dir: Path,
    train_df,
    val_df,
) -> tuple:
    """
    Load pre-packed features from data/packed/<sig>/ in one shot and return
    (train_seq, train_y, train_ids, val_seq, val_y, val_ids).

    Samples present in the split but absent from the pack are silently dropped
    (this only happens if extract_all.py was interrupted — re-run it to fill
    the gap, then re-run pack_features.py with --force).
    """
    id_to_idx   = json.loads((packed_dir / "ids.json").read_text(encoding="utf-8"))
    all_features = np.load(packed_dir / "features.npy")   # [N, T, D]
    all_labels   = np.load(packed_dir / "labels.npy")     # [N]

    def _split(df):
        ids, idx = [], []
        for i, row in df.iterrows():
            sid = str(row.get("id", i))
            if sid in id_to_idx:
                ids.append(sid)
                idx.append(id_to_idx[sid])
        missing = len(df) - len(ids)
        if missing:
            print(f"  WARNING: {missing} samples from split not found in packed features.")
        feats  = all_features[idx].copy()           # copy out of mmap for fast indexing
        labels = all_labels[idx].astype(np.int64)
        return feats, labels, ids

    tr_feat, tr_y, tr_ids = _split(train_df)
    va_feat, va_y, va_ids = _split(val_df)
    return tr_feat, tr_y, tr_ids, va_feat, va_y, va_ids


def _print_run_banner(cfg: dict, config_path: str) -> None:
    W = 60
    model_cfg = cfg.get("model", {})
    cnn_cfg   = cfg["features"].get("cnn", {})
    fusion    = "CrossModalAttention" if model_cfg.get("cross_attention") else "ModalAttention (scalar gate)"
    backbone  = cnn_cfg.get("backbone", "resnet18")
    gbwso     = "enabled" if cfg["optimization"].get("feature_selection", True) else "disabled"
    print("=" * W)
    print(f"  EXPERIMENT : {cfg['experiment_name']}")
    print(f"  Config     : {config_path}")
    print(f"  Dataset    : {cfg['dataset']['name']}  ->  {cfg['dataset']['metadata_file']}")
    print(f"  Backbone   : {backbone}")
    print(f"  Fusion     : {fusion}")
    print(f"  GBWSO      : {gbwso}")
    print(f"  Epochs     : {cfg['training']['epochs']}  patience={cfg['training']['early_stopping_patience']}")
    print(f"  LR / WD    : {cfg['training']['lr']:.0e}  /  {cfg['training']['weight_decay']:.0e}")
    print(f"  Seq length : {cfg['training'].get('sequence_length', 12)}")
    print("=" * W)


def run(config_path: str, pre_extract: bool = False, workers: int = 0):
    cfg = load_config(config_path)
    # CMIS needs the GFC band count to slice the spectral sub-features out of the
    # video stream. Derive it once from the feature config so model and data agree.
    _gfc_cfg = cfg.get("features", {}).get("gfc", {})
    cfg.setdefault("model", {}).setdefault(
        "gfc_bands",
        int(_gfc_cfg.get("n_bands", 0)) if _gfc_cfg.get("enabled", False) else 0,
    )
    seed_everything(cfg["seed"])
    _print_run_banner(cfg, config_path)
    sequence_length = cfg["training"].get("sequence_length", 12)
    cache_signature = build_cache_signature(cfg)
    work_dir = Path(".") / cache_signature
    cache_dir = Path(cfg["dataset"]["cache_dir"]) / cache_signature
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, _ = build_dataset_splits(cfg)

    # ── Packed fast-path: load all features in one shot ───────────────────────
    # Populated by:  python scripts/extract_all.py  then  scripts/pack_features.py
    packed_dir   = Path("data/packed") / cache_signature
    using_packed = (packed_dir / "features.npy").exists()

    if using_packed:
        print("============================================================")
        print(f"  [PACKED] Loading pre-packed features from {packed_dir}")
        t_load = time.perf_counter()
        train_seq, train_y, train_ids, val_seq, val_y, val_ids = _load_packed_splits(
            packed_dir, train_df, val_df
        )
        print(f"  [PACKED] {len(train_seq):,} train + {len(val_seq):,} val loaded in {time.perf_counter() - t_load:.1f}s")
        print("============================================================")
    else:
        if pre_extract:
            import os as _os
            n_workers = workers if workers > 0 else (_os.cpu_count() or 8)
            parallel_extract_features(
                pd.concat([train_df, val_df]).reset_index(drop=True),
                cache_dir=cache_dir,
                cfg=cfg,
                work_dir=work_dir,
                sequence_length=sequence_length,
                n_workers=n_workers,
            )

        # Auto-fill GFC cache before loading (runs only if files are missing)
        import os as _os
        _ensure_gfc_cache(pd.concat([train_df, val_df], ignore_index=True), cfg, workers=_os.cpu_count() or 8)

        train_seq, train_y, train_ids = prepare_sequences(train_df, cfg, work_dir, sequence_length, cache_dir=cache_dir, split_name="train")
        val_seq, val_y, val_ids       = prepare_sequences(val_df,   cfg, work_dir, sequence_length, cache_dir=cache_dir, split_name="val")

    split_summary = {
        "train": {"count": int(len(train_seq)), "labels": {str(int(k)): int(v) for k, v in zip(*np.unique(train_y, return_counts=True))}},
        "val":   {"count": int(len(val_seq)),   "labels": {str(int(k)): int(v) for k, v in zip(*np.unique(val_y,   return_counts=True))}},
    }

    # ── Global normalisation: fit on training set, apply to train + val ──────
    # Per-sample normalisation (inside _build_features_for_row) removes absolute
    # signal. A global scaler fitted on all training samples restores it so the
    # model sees cross-sample differences in feature magnitude.
    print("============================================================")
    print(f"  Fitting global feature normalizer on {len(train_seq)} training samples …")
    global_norm = FeatureFusionNormalizer()
    train_seq = global_norm.fit_transform(train_seq)
    val_seq   = global_norm.transform(val_seq)
    norm_cache_dir = Path(cfg["training"]["checkpoint_dir"]) / "norm_cache"
    global_norm.save(Path(cfg["training"]["checkpoint_dir"]) / "global_scaler.pkl")
    print(f"  Global scaler saved → {Path(cfg['training']['checkpoint_dir']) / 'global_scaler.pkl'}")
    if using_packed:
        print("  [PACKED] Skipping norm_cache writes — features stay in RAM.")
    print("============================================================")

    if not using_packed:
        # Non-packed path: persist normalised sequences so MultiModalSequenceDataset
        # can read them from norm_cache_dir during DataLoader iteration.
        norm_cache_dir.mkdir(parents=True, exist_ok=True)
        for sid, seq in zip(train_ids, train_seq):
            np.save(norm_cache_dir / f"{sid}_sequence.npy", seq)
        for sid, seq in zip(val_ids, val_seq):
            np.save(norm_cache_dir / f"{sid}_sequence.npy", seq)

    n_total_features = train_seq.shape[-1]
    audio_dim = audio_feature_dim(cfg["features"]["audio"]["n_mfcc"])
    video_dim = n_total_features - audio_dim
    gfc_n_bands = int(cfg.get("features", {}).get("gfc", {}).get("n_bands", 8))
    gfc_enabled = cfg.get("features", {}).get("gfc", {}).get("enabled", False)
    resnet_dim = video_dim - gfc_n_bands if gfc_enabled else video_dim

    groups = {
        "mfcc": (0, cfg["features"]["audio"]["n_mfcc"]),
        "delta": (cfg["features"]["audio"]["n_mfcc"], 2 * cfg["features"]["audio"]["n_mfcc"]),
        "delta_delta": (2 * cfg["features"]["audio"]["n_mfcc"], audio_dim),
        "resnet": (audio_dim, audio_dim + resnet_dim),
    }
    if gfc_enabled:
        groups["gfc"] = (audio_dim + resnet_dim, n_total_features)

    feature_selection_enabled = cfg["optimization"].get("feature_selection", True)
    if feature_selection_enabled:
        fit_eval = FitnessEvaluator(
            k_folds=cfg["dataset"]["k_fold"].get("n_splits", 5),
            random_state=cfg["seed"],
            weights=cfg["optimization"].get("fitness_weights", None),
        )
        selector = GBWSOFeatureSelector(
            n_features=n_total_features,
            n_sharks=cfg["optimization"]["n_sharks"],
            max_iter=cfg["optimization"]["max_iter"],
            min_features=cfg["optimization"]["min_features"],
            max_features=cfg["optimization"]["max_features"],
            workers=cfg["optimization"]["parallel_workers"],
            random_state=cfg["seed"],
            checkpoint_path=Path(cfg["training"]["checkpoint_dir"]) / "gbwso_state.npz",
            mask_cache_dir=Path("./logs/gbwso_mask_cache") / cfg["experiment_name"],
        )

        x_flat = train_seq.mean(axis=1)

        # Subsample for GBWSO speed — 5K samples is enough to rank features.
        # Using all 51K makes each RandomForest fitness call ~10s (hours total).
        gbwso_sample = 5000
        if len(x_flat) > gbwso_sample:
            rng_sub = np.random.default_rng(cfg["seed"])
            sub_idx = rng_sub.choice(len(x_flat), gbwso_sample, replace=False)
            x_gbwso, y_gbwso = x_flat[sub_idx], train_y[sub_idx]
        else:
            x_gbwso, y_gbwso = x_flat, train_y

        print("============================================================")
        print(f"  GBWSO feature selection  ({cfg['optimization']['n_sharks']} sharks,"
              f" {cfg['optimization']['max_iter']} iterations,"
              f" {n_total_features} features → target {cfg['optimization']['min_features']}–{cfg['optimization']['max_features']})")
        print(f"  Fitness evaluated on {len(x_gbwso)} samples (subsampled for speed)")
        print("============================================================")
        best_mask, _ = selector.fit(x_gbwso, y_gbwso, fit_eval, groups=groups)
        print("============================================================")
        print(f"  GBWSO done  —  selected {int(best_mask.sum())} / {n_total_features} features")
        print("============================================================")
    else:
        best_mask = np.ones(n_total_features, dtype=np.float32)
        print("============================================================")
        print(f"  GBWSO      : skipped — using all {n_total_features} features")
        gfc_info = f"  (ResNet={resnet_dim} + GFC={gfc_n_bands})" if gfc_enabled else ""
        print(f"  Audio dim  : {audio_dim}   Video dim : {video_dim}{gfc_info}")
        print("============================================================")
    feature_mask_path = Path(cfg["training"]["checkpoint_dir"]) / "feature_mask.npy"
    feature_mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature_mask_path, best_mask.astype(np.float32))

    # Resolve optional LAV-DF annotation JSON for frame-level labels (TFLH)
    annotation_json = cfg.get("dataset", {}).get("annotation_json", None)
    fps = cfg["preprocessing"].get("fps", 1.0)

    if using_packed:
        # All features are already in RAM — zero disk reads during training.
        train_ds = PackedSequenceDataset(
            features=train_seq,
            labels=train_y,
            ids=train_ids,
            sequence_length=sequence_length,
            feature_mask=best_mask,
            annotation_json=annotation_json,
            fps=fps,
        )
        val_ds = PackedSequenceDataset(
            features=val_seq,
            labels=val_y,
            ids=val_ids,
            sequence_length=sequence_length,
            feature_mask=best_mask,
            annotation_json=annotation_json,
            fps=fps,
        )
    else:
        train_rows = train_df.copy()
        train_rows["id"] = train_ids
        val_rows = val_df.copy()
        val_rows["id"] = val_ids
        train_ds = MultiModalSequenceDataset(
            train_rows, norm_cache_dir,
            sequence_length=sequence_length,
            feature_mask=best_mask,
            annotation_json=annotation_json,
            fps=fps,
        )
        val_ds = MultiModalSequenceDataset(
            val_rows, norm_cache_dir,
            sequence_length=sequence_length,
            feature_mask=best_mask,
            annotation_json=annotation_json,
            fps=fps,
        )

    tr_cfg = cfg["training"]
    if tr_cfg.get("class_weighting", False):
        positives = max(int((train_y == 1).sum()), 1)
        negatives = max(int((train_y == 0).sum()), 1)
        tr_cfg["pos_weight"] = negatives / positives
        split_summary["pos_weight"] = float(tr_cfg["pos_weight"])
    Path(tr_cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(tr_cfg["checkpoint_dir"], "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    sampler = None
    shuffle = True
    if tr_cfg.get("balanced_sampler", False):
        class_counts = np.bincount(train_y.astype(int), minlength=2)
        sample_weights = np.array([1.0 / max(class_counts[int(label)], 1) for label in train_y], dtype=np.float32)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    # cuDNN auto-tune for fixed-size sequences; TF32 for Ampere+ tensor cores
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    import os as _os
    # PackedSequenceDataset is pure in-memory indexing — pickling 8+ GB through
    # a Windows spawn pipe causes OSError/truncation. Use 0 workers (main process
    # only); __getitem__ is fast enough that the DataLoader is never the bottleneck.
    num_workers = 0 if using_packed else 4
    pin_memory  = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=tr_cfg["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tr_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BiGRUSoftplusDetector(audio_dim=audio_dim, video_dim=video_dim, cfg=cfg["model"])

    # Seed TDFS from GBWSO prior mask when both are enabled
    if cfg["model"].get("temporal_feature_selection", False):
        model.init_tdfs_from_prior(best_mask)
        print("  TDFS       : seeded from GBWSO prior mask")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fusion_type = type(model.modal_fusion).__name__
    tdfs_status = "on (seeded from GBWSO)" if cfg["model"].get("temporal_feature_selection") else "off"
    tflh_status = "on" if cfg["model"].get("frame_localization") else "off"
    ann_status  = annotation_json if annotation_json else "none (frame labels disabled)"
    print("============================================================")
    print(f"  Model      : {model.__class__.__name__}")
    print(f"  Fusion     : {fusion_type}")
    print(f"  TDFS       : {tdfs_status}")
    print(f"  TFLH       : {tflh_status}")
    print(f"  Annotations: {ann_status}")
    print(f"  Parameters : {n_params:,}")
    print(f"  Device     : {device}")
    print(f"  Features   : {int(best_mask.sum())} active  (audio={audio_dim}  video={video_dim})")
    print("============================================================")
    trainer = Trainer(model, cfg, audio_dim=audio_dim, video_dim=video_dim, device=device)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--pre_extract",
        action="store_true",
        help="Pre-extract all features in parallel before training",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Extraction workers (0=auto: cpu_count-4)",
    )
    args = parser.parse_args()
    run(args.config, pre_extract=args.pre_extract, workers=args.workers)
