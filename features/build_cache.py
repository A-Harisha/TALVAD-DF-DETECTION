from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.builder import build_dataset_splits
from features.audio_features import extract_mfcc_with_deltas
from features.cache import build_cache_signature, save_cached_feature
from features.fusion import FeatureFusionNormalizer, align_and_pack_sequences
from features.video_cnn import extract_resnet18_frame_features
from preprocessing.pipeline import process_video_sample
from utils.config import load_config


def run(config_path: str):
    cfg = load_config(config_path)
    cache_signature = build_cache_signature(cfg)
    cache_dir = Path(cfg["dataset"]["cache_dir"]) / cache_signature
    cache_dir.mkdir(parents=True, exist_ok=True)
    sequence_length = cfg["training"].get("sequence_length", 12)

    train_df, val_df, test_df = build_dataset_splits(cfg)
    full_df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)

    for i, row in full_df.iterrows():
        row = row.to_dict()
        sid = str(row.get("id", i))
        video_path = row["video_path"]

        y, sr, frames = process_video_sample(video_path, Path("./cache_build") / cache_signature / sid, cfg)
        a = extract_mfcc_with_deltas(y, sr, **cfg["features"]["audio"])
        cnn_cfg = cfg["features"].get("cnn", {})
        v = extract_resnet18_frame_features(
            frames,
            image_size=cnn_cfg.get("image_size", 224),
            batch_size=cnn_cfg.get("batch_size", 16),
            pretrained=cnn_cfg.get("pretrained", True),
            device=cnn_cfg.get("device", "cpu"),
        )
        n = min(len(a), len(v))
        fused = FeatureFusionNormalizer().fit_transform(np.concatenate([a[:n], v[:n]], axis=1))
        seq = align_and_pack_sequences(fused, sequence_length=sequence_length).mean(axis=0)

        save_cached_feature(cache_dir / f"{sid}_sequence.npy", seq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run(args.config)
