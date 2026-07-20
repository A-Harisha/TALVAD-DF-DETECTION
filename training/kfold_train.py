from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.builder import build_dataset_splits, iter_kfold
from datasets.sequence_dataset import MultiModalSequenceDataset
from features.audio_features import audio_feature_dim
from features.cache import build_cache_signature
from models.bigru_detector import BiGRUSoftplusDetector
from training.trainer import Trainer
from utils.config import load_config
from utils.io import save_checkpoint
from utils.reproducibility import seed_everything


def run(config_path: str):
    cfg = load_config(config_path)
    seed_everything(cfg["seed"])
    sequence_length = cfg["training"].get("sequence_length", 12)
    cache_signature = build_cache_signature(cfg)

    train_df, _, _ = build_dataset_splits(cfg)
    cache_dir = Path(cfg["dataset"]["cache_dir"]) / cache_signature
    mask_path = Path(cfg["training"]["checkpoint_dir"]) / "feature_mask.npy"
    feature_mask = np.load(mask_path) if mask_path.exists() else None

    fold_results = []
    for fold_idx, (tr_df, va_df) in enumerate(iter_kfold(train_df, cfg), start=1):
        tr_df = tr_df.copy()
        va_df = va_df.copy()
        tr_df["id"] = [str(r.get("id", i)) for i, (_, r) in enumerate(tr_df.iterrows())]
        va_df["id"] = [str(r.get("id", i)) for i, (_, r) in enumerate(va_df.iterrows())]

        train_ds = MultiModalSequenceDataset(tr_df, cache_dir, sequence_length=sequence_length, feature_mask=feature_mask)
        val_ds = MultiModalSequenceDataset(va_df, cache_dir, sequence_length=sequence_length, feature_mask=feature_mask)

        x0 = np.load(cache_dir / f"{tr_df.iloc[0]['id']}_sequence.npy")
        in_dim = x0.shape[-1]
        audio_dim = audio_feature_dim(cfg["features"]["audio"]["n_mfcc"])
        video_dim = in_dim - audio_dim

        model = BiGRUSoftplusDetector(audio_dim, video_dim, cfg["model"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = Trainer(model, cfg, audio_dim=audio_dim, video_dim=video_dim, device=device)

        train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)
        state = trainer.fit(train_loader, val_loader)
        fold_results.append({"fold": fold_idx, "best_val_loss": state.best_val_loss, "best_epoch": state.best_epoch})

    save_checkpoint(Path(cfg["training"]["checkpoint_dir"]) / "kfold_summary.pt", {"fold_results": fold_results})
    print(fold_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run(args.config)
