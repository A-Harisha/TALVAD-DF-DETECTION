from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.builder import build_dataset_splits
from datasets.sequence_dataset import MultiModalSequenceDataset
from evaluation.metrics import compute_binary_metrics
from features.audio_features import audio_feature_dim
from features.cache import build_cache_signature
from models.bigru_detector import BiGRUSoftplusDetector
from visualizations.plots import plot_confusion_matrix, plot_roc
from training.train import _build_features_for_row
from utils.config import load_config
from utils.io import load_checkpoint


def run(config_path: str, checkpoint_path: str):
    cfg = load_config(config_path)
    _, _, test_df = build_dataset_splits(cfg)

    sequence_length = cfg["training"].get("sequence_length", 12)
    cache_signature = build_cache_signature(cfg)
    cache_dir = Path(cfg["dataset"]["cache_dir"]) / cache_signature
    mask_path = Path(cfg["training"]["checkpoint_dir"]) / "feature_mask.npy"
    feature_mask = np.load(mask_path) if mask_path.exists() else None
    test_df = test_df.copy()
    test_df["id"] = [str(r.get("id", i)) for i, (_, r) in enumerate(test_df.iterrows())]

    work_dir = Path(".") / cache_signature
    for _, row in test_df.iterrows():
        seq_path = cache_dir / f"{row['id']}_sequence.npy"
        if not seq_path.exists():
            seq = _build_features_for_row(row, cfg, work_dir, sequence_length)
            seq_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(seq_path, seq)

    ds = MultiModalSequenceDataset(test_df, cache_dir, sequence_length=sequence_length, feature_mask=feature_mask)
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    dummy = np.load(cache_dir / f"{test_df.iloc[0]['id']}_sequence.npy")
    in_dim = dummy.shape[-1]
    audio_dim = audio_feature_dim(cfg["features"]["audio"]["n_mfcc"])
    video_dim = in_dim - audio_dim

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BiGRUSoftplusDetector(audio_dim=audio_dim, video_dim=video_dim, cfg=cfg["model"]).to(device)
    cp = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(cp["model"])
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            a = x[..., :audio_dim]
            v = x[..., audio_dim:]
            logits, _ = model(a, v)
            prob = torch.sigmoid(logits)
            ys.append(y.numpy())
            ps.append(prob.squeeze(-1).cpu().numpy())

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)

    threshold = float(cp.get("threshold", cfg["evaluation"]["threshold"]))
    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    metrics["threshold"] = threshold
    out_dir = Path(cfg["evaluation"].get("output_dir", "visualizations"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in metrics.items()}, f, indent=2)

    plot_confusion_matrix(metrics["confusion_matrix"], out_dir / "confusion_matrix.png")
    plot_roc(metrics["fpr"], metrics["tpr"], metrics["roc_curve_auc"], out_dir / "roc_curve.png")

    print(json.dumps({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    args = parser.parse_args()
    run(args.config, args.checkpoint)
