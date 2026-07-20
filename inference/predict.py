from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch

from features.audio_features import audio_feature_dim
from features.audio_features import extract_mfcc_with_deltas
from features.cache import build_cache_signature
from features.fusion import FeatureFusionNormalizer, align_and_pack_sequences, apply_feature_mask
from features.video_cnn import extract_resnet18_frame_features
from models.bigru_detector import BiGRUSoftplusDetector
from preprocessing.pipeline import process_video_sample
from utils.config import load_config
from utils.io import load_checkpoint


def predict_video(video_path: str, model, cfg, device: str):
    cache_signature = cfg.get("cache_signature", "default")
    y, sr, frames = process_video_sample(video_path, Path("./inference_cache") / cache_signature / Path(video_path).stem, cfg)

    a_seq = extract_mfcc_with_deltas(y, sr, **cfg["features"]["audio"])
    cnn_cfg = cfg["features"].get("cnn", {})
    v_seq = extract_resnet18_frame_features(
        frames,
        image_size=cnn_cfg.get("image_size", 224),
        batch_size=cnn_cfg.get("batch_size", 16),
        pretrained=cnn_cfg.get("pretrained", True),
        device=cnn_cfg.get("device", "cpu"),
    )
    n = min(len(a_seq), len(v_seq))
    fused = FeatureFusionNormalizer().fit_transform(np.concatenate([a_seq[:n], v_seq[:n]], axis=1))

    seq = align_and_pack_sequences(fused, sequence_length=cfg["training"].get("sequence_length", 12)).mean(axis=0, keepdims=True)
    seq = apply_feature_mask(seq, cfg.get("feature_mask"))
    x = torch.from_numpy(seq).float().to(device)

    audio_dim = audio_feature_dim(cfg["features"]["audio"]["n_mfcc"])
    a = x[..., :audio_dim]
    v = x[..., audio_dim:]

    with torch.no_grad():
        logits, _frame_logits, att = model(a, v)
        prob = torch.sigmoid(logits)

    p = float(prob.squeeze().cpu().item())

    result = {
        "video": video_path,
        "label": "FAKE" if p >= cfg["evaluation"]["threshold"] else "REAL",
        "confidence": p,
        "attention_modal": att["modal"].squeeze(0).cpu().numpy().tolist(),
        "attention_temporal": att["temporal"].squeeze(0).cpu().numpy().tolist(),
    }

    # Modality attribution — present only when MAA heads are enabled in the model
    if att.get("audio_logits") is not None:
        result["audio_fake_score"] = float(torch.sigmoid(att["audio_logits"]).squeeze().cpu().item())
        result["video_fake_score"] = float(torch.sigmoid(att["video_logits"]).squeeze().cpu().item())

    return result


def run(config_path: str, checkpoint_path: str, input_path: str):
    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["cache_signature"] = build_cache_signature(cfg)
    mask_path = Path(cfg["training"]["checkpoint_dir"]) / "feature_mask.npy"
    if mask_path.exists():
        cfg["feature_mask"] = np.load(mask_path)

    input_dim = cfg["model"].get("input_dim", 256)
    audio_dim = audio_feature_dim(cfg["features"]["audio"]["n_mfcc"])
    video_dim = input_dim - audio_dim

    model = BiGRUSoftplusDetector(audio_dim=audio_dim, video_dim=video_dim, cfg=cfg["model"]).to(device)
    cp = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(cp["model"], strict=False)
    model.eval()
    cfg["evaluation"]["threshold"] = float(cp.get("threshold", cfg["evaluation"]["threshold"]))

    path = Path(input_path)
    if path.is_file():
        results = [predict_video(str(path), model, cfg, device)]
    else:
        videos = sorted([p for p in path.rglob("*") if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}])
        results = [predict_video(str(v), model, cfg, device) for v in videos]

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.input)
