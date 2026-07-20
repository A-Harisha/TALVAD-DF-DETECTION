from __future__ import annotations

from functools import lru_cache
from typing import Dict
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


CNN_FEATURE_DIM = 512           # ResNet-18 backbone
CNN_FEATURE_DIM_MULTISCALE = 1024  # ResNet-50 multi-scale backbone (layer2+layer3+layer4 → projected)


@lru_cache(maxsize=2)
def _load_resnet18_feature_extractor(pretrained: bool = True) -> nn.Module:
    weights = None
    if pretrained:
        try:
            weights = ResNet18_Weights.DEFAULT
        except Exception as exc:
            warnings.warn(f"Could not load ResNet18 weights metadata; using random weights. {exc}", RuntimeWarning, stacklevel=2)

    try:
        model = resnet18(weights=weights)
    except Exception as exc:
        warnings.warn(f"Could not load pretrained ResNet18 weights; using random weights. {exc}", RuntimeWarning, stacklevel=2)
        model = resnet18(weights=None)

    model.fc = nn.Identity()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@lru_cache(maxsize=2)
def _load_multiscale_resnet50_extractor(pretrained: bool = True) -> nn.Module:
    """
    Multi-scale ResNet-50 feature extractor.

    Extracts features at three depths and concatenates after global average pooling:
      layer2 →  512-dim  (texture / edge artifacts)
      layer3 → 1024-dim  (semantic inconsistencies)
      layer4 → 2048-dim  (high-level forgery patterns)
    Total 3584-dim projected to CNN_FEATURE_DIM_MULTISCALE (1024) via a linear layer.

    Note: invalidates any ResNet-18 feature cache — re-run build_cache after switching.
    """
    weights = None
    if pretrained:
        try:
            weights = ResNet50_Weights.DEFAULT
        except Exception as exc:
            warnings.warn(f"Could not load ResNet50 weights: {exc}", RuntimeWarning, stacklevel=2)

    try:
        base = resnet50(weights=weights)
    except Exception as exc:
        warnings.warn(f"Could not load pretrained ResNet50 weights: {exc}", RuntimeWarning, stacklevel=2)
        base = resnet50(weights=None)

    class _MultiScaleExtractor(nn.Module):
        def __init__(self, base_model: nn.Module) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool
            )
            self.layer1 = base_model.layer1
            self.layer2 = base_model.layer2   # output channels: 512
            self.layer3 = base_model.layer3   # output channels: 1024
            self.layer4 = base_model.layer4   # output channels: 2048
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.proj = nn.Linear(512 + 1024 + 2048, CNN_FEATURE_DIM_MULTISCALE)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            f2 = self.gap(x).flatten(1)        # [B, 512]
            x = self.layer3(x)
            f3 = self.gap(x).flatten(1)        # [B, 1024]
            x = self.layer4(x)
            f4 = self.gap(x).flatten(1)        # [B, 2048]
            return self.proj(torch.cat([f2, f3, f4], dim=1))  # [B, 1024]

    model = _MultiScaleExtractor(base)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _prepare_frames(frames: np.ndarray, image_size: int) -> torch.Tensor:
    prepared = []
    for frame in frames:
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        elif frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=2)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        prepared.append(frame)

    arr = np.stack(prepared, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - mean) / std


def extract_resnet18_frame_features(
    frames: np.ndarray,
    image_size: int = 224,
    batch_size: int = 16,
    pretrained: bool = True,
    device: str = "cpu",
) -> np.ndarray:
    if frames.shape[0] == 0:
        return np.zeros((1, CNN_FEATURE_DIM), dtype=np.float32)

    model = _load_resnet18_feature_extractor(pretrained=pretrained).to(device)
    x = _prepare_frames(frames, image_size=image_size)
    outputs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], max(batch_size, 1)):
            batch = x[start : start + batch_size].to(device)
            feats = model(batch).detach().cpu().numpy().astype(np.float32)
            outputs.append(feats)

    seq = np.concatenate(outputs, axis=0)
    mean = seq.mean(axis=0, keepdims=True)
    std = seq.std(axis=0, keepdims=True) + 1e-8
    return (seq - mean) / std


def extract_multiscale_resnet50_frame_features(
    frames: np.ndarray,
    image_size: int = 224,
    batch_size: int = 8,
    pretrained: bool = True,
    device: str = "cpu",
) -> np.ndarray:
    if frames.shape[0] == 0:
        return np.zeros((1, CNN_FEATURE_DIM_MULTISCALE), dtype=np.float32)

    model = _load_multiscale_resnet50_extractor(pretrained=pretrained).to(device)
    x = _prepare_frames(frames, image_size=image_size)
    outputs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], max(batch_size, 1)):
            batch = x[start : start + batch_size].to(device)
            feats = model(batch).detach().cpu().numpy().astype(np.float32)
            outputs.append(feats)

    seq = np.concatenate(outputs, axis=0)
    mean = seq.mean(axis=0, keepdims=True)
    std = seq.std(axis=0, keepdims=True) + 1e-8
    return (seq - mean) / std


def cnn_feature_dim(cfg: Dict | None = None) -> int:
    if cfg is not None and not cfg.get("enabled", False):
        return 0
    backbone = cfg.get("backbone", "resnet18") if cfg else "resnet18"
    return CNN_FEATURE_DIM_MULTISCALE if backbone == "resnet50_multiscale" else CNN_FEATURE_DIM
