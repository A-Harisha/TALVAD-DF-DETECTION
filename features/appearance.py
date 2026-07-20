"""
Frozen EfficientNet-B0 appearance features for TALVAD-X (replaces full-frame
ResNet-18). Run on FACE CROPS (features/face_crop.py), so the appearance encoder
sees the manipulated region rather than background. Frozen, inference-only.

Output: float32 [T, 1280] (EfficientNet-B0 penultimate features).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

EFFICIENTNET_B0_DIM = 1280


@lru_cache(maxsize=1)
def _load_effnet(device: str):
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    net = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    net.classifier = torch.nn.Identity()  # keep 1280-d pooled features
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


@torch.inference_mode()
def extract_efficientnet_features(
    frames: np.ndarray,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Args:
        frames: [T, H, W, 3] uint8 RGB face crops (any size; resized via model
                input expectations — EfficientNet-B0 handles 224 well).
        device: 'cuda' or 'cpu'.
        batch_size: frames per GPU batch.

    Returns:
        float32 [T, 1280].
    """
    frames = np.asarray(frames)
    if frames.shape[0] == 0:
        return np.zeros((0, EFFICIENTNET_B0_DIM), dtype=np.float32)

    net = _load_effnet(device)
    x = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0  # [T,3,H,W]
    mean = _IMAGENET_MEAN.to(x.dtype)
    std = _IMAGENET_STD.to(x.dtype)

    outs = []
    for i in range(0, x.shape[0], batch_size):
        b = ((x[i:i + batch_size] - mean) / std).to(device)
        outs.append(net(b).float().cpu())
    feat = torch.cat(outs, dim=0).numpy().astype(np.float32)
    assert feat.shape == (frames.shape[0], EFFICIENTNET_B0_DIM), feat.shape
    return feat


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    frames = (np.random.rand(10, 224, 224, 3) * 255).astype(np.uint8)
    f1 = extract_efficientnet_features(frames, device=dev, batch_size=4)
    f2 = extract_efficientnet_features(frames, device=dev, batch_size=8)
    f1b = extract_efficientnet_features(frames, device=dev, batch_size=4)
    print(f"[OK] effnet feats shape={f1.shape} finite={np.isfinite(f1).all()}")
    assert f1.shape == (10, 1280) and np.isfinite(f1).all()
    # Same batch size must be bit-stable (reproducible cache).
    assert np.allclose(f1, f1b, atol=1e-5), "non-deterministic at fixed batch size"
    # Across batch sizes, only loose agreement is expected (GPU conv kernel choice).
    print(f"[CHECK] cross-batch max|diff|={np.abs(f1 - f2).max():.5f} (GPU kernel variance)")
    assert np.allclose(f1, f2, atol=5e-2), "unexpectedly large batch-size dependence"
    print("[OK] appearance.py self-test passed")
