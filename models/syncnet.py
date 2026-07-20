"""
models/syncnet.py — vendored SyncNet (Chung & Zisserman, "Out of time: automated
lip sync in the wild"), the canonical `syncnet_v2.model` weights from Oxford VGG.

Why it's here (TALVAD-X v2, the AV-sync backbone): our correspondence prior on the
existing EfficientNet+wav2vec2 features tops out ~0.68 cross-dataset because those
features were not built for lip-sync. SyncNet is pretrained on a large corpus of REAL
talking faces to model exactly "does this audio match these lips" — a generator-
agnostic, physics-grounded signal that should transfer. We run it FROZEN as a feature
extractor (1024-d audio + 1024-d lip embedding per 5-frame window) and build the
detector on top.

This is a faithful, dependency-light re-implementation (plain torch, no fairseq) of the
original `SyncNetModel.py` `S` class, reconstructed to match the published weight tensor
shapes exactly (verified: strict state_dict load). Runs on modern torch 2.x + GPU.

INPUT CONVENTIONS (must match the weights — getting these wrong silently degrades it):
  lip  : (B, 3, 5, 224, 224)  BGR, float 0-255, NO /255 normalisation, 5-frame window
  aud  : (B, 1, 13, 20)       MFCC (numcep=13) at 100 fps, 20 frames (=0.2s) per window
Both forward_* return (B, 1024) embeddings; sync = small L2 distance between them.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class S(nn.Module):
    def __init__(self, num_layers_in_fc_layers: int = 1024) -> None:
        super().__init__()

        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),

            nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),

            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(384), nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(256, 512, kernel_size=(5, 4), padding=(0, 0)),
            nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.netfcaud = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

        self.netcnnlip = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=0),
            nn.BatchNorm3d(96), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),

            nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),

            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),

            nn.Conv3d(256, 512, kernel_size=(1, 6, 6), padding=0),
            nn.BatchNorm3d(512), nn.ReLU(),
        )
        self.netfclip = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

    def forward_aud(self, x: torch.Tensor) -> torch.Tensor:
        out = self.netcnnaud(x)            # (B,512,1,1)
        out = out.reshape(out.size(0), -1)  # (B,512)
        return self.netfcaud(out)          # (B,1024)

    def forward_lip(self, x: torch.Tensor) -> torch.Tensor:
        out = self.netcnnlip(x)            # (B,512,1,1,1)
        out = out.reshape(out.size(0), -1)  # (B,512)
        return self.netfclip(out)          # (B,1024)


def load_syncnet(weights_path: str | Path = "checkpoints/syncnet/syncnet_v2.model",
                 device: str = "cpu") -> S:
    """Load frozen SyncNet (strict). eval() so BatchNorm uses running stats."""
    model = S().to(device)
    sd = torch.load(str(weights_path), map_location=device, weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    model.load_state_dict(sd, strict=True)   # strict -> any mismatch raises
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


if __name__ == "__main__":
    # Self-test: strict load + forward shapes on the documented input conventions.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_syncnet(device=dev)
    n = sum(p.numel() for p in m.parameters())
    lip = torch.zeros(2, 3, 5, 224, 224, device=dev)   # BGR 0-255 window
    aud = torch.zeros(2, 1, 13, 20, device=dev)        # 13-MFCC x 20 frames
    with torch.no_grad():
        le = m.forward_lip(lip)
        ae = m.forward_aud(aud)
    print(f"[OK] strict load, params={n/1e6:.2f}M, device={dev}")
    print(f"[OK] lip_emb={tuple(le.shape)}  aud_emb={tuple(ae.shape)}  (expect (2,1024) each)")
    assert le.shape == (2, 1024) and ae.shape == (2, 1024)
    print("SYNCNET SELF-TEST OK")
