"""
models/avsr_encoder.py — frozen auto_avsr VSR encoder as an AV-HuBERT-class lip-reading
feature extractor (TALVAD-X v2, the real high-ceiling backbone).

We load the pretrained VSR model `vsr_trlrs3vox2_base.pth` (ResNet video frontend ->
Linear(512,768) -> 12-block Conformer, trained on 1759h LRS3+VoxCeleb2 — the same data
scale as AV-HuBERT) and tap the 768-d CONFORMER ENCODER output (before the text decoder).
Those per-frame features encode visual speech / articulation and should transfer across
deepfake generators far better than EfficientNet appearance features.

Runs on modern torch 2.x + GPU. We build E2E directly and infer odim from the checkpoint's
CTC head, so we DON'T need pytorch_lightning / sentencepiece / hydra — just the vendored
espnet model code in third_party/auto_avsr and torch/torchvision.

Input convention (matches datamodule/transforms.py val/test VideoTransform):
  mouth-crop frames RGB [T,1,96,96] uint8/float -> /255 -> CenterCrop(88) -> Grayscale
  -> Normalize(mean=0.421, std=0.165) -> model input [T,1,88,88]; batched as [B,T,1,88,88].
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
_AVSR = REPO / "third_party" / "auto_avsr"
if str(_AVSR) not in sys.path:
    sys.path.insert(0, str(_AVSR))


class AVSRVideoEncoder(nn.Module):
    def __init__(self, ckpt_path: str = "checkpoints/auto_avsr/vsr_trlrs3vox2_base.pth",
                 device: str = "cpu") -> None:
        super().__init__()
        from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E

        ckpt = torch.load(str(REPO / ckpt_path) if not Path(ckpt_path).is_absolute() else ckpt_path,
                          map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        odim = ckpt["ctc.ctc_lo.weight"].shape[0]      # infer vocab size from CTC head
        self.model = E2E(odim, "video")
        self.model.load_state_dict(ckpt, strict=True)  # strict -> proves exact match
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.odim = odim

    @torch.no_grad()
    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        """vid: [B,T,1,88,88] normalized -> encoder features [B,T',768]."""
        x = self.model.frontend(vid)
        x = self.model.proj_encoder(x)
        feat, _ = self.model.encoder(x, None)
        return feat


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = AVSRVideoEncoder(device=dev)
    n = sum(p.numel() for p in enc.model.parameters())
    dummy = torch.zeros(1, 30, 1, 88, 88, device=dev)   # 30-frame mouth clip
    out = enc(dummy)
    print(f"[OK] strict load, odim={enc.odim}, params={n/1e6:.1f}M, device={dev}")
    print(f"[OK] enc_feat={tuple(out.shape)}  (expect [1, ~15, 768] — conformer subsamples ~2x)")
    assert out.shape[-1] == 768
    print("AVSR ENCODER VERIFY OK")
