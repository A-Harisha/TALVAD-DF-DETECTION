"""
models/correspondence.py — self-supervised audio<->visual correspondence PRIOR
(TALVAD-X v2, Phase 2: the "learn real, detect deviation" generalization engine).

Trained on REAL clips ONLY with temporal InfoNCE (+ optional masked reconstruction):
it learns natural per-frame audio<->(appearance+motion) correspondence. Because no
fake is ever seen, there is no generator fingerprint to overfit — so the detection
signal (per-clip correspondence DEVIATION) is generator-agnostic and should transfer.

Deliberately uses audio + appearance(+motion) and EXCLUDES the spectral/frequency
dims — the gate proved frequency is a non-transferable shortcut.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalCorrespondence(nn.Module):
    def __init__(self, audio_dim: int = 376, visual_dim: int = 256, emb_dim: int = 128,
                 hidden: int = 256, motion: bool = True, temperature: float = 0.07,
                 recon_weight: float = 0.5) -> None:
        super().__init__()
        self.motion = motion
        self.temperature = float(temperature)
        self.recon_weight = float(recon_weight)
        vin = visual_dim * (2 if motion else 1)
        self.audio_enc = nn.Sequential(
            nn.Linear(audio_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, emb_dim))
        self.visual_enc = nn.Sequential(
            nn.Linear(vin, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, emb_dim))
        self.temp_a = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.temp_v = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.recon = nn.Linear(emb_dim, emb_dim)  # predict z_v[t] from z_a[t]

    def _with_motion(self, v: torch.Tensor) -> torch.Tensor:
        d = torch.zeros_like(v)
        d[:, 1:] = v[:, 1:] - v[:, :-1]          # frame-to-frame delta (articulation proxy)
        return torch.cat([v, d], dim=-1)

    def forward(self, audio_x: torch.Tensor, visual_x: torch.Tensor):
        if self.motion:
            visual_x = self._with_motion(visual_x)
        za = self.audio_enc(audio_x)
        zv = self.visual_enc(visual_x)
        za, _ = self.temp_a(za)
        zv, _ = self.temp_v(zv)
        za = F.normalize(za, dim=-1)
        zv = F.normalize(zv, dim=-1)
        return za, zv                            # [B,T,E] each, L2-normalized

    def infonce(self, za: torch.Tensor, zv: torch.Tensor) -> torch.Tensor:
        """Symmetric temporal InfoNCE: z_a[t] must match z_v[t] but NOT z_v[t']."""
        B, T, _ = za.shape
        sim = torch.einsum("bte,bse->bts", za, zv) / self.temperature   # [B,T,T]
        tgt = torch.arange(T, device=za.device).expand(B, T).reshape(B * T)
        a2v = F.cross_entropy(sim.reshape(B * T, T), tgt)
        v2a = F.cross_entropy(sim.transpose(1, 2).reshape(B * T, T), tgt)
        return 0.5 * (a2v + v2a)

    def recon_loss(self, za: torch.Tensor, zv: torch.Tensor) -> torch.Tensor:
        pred = F.normalize(self.recon(za), dim=-1)
        return (1.0 - (pred * zv).sum(-1)).mean()

    def loss(self, audio_x: torch.Tensor, visual_x: torch.Tensor):
        za, zv = self.forward(audio_x, visual_x)
        nce = self.infonce(za, zv)
        rec = self.recon_loss(za, zv) if self.recon_weight > 0 else torch.zeros((), device=za.device)
        return nce + self.recon_weight * rec, {"nce": float(nce.detach()), "recon": float(rec.detach())}

    @torch.no_grad()
    def deviation(self, audio_x: torch.Tensor, visual_x: torch.Tensor) -> torch.Tensor:
        """Per-FRAME correspondence deviation [B,T] (higher -> less consistent).
        Whole-clip reductions (mean / top-k / max) are applied downstream so a
        LOCAL fake (few altered frames) isn't diluted by a clip-mean."""
        za, zv = self.forward(audio_x, visual_x)
        return 1.0 - (za * zv).sum(-1)            # [B,T]

    @torch.no_grad()
    def anomaly(self, audio_x: torch.Tensor, visual_x: torch.Tensor) -> torch.Tensor:
        """Per-clip correspondence DEVIATION (higher -> more likely fake)."""
        return self.deviation(audio_x, visual_x).mean(dim=1)   # [B] clip-mean deviation
