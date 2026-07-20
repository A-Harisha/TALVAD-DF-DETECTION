"""
TALVAD-X detector — the upgraded audio-visual deepfake detector built around
CMIS-2 (Consistency-Modulated Spectral-Temporal attention).

Kept SEPARATE from models/bigru_detector.py so the working v8 / v8gen path is
untouched. Reuses the shared building blocks in models/attention.py.

Feature layout (contiguous, driven by config `model.dims`):
    x = [ audio (MFCC + wav2vec2) | appearance (face EfficientNet) | spectral ]
        |<------- audio_dim ----->|<----- app_dim ----->|<-- spec_dim -->|
    audio_x = x[..., :audio_dim]                 -> fusion + CMIS-2 audio side
    video_x = x[..., audio_dim:]                 -> fusion visual side
              = [appearance | spectral]; CMIS-2 reads appearance + spectral
              separately (spectral feeds the consistency-gated frequency branch).

Forward returns (logits, frame_logits, attn) — drop-in compatible with the
existing Trainer. `attn` carries the CMIS-2 tensors the trainer needs for the
inconsistency-localisation BCE and the InfoNCE correspondence loss.

Ablation flags (config `model.cmis`):
    enabled            : master switch (False -> A0 baseline, no CMIS branch)
    spectral_branch    : concat the consistency-gated spectral feature  (A1/A3)
    consistency_branch : concat the projected inconsistency feature      (A2)
    coupling           : let inconsistency modulate spectral bands        (A3)
    warm_init          : warm-start the coupling/temporal gates           (A5)
    contrastive        : (read by trainer) add the InfoNCE objective      (A4)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.attention import (
    ConsistencyModulatedSpectralTemporalAttention,
    CrossModalAttentionFusion,
    TemporalAttention,
)


class TALVADXDetector(nn.Module):
    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        dims = cfg["dims"]
        self.audio_dim = int(dims["audio"])
        self.app_dim = int(dims["appearance"])
        self.spec_dim = int(dims["spectral"])
        self.video_dim = self.app_dim + self.spec_dim

        h1 = int(cfg.get("hidden_dim_1", 128))
        h2 = int(cfg.get("hidden_dim_2", 64))
        dropout = float(cfg.get("dropout", 0.4))
        dense = int(cfg.get("dense_dim", 64))
        n_heads = int(cfg.get("n_heads", 4))

        # Cross-modal fusion: audio vs the full visual stream (appearance+spectral).
        self.fusion = CrossModalAttentionFusion(self.audio_dim, self.video_dim, h1, n_heads=n_heads)

        # CMIS-2 branch.
        cmis = cfg.get("cmis", {}) or {}
        self.use_cmis = bool(cmis.get("enabled", True))
        self.use_spectral_branch = bool(cmis.get("spectral_branch", True))
        self.use_consistency_branch = bool(cmis.get("consistency_branch", True))
        self.contrastive = bool(cmis.get("contrastive", True))  # read by trainer
        sh = int(cmis.get("spectral_hidden", 64))
        ih = int(cmis.get("incon_hidden", 16))

        extra = 0
        if self.use_cmis:
            self.cmis = ConsistencyModulatedSpectralTemporalAttention(
                audio_dim=self.audio_dim,
                visual_dim=self.app_dim,
                spectral_dim=self.spec_dim,
                corr_dim=int(cmis.get("corr_dim", 128)),
                spectral_hidden=sh,
                incon_hidden=ih,
                band_mlp=int(cmis.get("band_mlp", 32)),
                coupling=bool(cmis.get("coupling", True)),
                warm_init=bool(cmis.get("warm_init", True)),
                motion=bool(cmis.get("motion", False)),
            )
            if self.use_spectral_branch:
                extra += sh
            if self.use_consistency_branch:
                extra += ih

        temporal_in = h1 + extra
        self.bigru1 = nn.GRU(temporal_in, h1, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(h1 * 2)
        self.bigru2 = nn.GRU(h1 * 2, h2, batch_first=True, bidirectional=True)

        self.temporal_attention = TemporalAttention(h2 * 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(h2 * 2, dense)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dense, 1)

        self.use_tflh = bool(cfg.get("frame_localization", True))
        if self.use_tflh:
            if bool(cfg.get("frame_head_strong", False)):
                # 2-layer MLP head — more capacity for per-frame forgery scoring.
                self.frame_head = nn.Sequential(
                    nn.Linear(h2 * 2, h2), nn.GELU(), nn.Dropout(dropout), nn.Linear(h2, 1)
                )
            else:
                self.frame_head = nn.Linear(h2 * 2, 1)

    def _split_visual(self, video_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        app = video_x[..., : self.app_dim]
        spec = video_x[..., self.app_dim : self.app_dim + self.spec_dim]
        return app, spec

    def forward(
        self,
        audio_x: torch.Tensor,  # [B,T,audio_dim]
        video_x: torch.Tensor,  # [B,T,app_dim+spec_dim]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        app, spec = self._split_visual(video_x)
        fused, _ = self.fusion(audio_x, video_x)  # [B,T,h1]

        attn: Dict[str, Optional[torch.Tensor]] = {}
        if self.use_cmis:
            c = self.cmis(audio_x, app, spec)
            parts = [fused]
            if self.use_spectral_branch:
                parts.append(c["spectral_feat"])
            if self.use_consistency_branch:
                parts.append(c["incon_feat"])
            x = torch.cat(parts, dim=-1)
            attn.update(
                cmis_incon=c["incon"],       # [B,T] for inconsistency-localisation BCE
                cmis_z_a=c["z_a"],           # [B,T,C] for InfoNCE
                cmis_z_v=c["z_v"],
                cmis_band_w=c["band_w"],     # [B,T,S] for figures
            )
        else:
            x = fused

        x, _ = self.bigru1(x)
        x = self.dropout1(x)
        x = self.layer_norm1(x)
        x, _ = self.bigru2(x)  # [B,T,h2*2]

        frame_logits = self.frame_head(x) if self.use_tflh else None

        context, temporal_weights = self.temporal_attention(x)
        h = self.dropout2(context)
        h = self.fc1(h)
        h = self.act(h)
        logits = self.fc2(h)  # [B,1]

        attn["temporal"] = temporal_weights
        attn["a_ctx"] = getattr(self.fusion, "last_a_ctx", None)
        attn["v_ctx"] = getattr(self.fusion, "last_v_ctx", None)
        # Time-pooled representation for the DANN domain classifier (DG).
        attn["temporal_context"] = context
        return logits, frame_logits, attn
