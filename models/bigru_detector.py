from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.attention import (
    CrossModalAttentionFusion,
    CrossModalInconsistencySpectralAttention,
    ModalAttentionFusion,
    TemporalAttention,
)
from models.feature_selector import TemporalDifferentiableFeatureSelector


class BiGRUSoftplusDetector(nn.Module):
    """
    TALVAD+GBWSO hybrid detector.

    Optional components (each enabled via cfg flags):
      - TDFS  (temporal_feature_selection: true)
          Learned per-timestep soft feature mask, seeded from GBWSO prior.
      - CrossModalAttentionFusion  (cross_attention: true)
          Bidirectional audio-attends-video / video-attends-audio fusion.
      - TFLH  (frame_localization: true)
          Per-timestep forgery head trained with fake_periods supervision.
    """

    def __init__(self, audio_dim: int, video_dim: int, cfg: Dict) -> None:
        super().__init__()
        hidden_1  = cfg["hidden_dim_1"]
        hidden_2  = cfg["hidden_dim_2"]
        dropout   = cfg["dropout"]
        dense_dim = cfg["dense_dim"]
        n_features = audio_dim + video_dim

        # ── TDFS (optional) ──────────────────────────────────────────────
        self.use_tdfs = cfg.get("temporal_feature_selection", False)
        if self.use_tdfs:
            self.tdfs = TemporalDifferentiableFeatureSelector(
                n_features   = n_features,
                selector_dim = cfg.get("selector_dim", 64),
                temperature  = cfg.get("tdfs_temperature", 0.5),
                l1_weight    = cfg.get("tdfs_l1_weight", 1e-4),
            )

        # ── Cross-modal fusion ────────────────────────────────────────────
        if cfg.get("cross_attention", False):
            n_heads = cfg.get("n_heads", 4)
            self.modal_fusion = CrossModalAttentionFusion(
                audio_dim, video_dim, hidden_1, n_heads=n_heads
            )
        else:
            self.modal_fusion = ModalAttentionFusion(audio_dim, video_dim, hidden_1)

        # ── CMIS — Cross-Modal Inconsistency-Guided Spectral Attention ────
        # Couples the cross-attention's A/V inconsistency to per-band GFC
        # attention (see CrossModalInconsistencySpectralAttention). Requires
        # cross-attention (for the contexts) and GFC bands (for the spectrum).
        cmis_cfg = cfg.get("cmis", {})
        self.gfc_bands = int(cfg.get("gfc_bands", 0))
        self.use_cmis = bool(cmis_cfg.get("enabled", False))
        if self.use_cmis:
            if not cfg.get("cross_attention", False) or self.gfc_bands <= 0:
                warnings.warn(
                    "cmis.enabled is set but requires cross_attention=true and "
                    f"gfc_bands>0 (got cross_attention={cfg.get('cross_attention', False)}, "
                    f"gfc_bands={self.gfc_bands}); CMIS will be disabled.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.use_cmis = False
            else:
                self.cmis = CrossModalInconsistencySpectralAttention(
                    ctx_dim=hidden_1,
                    n_bands=self.gfc_bands,
                    hidden_dim=hidden_1,
                    spectral_dim=int(cmis_cfg.get("spectral_dim", 32)),
                    coupling=bool(cmis_cfg.get("inconsistency_coupling", True)),
                )

        # ── Temporal BiGRU stack ─────────────────────────────────────────
        self.bigru1 = nn.GRU(
            input_size=hidden_1, hidden_size=hidden_1,
            batch_first=True, bidirectional=True,
        )
        self.dropout1    = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_1 * 2)
        self.bigru2 = nn.GRU(
            input_size=hidden_1 * 2, hidden_size=hidden_2,
            batch_first=True, bidirectional=True,
        )
        self.temporal_attention = TemporalAttention(hidden_2 * 2)
        self.dropout2 = nn.Dropout(dropout)

        # ── Video-level classification head ──────────────────────────────
        self.fc1      = nn.Linear(hidden_2 * 2, dense_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dense_dim, 1)

        # ── TFLH — per-timestep forgery localisation (optional) ──────────
        self.use_tflh = cfg.get("frame_localization", False)
        if self.use_tflh:
            self.frame_head = nn.Linear(hidden_2 * 2, 1)

        # ── MAA — Modality Attribution Heads (optional) ───────────────────
        # Trained with per-modality labels (modify_audio / modify_video from
        # LAV-DF) so the model can attribute a detected forgery to its source
        # modality at inference time.  Operates on the time-pooled cross-modal
        # contexts a_ctx / v_ctx, which encode each modality after it has
        # attended to the other — exactly where A/V inconsistency is latent.
        self.use_maa = cfg.get("modality_attribution", False)
        if self.use_maa:
            # MAA reads the per-modality cross-attention contexts (a_ctx / v_ctx),
            # which only exist when cross_attention is on. Without it the heads
            # never receive gradients and stay silently untrained — warn loudly.
            if not cfg.get("cross_attention", False):
                warnings.warn(
                    "modality_attribution is enabled but cross_attention is off; "
                    "the MAA heads have no cross-modal context to read and will not "
                    "train. Enable cross_attention or disable modality_attribution.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.audio_fake_head = nn.Linear(hidden_1, 1)
            self.video_fake_head = nn.Linear(hidden_1, 1)

    # ------------------------------------------------------------------
    def init_tdfs_from_prior(self, prior_mask) -> None:
        """Seed TDFS importance head from GBWSO binary mask."""
        if self.use_tdfs:
            self.tdfs.init_from_prior_mask(prior_mask)

    # ------------------------------------------------------------------
    def forward(
        self,
        audio_x: torch.Tensor,   # [B, T, audio_dim]
        video_x: torch.Tensor,   # [B, T, video_dim]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:

        # ── TDFS ─────────────────────────────────────────────────────────
        tdfs_l1 = torch.tensor(0.0, device=audio_x.device)
        soft_mask = None
        if self.use_tdfs:
            combined = torch.cat([audio_x, video_x], dim=-1)   # [B, T, D]
            combined, soft_mask, tdfs_l1 = self.tdfs(combined)
            audio_x = combined[..., : audio_x.size(-1)]
            video_x = combined[..., audio_x.size(-1):]

        # ── Fusion ───────────────────────────────────────────────────────
        fused, modal_weights = self.modal_fusion(audio_x, video_x)   # [B, T, H1]

        # ── CMIS — inconsistency-guided spectral attention (optional) ─────
        cmis_incon = None
        cmis_spectral_w = None
        if self.use_cmis:
            a_seq = getattr(self.modal_fusion, "last_a_ctx_seq", None)
            v_seq = getattr(self.modal_fusion, "last_v_ctx_seq", None)
            if a_seq is not None and v_seq is not None:
                gfc = video_x[..., -self.gfc_bands:]                  # post-TDFS GFC bands
                injection, cmis_incon, cmis_spectral_w = self.cmis(a_seq, v_seq, gfc)
                fused = fused + injection                            # zero-gated at init

        # ── BiGRU stack ──────────────────────────────────────────────────
        x, _ = self.bigru1(fused)
        x     = self.dropout1(x)
        x     = self.layer_norm1(x)
        x, _  = self.bigru2(x)                                        # [B, T, H2*2]

        # ── TFLH (frame logits before temporal pooling) ──────────────────
        frame_logits = None
        if self.use_tflh:
            frame_logits = self.frame_head(x)                          # [B, T, 1]

        # ── Video-level head ─────────────────────────────────────────────
        context, temporal_weights = self.temporal_attention(x)
        x = self.dropout2(context)
        x = self.fc1(x)
        x = self.act(x)
        logits = self.fc2(x)                                           # [B, 1]

        # Expose cross-modal contexts for A/V agreement loss (set by CrossModalAttentionFusion)
        a_ctx_pooled = getattr(self.modal_fusion, "last_a_ctx", None)
        v_ctx_pooled = getattr(self.modal_fusion, "last_v_ctx", None)

        # ── MAA forward ──────────────────────────────────────────────────
        audio_logits = None
        video_logits = None
        if self.use_maa and a_ctx_pooled is not None and v_ctx_pooled is not None:
            audio_logits = self.audio_fake_head(a_ctx_pooled)   # [B, 1]
            video_logits = self.video_fake_head(v_ctx_pooled)   # [B, 1]

        attn = {
            "modal":        modal_weights,
            "temporal":     temporal_weights,
            "tdfs_l1":      tdfs_l1,
            "soft_mask":    soft_mask,
            "a_ctx":        a_ctx_pooled,    # [B, H] or None
            "v_ctx":        v_ctx_pooled,    # [B, H] or None
            "audio_logits": audio_logits,    # [B, 1] or None
            "video_logits": video_logits,    # [B, 1] or None
            "cmis_incon":   cmis_incon,      # [B, T] A/V inconsistency map or None
            "cmis_spectral_w": cmis_spectral_w,  # [B, T, n_bands] band attn or None
        }
        return logits, frame_logits, attn
