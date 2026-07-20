from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalAttentionFusion(nn.Module):
    def __init__(self, audio_dim: int, video_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, audio_x: torch.Tensor, video_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.tanh(self.audio_proj(audio_x))
        v = torch.tanh(self.video_proj(video_x))
        g = self.softmax(self.gate(torch.cat([a, v], dim=-1)))
        fused = g[..., 0:1] * a + g[..., 1:2] * v
        return fused, g


class CrossModalAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion between audio and video streams.

    Audio attends to video  → "given this sound, what visual context matches?"
    Video attends to audio  → "given this frame, what audio context matches?"

    Inconsistency between the two directions is the core forgery signal for
    audio-visual deepfakes — something the original scalar gate cannot detect.

    The time-pooled a_ctx and v_ctx are stored as instance attributes after
    each forward pass so the trainer can compute the A/V agreement loss without
    changing the return signature consumed by the rest of the model.
    """

    def __init__(self, audio_dim: int, video_dim: int, hidden_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        assert hidden_dim % n_heads == 0, f"hidden_dim {hidden_dim} must be divisible by n_heads {n_heads}"
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)

        self.audio_to_video = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=0.1)
        self.video_to_audio = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=0.1)

        self.norm_a = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)

        # Merge both cross-attended streams back to hidden_dim
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Set by forward(); read by trainer for A/V agreement loss
        self.last_a_ctx: torch.Tensor | None = None   # [B, H] time-pooled audio context
        self.last_v_ctx: torch.Tensor | None = None   # [B, H] time-pooled video context
        # Per-timestep contexts (un-pooled) consumed by CMIS spectral attention.
        self.last_a_ctx_seq: torch.Tensor | None = None   # [B, T, H]
        self.last_v_ctx_seq: torch.Tensor | None = None   # [B, T, H]

    def forward(self, audio_x: torch.Tensor, video_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.audio_proj(audio_x)   # [B, T, H]
        v = self.video_proj(video_x)   # [B, T, H]

        # Audio queries into Video — residual connection preserves original audio signal
        a_ctx, _ = self.audio_to_video(a, v, v)
        a_ctx = self.norm_a(a + a_ctx)

        # Video queries into Audio — residual connection preserves original video signal
        v_ctx, _ = self.video_to_audio(v, a, a)
        v_ctx = self.norm_v(v + v_ctx)

        fused = self.output_proj(torch.cat([a_ctx, v_ctx], dim=-1))  # [B, T, H]

        # Store time-pooled contexts for A/V agreement loss in trainer
        self.last_a_ctx = a_ctx.mean(dim=1)   # [B, H]
        self.last_v_ctx = v_ctx.mean(dim=1)   # [B, H]
        # Store per-timestep contexts for CMIS (inconsistency-guided spectral attn)
        self.last_a_ctx_seq = a_ctx           # [B, T, H]
        self.last_v_ctx_seq = v_ctx           # [B, T, H]

        # Return uniform weights as placeholder (no scalar gate here)
        dummy_weights = torch.full(
            (audio_x.size(0), audio_x.size(1), 2), 0.5,
            device=audio_x.device, dtype=audio_x.dtype,
        )
        return fused, dummy_weights


class CrossModalInconsistencySpectralAttention(nn.Module):
    """
    CMIS — Cross-Modal Inconsistency-Guided Spectral Attention.

    Novel contribution. Prior work treats frequency-domain artifact analysis
    (visual-only) and audio-visual synchrony (cross-modal) as *separate*
    detectors that are fused late. CMIS instead *couples* them: it derives a
    per-timestep audio-visual inconsistency signal from the cross-attention
    contexts and uses it to modulate per-band spectral attention over the GFC
    features. The hypothesis — operationalised here rather than merely asserted
    — is that GAN/upsampling spectral artifacts are most diagnostic at exactly
    the timesteps where audio and video desynchronise.

    Outputs an additive spectral embedding injected into the temporal stream,
    plus the per-timestep inconsistency map (for supervision / visualisation)
    and the per-band spectral attention weights.

    Stability: both the coupling strength (alpha) and the residual injection
    gate are initialised to 0, so at the start of training CMIS is a no-op and
    the model degrades gracefully to the baseline; both are learned end-to-end.

    Ablation knobs:
      coupling=False   → spectral attention independent of inconsistency
                         (isolates the *coupling* as the contribution: B1 vs B2)
    """

    def __init__(
        self,
        ctx_dim: int,
        n_bands: int,
        hidden_dim: int,
        spectral_dim: int = 32,
        coupling: bool = True,
    ) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.coupling = coupling

        # Per-timestep A/V inconsistency detector on the cross-modal contexts.
        # Reads [a_ctx, v_ctx, a_ctx - v_ctx] so it can key on the *difference*.
        self.incon_net = nn.Sequential(
            nn.Linear(ctx_dim * 3, ctx_dim),
            nn.GELU(),
            nn.Linear(ctx_dim, 1),
        )
        # Base per-band spectral importance from the GFC bands themselves.
        self.band_net = nn.Sequential(
            nn.Linear(n_bands, spectral_dim),
            nn.GELU(),
            nn.Linear(spectral_dim, n_bands),
        )
        # Learnable coupling strength; starts at 0 → inconsistency has no effect
        # until the model learns to use it.
        self.alpha = nn.Parameter(torch.zeros(1))
        # Project the attended spectral feature into the temporal hidden space.
        self.spectral_proj = nn.Linear(n_bands, hidden_dim)
        # Residual injection gate; starts at 0 (tanh(0)=0) → CMIS is a no-op at
        # init, so adding it never destabilises a previously-working baseline.
        self.inject_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        a_seq: torch.Tensor,   # [B, T, ctx_dim]
        v_seq: torch.Tensor,   # [B, T, ctx_dim]
        gfc: torch.Tensor,     # [B, T, n_bands]  (post-TDFS GFC sub-features)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        incon_logit = self.incon_net(torch.cat([a_seq, v_seq, a_seq - v_seq], dim=-1))  # [B,T,1]
        incon = torch.sigmoid(incon_logit)                                              # [B,T,1] in (0,1)

        band_logits = self.band_net(gfc)                                                # [B,T,n_bands]
        if self.coupling:
            # High inconsistency sharpens/amplifies spectral scrutiny at that step.
            band_logits = band_logits * (1.0 + self.alpha * incon)
        spectral_w = torch.sigmoid(band_logits)                                         # [B,T,n_bands]

        spectral_feat = gfc * spectral_w                                                # [B,T,n_bands]
        injection = torch.tanh(self.inject_gate) * self.spectral_proj(spectral_feat)    # [B,T,H]
        return injection, incon.squeeze(-1), spectral_w


class TemporalAttention(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)
        context = torch.sum(x * w.unsqueeze(-1), dim=1)
        return context, w


class ConsistencyModulatedSpectralTemporalAttention(nn.Module):
    """
    CMIS-2 — Consistency-Modulated Spectral-Temporal attention (TALVAD-X novelty).

    Redesign of the original CMIS, fixing its four structural defects:

      (1) Pre-fusion, contrastively-grounded consistency.
          The original read "inconsistency" from the *post* cross-attention
          contexts, which have already exchanged information (a laundered, weak
          signal). CMIS-2 derives it from the *un-fused* per-timestep audio and
          visual embeddings: it projects each into a shared space, L2-normalises,
          and takes their cosine correspondence. The normalised embeddings z_a/z_v
          are exposed so the trainer can add a temporal InfoNCE objective on REAL
          segments (audio[t] <-> video[t]); this teaches a *generator-agnostic*
          notion of natural audio-visual correspondence, so fakes — whatever their
          generator — surface as high inconsistency. (Distinct from AuViRe's
          reconstruction discrepancy: we use contrastive correspondence and route
          it into spectral attention.)

      (2) Non-degenerate, warm-started gates.
          The original coupling strength and injection gate both initialised to 0
          (tanh(0)=0), so CMIS injected *nothing* and barely received gradient.
          Here the coupling/temporal strengths use softplus and warm-start at ~1,
          so the mechanism contributes from epoch 0. `warm_init=False` reproduces
          the near-zero start for the ablation that proves this matters.

      (3) Enriched spectrum.
          Operates on the richer radial x angular FFT descriptor (features/
          spectral.py) instead of 8 scalar radial bands.

      (4) A real path to the decision.
          The attended spectral feature is returned to be *concatenated* into the
          temporal stream (not a zero-gated residual add), and a separate
          consistency feature is exposed, so the classifier can actually use both.

    Inputs (per timestep):
      a_emb [B,T,Da] : pre-fusion audio features  (MFCC+wav2vec2)
      v_emb [B,T,Dv] : pre-fusion visual appearance features (face EfficientNet)
      spec  [B,T,S]  : enriched spectral descriptor (face-crop FFT)

    Returns a dict:
      spectral_feat [B,T,spectral_hidden] : consistency-gated spectral feature
      incon_feat    [B,T,incon_hidden]    : projected inconsistency feature
      incon         [B,T]                  : per-timestep inconsistency in (0,1)
      z_a, z_v      [B,T,corr_dim]         : normalised embeddings (for InfoNCE)
      band_w        [B,T,S]                : per-band attention weights (figures)

    Ablation knobs:
      coupling=False  -> spectral attention independent of inconsistency (A1)
      warm_init=False -> coupling/temporal strengths start ~0 (A5 zero-init)
      motion=True     -> the visual correspondence side reads appearance ⊕ its
                         frame-to-frame delta, so "A/V consistency" keys on mouth
                         *motion* (which the audio actually drives) rather than
                         static identity/appearance. Cheap (computed from cached
                         appearance features) and the main cross-dataset lever for
                         making the inconsistency signal generator-agnostic.
    """

    # softplus(0.5413) == 1.0 (warm) ; softplus(-4.0) ~= 0.018 (~off)
    _WARM = 0.5413
    _COLD = -4.0

    def __init__(
        self,
        audio_dim: int,
        visual_dim: int,
        spectral_dim: int,
        corr_dim: int = 128,
        spectral_hidden: int = 64,
        incon_hidden: int = 16,
        band_mlp: int = 32,
        coupling: bool = True,
        warm_init: bool = True,
        motion: bool = False,
    ) -> None:
        super().__init__()
        self.spectral_in = spectral_dim
        self.coupling = coupling
        self.spectral_hidden = spectral_hidden
        self.incon_hidden = incon_hidden
        self.motion = motion

        # Pre-fusion correspondence projections. When motion is on, the visual
        # side ingests appearance concatenated with its temporal delta (2×dim).
        self.audio_proj = nn.Linear(audio_dim, corr_dim)
        self.video_proj = nn.Linear(visual_dim * (2 if motion else 1), corr_dim)

        # Per-band spectral importance.
        self.band_net = nn.Sequential(
            nn.Linear(spectral_dim, band_mlp),
            nn.GELU(),
            nn.Linear(band_mlp, spectral_dim),
        )

        # Learnable, warm-started coupling + temporal strengths (softplus -> >0).
        init = self._WARM if warm_init else self._COLD
        self.alpha_raw = nn.Parameter(torch.tensor(float(init)))   # spectral coupling
        self.tgate_raw = nn.Parameter(torch.tensor(float(init)))   # temporal emphasis

        # Projections into the temporal hidden space.
        self.spectral_proj = nn.Linear(spectral_dim, spectral_hidden)
        self.incon_proj = nn.Linear(1, incon_hidden)

    def forward(
        self,
        a_emb: torch.Tensor,  # [B,T,Da]
        v_emb: torch.Tensor,  # [B,T,Dv]
        spec: torch.Tensor,   # [B,T,S]
    ) -> Dict[str, torch.Tensor]:
        if self.motion:
            # Frame-to-frame appearance change (delta[0]=0). Concatenating it makes
            # the correspondence sensitive to articulation/motion, not just looks.
            delta = torch.zeros_like(v_emb)
            delta[:, 1:, :] = v_emb[:, 1:, :] - v_emb[:, :-1, :]
            v_in = torch.cat([v_emb, delta], dim=-1)        # [B,T,2*Dv]
        else:
            v_in = v_emb
        z_a = F.normalize(self.audio_proj(a_emb), dim=-1)   # [B,T,C]
        z_v = F.normalize(self.video_proj(v_in), dim=-1)    # [B,T,C]
        sim = (z_a * z_v).sum(dim=-1)                       # [B,T] cosine in [-1,1]
        incon = (1.0 - (1.0 + sim) * 0.5).clamp(0.0, 1.0)   # [B,T] high = inconsistent

        band_logits = self.band_net(spec)                  # [B,T,S]
        if self.coupling:
            strength = F.softplus(self.alpha_raw)           # >0, ~1 at warm init
            band_logits = band_logits * (1.0 + strength * incon.unsqueeze(-1))
        band_w = torch.sigmoid(band_logits)                # [B,T,S]

        # Temporal emphasis: multiplicatively boost (>=1) high-inconsistency steps.
        t_boost = 1.0 + F.softplus(self.tgate_raw) * incon  # [B,T]
        spec_att = spec * band_w * t_boost.unsqueeze(-1)    # [B,T,S]

        spectral_feat = self.spectral_proj(spec_att)        # [B,T,Hs]
        incon_feat = self.incon_proj(incon.unsqueeze(-1))   # [B,T,Hc]

        return {
            "spectral_feat": spectral_feat,
            "incon_feat": incon_feat,
            "incon": incon,
            "z_a": z_a,
            "z_v": z_v,
            "band_w": band_w,
        }
