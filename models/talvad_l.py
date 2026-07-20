"""
TALVAD-L — anchor-free, multi-scale temporal-pyramid LOCALIZER for audio-visual
deepfake forgery localization (LAV-DF; foundation for AV-Deepfake1M).

Separate additive path — does NOT touch the working TALVAD-X detector
(models/talvad_x.py) or the v8 BiGRU detector. Reuses the CMIS-2 correspondence
mechanism verbatim from models/attention.py.

Design (see docs/METHOD_L.md / the approved plan):

  Streams (per timestep, on a fixed seq_len grid):
      audio    = [MFCC | WavLM-large(PCA)]          -> correspondence audio side
      artic    = auto_avsr lip-Conformer (PCA)       -> correspondence VISUAL side
      appear   = EfficientNet-B0 face (PCA)          -> artifact stream
      spectral = radial x angular FFT descriptor     -> artifact / spectral side

  Front-end:
      CMIS-2(audio, artic, spectral) -> incon(t), z_a, z_v, spectral_feat, incon_feat
      CMRD(z_a, z_v)                 -> per-step cross-modal reconstruction error e(t)
      CrossModalAttentionFusion(audio, [artic|appear|spectral]) -> fused(t)
      pyramid_in = [fused | spectral_feat | incon_feat | proj(appear) | e(t)]

  Tail (ActionFormer-style):
      input proj -> D + sinusoidal PE
      L levels, each = stack of Transformer encoder layers; strided Conv1d between
      shared-weight heads applied at every level:
        (A) actionness  -> per-step P(inside fake)           [focal]
        (B) boundary    -> per-step (d_start, d_end) >= 0     [DIoU]
        (C) CBM         -> boundary energy b(t) (level 0)     [Gaussian-edge BCE]
      clip head: attention-pool level-0 -> clip logit         [BCE, keeps AUC]

  Decode score (eval) = actionness * CBM-gate(b) * recon-gate(e).

The model only PRODUCES per-level cls/reg + the aux signals. Center-based label
assignment, the losses, and Soft-NMS decoding live in training/trainer_loc.py and
scripts/eval_localization.py so the architecture stays decoupled from the protocol.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import (
    ConsistencyModulatedSpectralTemporalAttention,
    CrossModalAttentionFusion,
    TemporalAttention,
)


# ── building blocks ───────────────────────────────────────────────────────────

def sinusoidal_pe(T: int, dim: int, device, dtype) -> torch.Tensor:
    """Standard sinusoidal positional encoding -> [1, T, dim]."""
    pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    div = torch.exp(-math.log(10000.0) * i / dim)
    pe = torch.zeros(T, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0).to(dtype)


class CrossModalReconstructionDecoder(nn.Module):
    """CMRD — reconstruct one modality's correspondence embedding from the other.

    Trained on REAL frames only (the trainer masks the loss); reconstruction is
    easy where audio and video naturally correspond and breaks inside forged
    spans, so the per-step error e(t) is a forgery/boundary signal INDEPENDENT of
    the CMIS-2 inconsistency. Operates on the L2-normalised CMIS-2 embeddings
    z_a/z_v, so e(t) is a cosine-distance in (0, 1).
    """

    def __init__(self, corr_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.a2v = nn.Sequential(nn.Linear(corr_dim, hidden), nn.GELU(), nn.Linear(hidden, corr_dim))
        self.v2a = nn.Sequential(nn.Linear(corr_dim, hidden), nn.GELU(), nn.Linear(hidden, corr_dim))

    def forward(self, z_a: torch.Tensor, z_v: torch.Tensor) -> Dict[str, torch.Tensor]:
        v_hat = F.normalize(self.a2v(z_a), dim=-1)
        a_hat = F.normalize(self.v2a(z_v), dim=-1)
        e_v = 1.0 - (v_hat * z_v).sum(dim=-1)           # [B,T] in [0,2]
        e_a = 1.0 - (a_hat * z_a).sum(dim=-1)
        e = (0.5 * (e_v + e_a)).clamp(0.0, 1.0)          # [B,T] -> usable as a gate
        return {"recon_err": e, "recon_err_av": e_v, "recon_err_va": e_a}


class CorrespondenceBoundaryModule(nn.Module):
    """CBM — boundary energy from the CHANGE-POINT structure of correspondence.

    A spliced fake has two edges; at each edge the A/V correspondence changes
    regime. We feed the temporal derivatives of the (normalised) correspondence
    embeddings and of the inconsistency curve into a small temporal conv and
    predict a per-step boundary energy b(t). Supervised to peak at GT edges and
    used to gate the boundary-regression confidence at decode.
    """

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        # 4 input channels: ||Δz_a||, ||Δz_v||, |Δincon|, incon level.
        self.net = nn.Sequential(
            nn.Conv1d(4, hidden, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    @staticmethod
    def _delta_norm(z: torch.Tensor) -> torch.Tensor:
        d = torch.zeros(z.shape[0], z.shape[1], device=z.device, dtype=z.dtype)
        d[:, 1:] = (z[:, 1:] - z[:, :-1]).norm(dim=-1)
        return d  # [B,T]

    def forward(self, z_a: torch.Tensor, z_v: torch.Tensor, incon: torch.Tensor) -> torch.Tensor:
        dza = self._delta_norm(z_a)
        dzv = self._delta_norm(z_v)
        dinc = torch.zeros_like(incon)
        dinc[:, 1:] = (incon[:, 1:] - incon[:, :-1]).abs()
        feats = torch.stack([dza, dzv, dinc, incon], dim=1)   # [B,4,T]
        b = torch.sigmoid(self.net(feats)).squeeze(1)         # [B,T]
        return b


class ConvHead(nn.Module):
    """Shared 1-D conv head applied at every pyramid level."""

    def __init__(self, dim: int, out: int, n_conv: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(n_conv):
            layers += [nn.Conv1d(dim, dim, kernel_size=3, padding=1), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv1d(dim, out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x [B,D,T] -> [B,out,T]
        return self.out(self.body(x))


class PyramidLevel(nn.Module):
    """Transformer encoder block (n layers) at one temporal resolution."""

    def __init__(self, dim: int, n_layers: int, n_heads: int, ffn: int, dropout: float) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=ffn, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # key_padding_mask: True = PAD (ignored). [B,T]
        return self.enc(x, src_key_padding_mask=key_padding_mask)


# ── the localizer ─────────────────────────────────────────────────────────────

class TALVADLLocalizer(nn.Module):
    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        dims = cfg["dims"]
        self.audio_dim = int(dims["audio"])
        self.artic_dim = int(dims["artic"])
        self.appear_dim = int(dims["appearance"])
        self.spec_dim = int(dims["spectral"])

        h1 = int(cfg.get("fusion_hidden", 128))
        d_model = int(cfg.get("d_model", 256))
        n_heads = int(cfg.get("n_heads", 8))
        ffn = int(cfg.get("ffn", d_model * 2))
        dropout = float(cfg.get("dropout", 0.1))
        self.n_levels = int(cfg.get("n_levels", 4))
        n_layers = int(cfg.get("layers_per_level", 2))
        app_proj_dim = int(cfg.get("app_proj_dim", 128))

        cmis = cfg.get("cmis", {}) or {}
        corr_dim = int(cmis.get("corr_dim", 128))
        sh = int(cmis.get("spectral_hidden", 64))
        ih = int(cmis.get("incon_hidden", 16))
        self.contrastive = bool(cmis.get("contrastive", True))  # read by trainer

        # ── front-end (reused CMIS-2 + cross-modal fusion) ──
        self.cmis = ConsistencyModulatedSpectralTemporalAttention(
            audio_dim=self.audio_dim,
            visual_dim=self.artic_dim,            # AVSR articulation is the correspondence visual side
            spectral_dim=self.spec_dim,
            corr_dim=corr_dim,
            spectral_hidden=sh,
            incon_hidden=ih,
            band_mlp=int(cmis.get("band_mlp", 32)),
            coupling=bool(cmis.get("coupling", True)),
            warm_init=bool(cmis.get("warm_init", True)),
            motion=False,                         # redundant: AVSR already encodes articulation
        )
        video_dim = self.artic_dim + self.appear_dim + self.spec_dim
        self.fusion = CrossModalAttentionFusion(self.audio_dim, video_dim, h1, n_heads=min(n_heads, h1))

        self.use_cmrd = bool(cfg.get("cmrd", {}).get("enabled", True))
        if self.use_cmrd:
            self.cmrd = CrossModalReconstructionDecoder(corr_dim, int(cfg.get("cmrd", {}).get("hidden", 128)))

        self.use_cbm = bool(cfg.get("cbm", {}).get("enabled", True))
        if self.use_cbm:
            self.cbm = CorrespondenceBoundaryModule(int(cfg.get("cbm", {}).get("hidden", 32)))

        self.app_proj = nn.Linear(self.appear_dim, app_proj_dim)

        # pyramid input width = fused + spectral_feat + incon_feat + app_proj + recon_err(1)
        pyr_in = h1 + sh + ih + app_proj_dim + (1 if self.use_cmrd else 0)
        self.in_proj = nn.Linear(pyr_in, d_model)
        self.in_norm = nn.LayerNorm(d_model)
        self.in_drop = nn.Dropout(dropout)

        # ── pyramid ──
        self.levels = nn.ModuleList(
            [PyramidLevel(d_model, n_layers, n_heads, ffn, dropout) for _ in range(self.n_levels)]
        )
        self.downsamples = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
             for _ in range(self.n_levels - 1)]
        )
        self.down_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.n_levels - 1)])

        # ── shared heads ──
        self.cls_head = ConvHead(d_model, 1, n_conv=int(cfg.get("head_conv", 2)), dropout=dropout)
        # Distribution Focal Loss (DFL): predict a softmax over reg_max+1 offset bins
        # per side (in level-stride units) and take the expectation -> sub-step boundary
        # precision (the high-IoU lever). Falls back to scalar regression when off.
        self.use_dfl = bool(cfg.get("use_dfl", False))
        self.reg_max = int(cfg.get("reg_max", 16))
        reg_out = 2 * (self.reg_max + 1) if self.use_dfl else 2
        self.reg_head = ConvHead(d_model, reg_out, n_conv=int(cfg.get("head_conv", 2)), dropout=dropout)
        self.register_buffer("dfl_bins", torch.arange(self.reg_max + 1).float())
        # Centerness / localization-quality head (FCOS-style): ranks tight-boundary
        # steps above boundary steps so the decoder selects the well-localized box.
        self.qual_head = ConvHead(d_model, 1, n_conv=int(cfg.get("head_conv", 2)), dropout=dropout)
        # per-level learnable positive scale on the scalar-regression output (non-DFL path).
        self.reg_scales = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(2 ** i))) for i in range(self.n_levels)]
        )
        # Propose->refine stage: a light shared head predicts a BOUNDED residual
        # correction (in stride units) to the coarse (d_start, d_end). Deep-supervised
        # alongside the coarse output; decode uses the refined boundary -> tighter
        # high-IoU edges (the AP@0.9/0.95 lever). Off unless cfg["use_refine"].
        self.use_refine = bool(cfg.get("use_refine", False))
        if self.use_refine:
            self.refine_head = ConvHead(d_model, 2, n_conv=int(cfg.get("head_conv", 2)), dropout=dropout)
            self.refine_max = float(cfg.get("refine_max_offset", 4.0))   # max |correction|, stride units

        # clip head (keeps detection / AUC alive)
        self.clip_pool = TemporalAttention(d_model)
        self.clip_fc = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(d_model // 2, 1))

        self.level_strides = [2 ** i for i in range(self.n_levels)]   # in level-0 step units

    # ── helpers ──
    @staticmethod
    def _downsample_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        # mask True=PAD, [B,T]; stride-2 max-pool keeps a step PAD only if both children PAD.
        m = mask.float().unsqueeze(1)                       # [B,1,T]
        m = F.max_pool1d(m, kernel_size=3, stride=2, padding=1)
        return m.squeeze(1) > 0.5

    def level_centers(self, level: int, T0: int, device) -> torch.Tensor:
        """Level-0 coordinate of each step at `level` (for assignment/decoding)."""
        stride = self.level_strides[level]
        Tl = math.ceil(T0 / stride) if level > 0 else T0
        idx = torch.arange(Tl, device=device, dtype=torch.float32)
        return (idx + 0.5) * stride                          # [Tl]

    # ── forward ──
    def forward(
        self,
        audio: torch.Tensor,      # [B,T,audio_dim]
        artic: torch.Tensor,      # [B,T,artic_dim]
        appear: torch.Tensor,     # [B,T,appear_dim]
        spectral: torch.Tensor,   # [B,T,spec_dim]
        pad_mask: Optional[torch.Tensor] = None,   # [B,T] True=PAD
    ) -> Dict:
        B, T0, _ = audio.shape

        c = self.cmis(audio, artic, spectral)
        fused, _ = self.fusion(audio, torch.cat([artic, appear, spectral], dim=-1))

        parts = [fused, c["spectral_feat"], c["incon_feat"], self.app_proj(appear)]
        aux: Dict[str, torch.Tensor] = {
            "incon": c["incon"], "z_a": c["z_a"], "z_v": c["z_v"], "band_w": c["band_w"],
        }
        if self.use_cmrd:
            r = self.cmrd(c["z_a"], c["z_v"])
            aux["recon_err"] = r["recon_err"]
            parts.append(r["recon_err"].unsqueeze(-1))
        if self.use_cbm:
            aux["cbm_b"] = self.cbm(c["z_a"], c["z_v"], c["incon"])   # [B,T0]

        x = torch.cat(parts, dim=-1)
        x = self.in_drop(self.in_norm(self.in_proj(x)))
        x = x + sinusoidal_pe(T0, x.shape[-1], x.device, x.dtype)

        # pyramid: collect per-level cls/reg
        level_cls: List[torch.Tensor] = []
        level_reg: List[torch.Tensor] = []
        level_reg_coarse: List[Optional[torch.Tensor]] = []
        level_reg_logits: List[Optional[torch.Tensor]] = []
        level_qual: List[torch.Tensor] = []
        level_centers: List[torch.Tensor] = []
        mask = pad_mask
        feat = x
        clip_feat0 = None
        for lvl in range(self.n_levels):
            feat = self.levels[lvl](feat, key_padding_mask=mask)
            if lvl == 0:
                clip_feat0 = feat
            xc = feat.transpose(1, 2)                          # [B,D,Tl]
            cls = self.cls_head(xc).transpose(1, 2)            # [B,Tl,1]
            reg_raw = self.reg_head(xc).transpose(1, 2)        # [B,Tl,2 or 2*(reg_max+1)]
            stride = float(self.level_strides[lvl])
            if self.use_dfl:
                Bn, Tl, _ = reg_raw.shape
                logits = reg_raw.view(Bn, Tl, 2, self.reg_max + 1)
                e = (F.softmax(logits, dim=-1) * self.dfl_bins).sum(-1)   # [B,Tl,2] stride units
                reg = e * stride                                          # -> level-0 units
                level_reg_logits.append(logits)
            else:
                reg = F.relu(reg_raw) * self.reg_scales[lvl]
                level_reg_logits.append(None)
            qual = self.qual_head(xc).transpose(1, 2)          # [B,Tl,1] centerness logit
            if self.use_refine:
                delta = torch.tanh(self.refine_head(xc).transpose(1, 2)) * self.refine_max * stride
                reg_refined = (reg + delta).clamp_min(0.0)     # level-0 units, >= 0
                level_reg_coarse.append(reg)
                level_reg.append(reg_refined)
            else:
                level_reg_coarse.append(None)
                level_reg.append(reg)
            level_cls.append(cls)
            level_qual.append(qual)
            level_centers.append(self.level_centers(lvl, T0, feat.device))
            if lvl < self.n_levels - 1:
                feat = self.downsamples[lvl](feat.transpose(1, 2)).transpose(1, 2)
                feat = self.down_norms[lvl](feat)
                mask = self._downsample_mask(mask)

        # clip head on level-0
        ctx, _ = self.clip_pool(clip_feat0)                    # [B,D]
        clip_logits = self.clip_fc(ctx)                        # [B,1]

        return {
            "clip_logits": clip_logits,
            "feat0": clip_feat0,               # [B,T0,d_model] level-0 features (boundary-contrastive)
            "level_cls": level_cls,            # list of [B,Tl,1]
            "level_reg": level_reg,            # list of [B,Tl,2]  refined (or coarse if no refine)
            "level_reg_coarse": level_reg_coarse,  # list of [B,Tl,2] pre-refine, or None
            "level_reg_logits": level_reg_logits,  # list of [B,Tl,2,reg_max+1] (DFL) or None
            "level_qual": level_qual,          # list of [B,Tl,1]  centerness logits
            "level_centers": level_centers,    # list of [Tl] level-0 coords
            "level_strides": self.level_strides,
            **aux,
        }


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = {
        "dims": {"audio": 632, "artic": 256, "appearance": 256, "spectral": 34},
        "fusion_hidden": 128, "d_model": 256, "n_heads": 8, "ffn": 512, "dropout": 0.1,
        "n_levels": 4, "layers_per_level": 2, "app_proj_dim": 128,
        "cmis": {"corr_dim": 128, "spectral_hidden": 64, "incon_hidden": 16, "coupling": True, "warm_init": True},
        "cmrd": {"enabled": True, "hidden": 128},
        "cbm": {"enabled": True, "hidden": 32},
    }
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = TALVADLLocalizer(cfg).to(dev)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    B, T = 2, 192
    a = torch.randn(B, T, 632, device=dev)
    ar = torch.randn(B, T, 256, device=dev)
    ap = torch.randn(B, T, 256, device=dev)
    sp = torch.randn(B, T, 34, device=dev)
    pad = torch.zeros(B, T, dtype=torch.bool, device=dev); pad[0, 150:] = True   # clip 0 padded tail
    out = m(a, ar, ap, sp, pad_mask=pad)

    print(f"[OK] params={n_params/1e6:.2f}M  device={dev}")
    print(f"  clip_logits  {tuple(out['clip_logits'].shape)}")
    for i, (cls, reg, ctr) in enumerate(zip(out["level_cls"], out["level_reg"], out["level_centers"])):
        print(f"  level {i}: cls {tuple(cls.shape)}  reg {tuple(reg.shape)}  "
              f"centers[{ctr.numel()}] stride={out['level_strides'][i]}  reg_max={reg.max().item():.2f}")
    print(f"  incon {tuple(out['incon'].shape)}  recon_err {tuple(out['recon_err'].shape)}  "
          f"cbm_b {tuple(out['cbm_b'].shape)}  z_a {tuple(out['z_a'].shape)}")

    # backward sanity
    loss = (out["clip_logits"].mean()
            + sum(c.float().mean() for c in out["level_cls"])
            + sum(r.float().mean() for r in out["level_reg"])
            + out["cbm_b"].mean() + out["recon_err"].mean())
    loss.backward()
    g = [p.grad for p in m.parameters() if p.grad is not None]
    finite = all(torch.isfinite(gg).all() for gg in g)
    print(f"[OK] backward: {len(g)} tensors have grad, all finite={finite}, loss={float(loss):.4f}")
    assert out["clip_logits"].shape == (B, 1)
    assert out["level_cls"][0].shape == (B, T, 1)
    assert out["level_cls"][1].shape[1] == (T + 1) // 2 or out["level_cls"][1].shape[1] == T // 2
    assert finite
    print("TALVAD-L SMOKE OK")
