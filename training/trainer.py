from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluation.metrics import compute_binary_metrics, find_best_threshold
from visualizations.plots import plot_training_curves
from utils.io import load_checkpoint, save_checkpoint


@dataclass
class TrainState:
    best_val_loss: float = float("inf")
    best_score: float = float("-inf")
    best_epoch: int = -1
    patience_counter: int = 0
    best_threshold: float = 0.5


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        audio_dim: int,
        video_dim: int,
        device: str,
    ) -> None:
        self.model = model.to(device)
        self.cfg = config
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.device = device

        # Locate the frozen ResNet feature block (layout: audio | resnet | gfc) so
        # visual channel dropout can target appearance features without disturbing
        # the audio (MFCC) or GFC spectral bands.
        gfc_cfg = config.get("features", {}).get("gfc", {})
        gfc_bands = int(gfc_cfg.get("n_bands", 0)) if gfc_cfg.get("enabled", False) else 0
        resnet_dim = max(self.video_dim - gfc_bands, 0)
        self._resnet_slice = (
            (self.audio_dim, self.audio_dim + resnet_dim) if resnet_dim > 0 else None
        )

        tr_cfg = self.cfg["training"]

        # Spectral feature slice (TALVAD-X layout = [audio | appearance | spectral]).
        # 0 for the legacy model, so spectral augmentation is a no-op there.
        self._spec_dim = int(self.cfg.get("model", {}).get("dims", {}).get("spectral", 0))
        self._epoch = 0  # updated by fit(); used for InfoNCE warm-up + DANN λ schedule

        # DANN domain classifier (domain generalization). Built only when enabled;
        # its parameters are optimized jointly with the detector.
        self._domain_clf = None
        if float(tr_cfg.get("dann_weight", 0.0)) > 0.0:
            from models.domain_adv import DomainClassifier
            h2 = int(self.cfg.get("model", {}).get("hidden_dim_2", 64))
            n_domains = int(tr_cfg.get("n_domains", 4))
            self._domain_clf = DomainClassifier(h2 * 2, n_domains).to(self.device)

        params = list(model.parameters())
        if self._domain_clf is not None:
            params += list(self._domain_clf.parameters())
        self.optimizer = AdamW(
            params,
            lr=tr_cfg["lr"],
            weight_decay=tr_cfg["weight_decay"],
        )
        self._label_smoothing = float(tr_cfg.get("label_smoothing", 0.0))
        pos_weight = tr_cfg.get("pos_weight")
        if pos_weight is not None:
            weight = torch.tensor([float(pos_weight)], device=device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        scheduler_type = tr_cfg.get("lr_scheduler", "reduce_on_plateau")
        warmup_epochs = int(tr_cfg.get("warmup_epochs", 0))
        if scheduler_type == "cosine" and warmup_epochs > 0:
            # Linear warm-up then cosine decay
            scheduler_warmup = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            scheduler_cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=max(tr_cfg["epochs"] - warmup_epochs, 1),
                eta_min=tr_cfg.get("lr_min", 1e-6),
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[scheduler_warmup, scheduler_cosine],
                milestones=[warmup_epochs],
            )
            self._scheduler_type = "cosine"
        elif scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=tr_cfg["epochs"],
                eta_min=tr_cfg.get("lr_min", 1e-6),
            )
            self._scheduler_type = "cosine"
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=tr_cfg["lr_scheduler_patience"],
            )
            self._scheduler_type = "reduce_on_plateau"

        self._av_agreement_weight = float(tr_cfg.get("av_agreement_weight", 0.0))
        self.use_amp = tr_cfg.get("mixed_precision", True)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and "cuda" in device)
        self.accum_steps = tr_cfg.get("gradient_accumulation_steps", 1)
        self.accum_counter = 0

        # SWAD / SWA (dense stochastic weight averaging) — a high-ROI domain-
        # generalization lever: averaging weights over the flatter back-half of
        # training lands in a wider minimum that transfers better off-domain.
        # The model is LayerNorm-only (no BatchNorm), so no statistic recalibration
        # is needed after averaging. When enabled, full schedule runs (early stopping
        # is suppressed) so the averaging window is actually collected.
        self._swa_enabled = bool(tr_cfg.get("swa_enabled", False))
        self._swa_start_frac = float(tr_cfg.get("swa_start_frac", 0.6))
        self._swa_state: Dict[str, torch.Tensor] | None = None
        self._swa_n = 0

        self.ckpt_dir = Path(tr_cfg["checkpoint_dir"])
        self.log_dir = Path(tr_cfg["logs_dir"]) / self.cfg["experiment_name"]
        self.writer = SummaryWriter(str(self.log_dir))

    def _write_training_status(
        self,
        *,
        status: str,
        current_epoch: int,
        total_epochs: int,
        state: TrainState,
        train_loss: float,
        val_loss: float,
        val_acc: float,
        bal_acc: float,
        auc: float,
    ) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment": self.cfg["experiment_name"],
            "status": status,
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "best_epoch": state.best_epoch + 1 if state.best_epoch >= 0 else 0,
            "best_score": float(state.best_score),
            "best_metric": self.cfg["training"].get("checkpoint_metric", "val_loss"),
            "best_threshold": float(state.best_threshold),
            "patience_counter": int(state.patience_counter),
            "last_train_loss": float(train_loss),
            "last_val_loss": float(val_loss),
            "last_val_acc": float(val_acc),
            "last_bal_acc": float(bal_acc),
            "last_auc": float(auc),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        (self.log_dir / "training_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _format_duration(self, seconds: float) -> str:
        total_seconds = max(int(round(seconds)), 0)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        if minutes > 0:
            return f"{minutes}m {secs:02d}s"
        return f"{secs}s"

    def _gpu_memory_str(self) -> str | None:
        if "cuda" not in self.device or not torch.cuda.is_available():
            return None
        peak = torch.cuda.max_memory_allocated(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        torch.cuda.reset_peak_memory_stats(0)
        return f"{peak:.1f} / {total:.1f} GB"

    def _best_metric_display(self, state: "TrainState") -> Tuple[str, float]:
        """Resolve the human-readable name and value of the tracked best score.

        ``state.best_score`` stores ``-val_loss`` when the checkpoint metric is
        val_loss (so that "higher is better" holds uniformly), so flip the sign
        back for display. For all other metrics best_score is the metric itself.
        """
        name = self.cfg["training"].get("checkpoint_metric", "val_loss")
        value = -state.best_score if name == "val_loss" else state.best_score
        return name, value

    @staticmethod
    def _render_box(title: str, lines: list[str], width: int = 46) -> str:
        """Render a fixed-width box. Each row is padded/truncated to ``width`` so
        the borders always line up regardless of value lengths."""
        def row(text: str) -> str:
            text = " " + text
            text = text[:width] if len(text) > width else text
            return "│" + text + " " * (width - len(text)) + "│"

        out = ["┌" + "─" * width + "┐", row(title), "├" + "─" * width + "┤"]
        out += [row(l) for l in lines]
        out.append("└" + "─" * width + "┘")
        return "\n".join(out)

    def _print_training_start_banner(self, train_loader, val_loader) -> None:
        tr_cfg = self.cfg["training"]
        device_label = self.device
        if "cuda" in self.device and torch.cuda.is_available():
            device_label = f"cuda ({torch.cuda.get_device_name(0)})"
        pos_weight = tr_cfg.get("pos_weight")
        pos_weight_text = f"{float(pos_weight):.3f}" if pos_weight is not None else "n/a"
        sampler_text = "WeightedRandomSampler (balanced)" if tr_cfg.get("balanced_sampler", False) else "shuffle"
        print("============================================================")
        print("  Training started")
        print(f"  Model     : {self.model.__class__.__name__}")
        print(f"  Device    : {device_label}")
        print(f"  Epochs    : {tr_cfg['epochs']}")
        print(f"  Batch size: {tr_cfg['batch_size']}")
        print(f"  Train     : {len(train_loader.dataset)} samples")
        print(f"  Val       : {len(val_loader.dataset)} samples")
        print(f"  Optimizer : AdamW  lr={tr_cfg['lr']:.0e}  wd={tr_cfg['weight_decay']:.0e}")
        print(f"  Loss      : BCEWithLogitsLoss  pos_weight={pos_weight_text}")
        print(f"  Sampler   : {sampler_text}")
        print(f"  Checkpoint: {self.ckpt_dir / 'best_model.pt'}")
        print(f"  Metric    : {tr_cfg.get('checkpoint_metric', 'val_loss')}")
        print("============================================================")

    def _split_modal(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x[..., : self.audio_dim], x[..., self.audio_dim : self.audio_dim + self.video_dim]

    def _frame_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-frame localization loss: `bce` (default) | `focal` | `dice`.

        focal/dice down-weight the dominant real frames so the rare fake frames
        actually shape the gradient — the in-domain frame head is currently a
        weak localiser largely because plain BCE is swamped by real timesteps.
        """
        kind = str(self.cfg["training"].get("frame_loss", "bce")).lower()
        if kind == "focal":
            gamma = float(self.cfg["training"].get("focal_gamma", 2.0))
            ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            p = torch.sigmoid(logits)
            p_t = p * target + (1.0 - p) * (1.0 - target)
            return ((1.0 - p_t).clamp_min(0.0) ** gamma * ce).mean()
        if kind == "dice":
            p = torch.sigmoid(logits)
            num = 2.0 * (p * target).sum() + 1.0
            den = (p + target).sum() + 1.0
            return 1.0 - num / den
        return F.binary_cross_entropy_with_logits(logits, target)

    def _step(self, batch, train: bool = True):
        # Dataset returns (x, y), (x, y, frame_y), (x, y, domain_y, frame_y) [multisource],
        # or (x, y, frame_y, audio_y, video_y).
        domain_y = None
        if len(batch) == 5:
            x, y, frame_y, audio_y, video_y = batch
            frame_y = frame_y.to(self.device, non_blocking=True)   # [B, T]
            audio_y = audio_y.to(self.device, non_blocking=True)   # [B]
            video_y = video_y.to(self.device, non_blocking=True)   # [B]
        elif len(batch) == 4:
            # MultiSourceDataset: (x, y, domain, frame_label)
            x, y, domain_y, frame_y = batch
            frame_y  = frame_y.to(self.device, non_blocking=True)
            domain_y = domain_y.to(self.device, non_blocking=True)
            audio_y = video_y = None
        elif len(batch) == 3:
            x, y, frame_y = batch
            frame_y = frame_y.to(self.device, non_blocking=True)
            audio_y = video_y = None
        else:
            x, y = batch
            frame_y = audio_y = video_y = None

        x = x.to(self.device, non_blocking=True)
        y_hard = y.to(self.device, non_blocking=True).unsqueeze(-1).float()
        y = y_hard.clone()
        mixup_active = False

        if train and self._label_smoothing > 0.0:
            y = y * (1.0 - self._label_smoothing) + self._label_smoothing * 0.5

        if train:
            noise_std = self.cfg["training"].get("feature_noise_std", 0.0)
            if noise_std > 0.0:
                x = x + torch.randn_like(x) * noise_std

            # Temporal masking — forces BiGRU to distribute evidence across time.
            # Each timestep is independently zeroed with prob = mask_steps / T.
            t_mask_steps = int(self.cfg["training"].get("temporal_mask_steps", 0))
            if t_mask_steps > 0:
                T = x.shape[1]
                p_mask = t_mask_steps / T
                t_mask = (torch.rand(x.shape[0], T, 1, device=x.device) > p_mask).float()
                x = x * t_mask

            # Modality dropout — forces cross-attention to not rely on one modality.
            # Each sample: prob p → zero audio; prob p → zero video; 1-2p → keep both.
            mod_drop_p = float(self.cfg["training"].get("modality_dropout_p", 0.0))
            if mod_drop_p > 0.0:
                r = torch.rand(x.shape[0], device=x.device)
                audio_drop = r < mod_drop_p
                video_drop = (r >= mod_drop_p) & (r < 2.0 * mod_drop_p)
                x = x.clone()
                if audio_drop.any():
                    x[audio_drop, :, :self.audio_dim] = 0.0
                if video_drop.any():
                    x[video_drop, :, self.audio_dim:] = 0.0

            # Visual channel dropout — zero whole ResNet feature channels (per
            # sample, across all timesteps) to break reliance on identity/appearance
            # encoded in the frozen CNN features. Unlike modality dropout (drops the
            # entire visual block) and temporal masking (drops timesteps), this drops
            # individual appearance dimensions while leaving audio and GFC intact.
            vis_drop_p = float(self.cfg["training"].get("visual_channel_dropout_p", 0.0))
            if vis_drop_p > 0.0 and self._resnet_slice is not None:
                lo, hi = self._resnet_slice
                keep = (torch.rand(x.shape[0], hi - lo, device=x.device) > vis_drop_p).float()
                full = torch.ones(x.shape[0], x.shape[2], device=x.device)
                full[:, lo:hi] = keep
                x = x * full.unsqueeze(1)

            # Spectral augmentation — random per-band scaling + masking on the
            # spectral slice (TALVAD-X layout = [audio | appearance | spectral]).
            # Breaks memorisation of one generator's frequency fingerprint — the
            # feature most responsible for the cross-dataset inversion.
            spec_aug_p = float(self.cfg["training"].get("spectral_aug_p", 0.0))
            if spec_aug_p > 0.0 and self._spec_dim > 0:
                lo = x.shape[2] - self._spec_dim
                scale = 1.0 + (torch.rand(x.shape[0], self._spec_dim, device=x.device) * 2 - 1) * spec_aug_p
                keep  = (torch.rand(x.shape[0], self._spec_dim, device=x.device) > spec_aug_p).float()
                x = x.clone()
                x[..., lo:] = x[..., lo:] * (scale * keep).unsqueeze(1)

            # Instance normalization as augmentation — per-sample standardize a
            # fraction of the batch so the model is robust to the per-clip
            # normalization used at cross-dataset eval (label-free domain-shift fix).
            inst_p = float(self.cfg["training"].get("instance_norm_p", 0.0))
            if inst_p > 0.0:
                apply_in = torch.rand(x.shape[0], device=x.device) < inst_p
                if bool(apply_in.any()):
                    mu = x.mean(dim=(1, 2), keepdim=True)
                    sigma = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
                    x = torch.where(apply_in.view(-1, 1, 1), (x - mu) / sigma, x)

            # Pseudo-domains for single-source DANN — when no real domain labels
            # are provided (PackedSequenceDataset), split the batch into two
            # augmentation regimes and let the domain classifier try to tell them
            # apart; the GRL then forces representations invariant to that axis.
            if float(self.cfg["training"].get("dann_weight", 0.0)) > 0.0 and domain_y is None:
                domain_y = (torch.rand(x.shape[0], device=x.device) < 0.5).long()
                heavy = domain_y.bool()
                if bool(heavy.any()):
                    x = x.clone()
                    x[heavy] = x[heavy] + torch.randn_like(x[heavy]) * 0.10

            # Feature mixup — manifold-style regulariser for generalisation and
            # calibration (a strong, honest cross-dataset lever). Applied per
            # batch with prob mixup_p. Aux losses that need discrete modality
            # labels (A/V agreement, MAA) are skipped on mixed batches; the main
            # BCE, TFLH and CMIS losses use the soft (interpolated) labels.
            mixup_alpha = float(self.cfg["training"].get("mixup_alpha", 0.0))
            if mixup_alpha > 0.0 and torch.rand(1).item() < float(self.cfg["training"].get("mixup_p", 0.5)):
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(x.shape[0], device=x.device)
                x = lam * x + (1.0 - lam) * x[perm]
                y = lam * y + (1.0 - lam) * y[perm]
                if frame_y is not None:
                    frame_y = lam * frame_y + (1.0 - lam) * frame_y[perm]
                mixup_active = True

        audio_x, video_x = self._split_modal(x)

        tr_cfg = self.cfg["training"]
        loc_weight  = tr_cfg.get("localization_loss_weight", 0.3)
        tdfs_weight = tr_cfg.get("tdfs_l1_weight", 1.0)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", enabled=self.use_amp and "cuda" in self.device):
                logits, frame_logits, attn = self.model(audio_x, video_x)
                if domain_y is not None:
                    attn["domain_labels"] = domain_y

                # Video-level classification loss
                loss = self.criterion(logits, y)
                # Per-component loss breakdown (detached, for super-informative
                # logging). Only the BCE term is always present; the rest are
                # recorded only when their guard fires, so the printed table shows
                # exactly which objectives are live this run.
                comp: Dict[str, float] = {"bce": float(loss.detach())}

                # Frame-level localization loss (TFLH) — training only
                if frame_logits is not None and frame_y is not None and train:
                    # frame_logits: [B, T, 1] → [B, T]; frame_y: [B, T]
                    frame_loss = self._frame_loss(frame_logits.squeeze(-1), frame_y)
                    loss = loss + loc_weight * frame_loss
                    comp["tflh"] = float(frame_loss.detach())

                # TDFS sparsity regularisation — training only; not part of val metric
                if train:
                    tdfs_l1 = attn.get("tdfs_l1", torch.tensor(0.0, device=self.device))
                    loss = loss + tdfs_weight * tdfs_l1
                    if torch.is_tensor(tdfs_l1) and float(tdfs_l1.detach()) != 0.0:
                        comp["tdfs"] = float(tdfs_l1.detach())

                # A/V agreement loss — supervised on partial fakes only.
                # Full-synthesis fakes (both A and V replaced) are internally consistent
                # by construction, so demanding low sim for them conflicts with BCE.
                # We restrict the loss to: real → high sim; (audio XOR video) → low sim.
                # Jointly-modified samples are excluded from this objective.
                if train and self._av_agreement_weight > 0.0 and not mixup_active:
                    a_ctx = attn.get("a_ctx")
                    v_ctx = attn.get("v_ctx")
                    if a_ctx is not None and v_ctx is not None and audio_y is not None:
                        sim     = (1.0 + F.cosine_similarity(a_ctx, v_ctx, dim=-1)) / 2.0  # [0,1]
                        real    = ((audio_y + video_y) == 0).float()           # both real → want high sim
                        partial = (audio_y.bool() ^ video_y.bool()).float()    # A XOR V  → want low  sim
                        mask    = real + partial                               # excludes A=1 & V=1
                        av_loss = real * (1.0 - sim) + partial * sim
                        av_term = (av_loss * mask).sum() / mask.sum().clamp_min(1.0)
                        loss    = loss + self._av_agreement_weight * av_term
                        comp["av"] = float(av_term.detach())

                # Modality Attribution loss — separate heads for audio/video forgery
                # Trained with LAV-DF modify_audio / modify_video ground truth.
                # Enables inference-time attribution: which modality was manipulated?
                maa_weight = float(tr_cfg.get("modality_attribution_loss_weight", 0.0))
                if train and maa_weight > 0.0 and audio_y is not None and not mixup_active:
                    a_log = attn.get("audio_logits")
                    v_log = attn.get("video_logits")
                    if a_log is not None and v_log is not None:
                        maa_loss = (
                            F.binary_cross_entropy_with_logits(a_log.squeeze(-1), audio_y)
                            + F.binary_cross_entropy_with_logits(v_log.squeeze(-1), video_y)
                        ) / 2.0
                        loss = loss + maa_weight * maa_loss
                        comp["maa"] = float(maa_loss.detach())

                # CMIS inconsistency-localisation loss (optional) — directly
                # supervises the per-timestep A/V inconsistency map to be high
                # inside fake_periods. Distinct from TFLH (which reads the
                # post-BiGRU state): this supervises the pre-temporal cross-modal
                # disagreement signal itself. cmis_incon is already in (0,1).
                cmis_weight = float(tr_cfg.get("cmis_incon_loss_weight", 0.0))
                if train and cmis_weight > 0.0 and frame_y is not None:
                    cmis_incon = attn.get("cmis_incon")
                    if cmis_incon is not None:
                        # cmis_incon is a post-sigmoid probability. plain BCE on
                        # probabilities is on the autocast blocklist, so compute it
                        # in fp32 with autocast disabled for this op.
                        with torch.autocast(device_type="cuda", enabled=False):
                            cmis_loss = F.binary_cross_entropy(
                                cmis_incon.float().clamp(1e-6, 1.0 - 1e-6),
                                frame_y.float(),
                            )
                        loss = loss + cmis_weight * cmis_loss
                        comp["cmis"] = float(cmis_loss.detach())

                # InfoNCE audio-visual correspondence (CMIS-2) — REAL samples only.
                # Teaches a generator-agnostic notion of natural A/V synchrony; on
                # fakes the correspondence breaks, surfacing as high inconsistency.
                # Uses the pre-fusion normalised embeddings z_a / z_v from CMIS-2.
                ct_w = float(tr_cfg.get("contrastive_loss_weight", 0.0))
                warm = int(tr_cfg.get("contrastive_warmup_epochs", 0))
                if warm > 0:
                    ct_w *= min(1.0, (self._epoch + 1) / warm)
                if train and ct_w > 0.0 and not mixup_active:
                    z_a = attn.get("cmis_z_a")
                    z_v = attn.get("cmis_z_v")
                    if z_a is not None and z_v is not None:
                        real_mask = (y_hard.squeeze(-1) < 0.5)
                        if bool(real_mask.any()):
                            temp = float(tr_cfg.get("contrastive_temperature", 0.07))
                            za, zv = z_a[real_mask], z_v[real_mask]            # [R,T,C]
                            R, Tt, _ = za.shape
                            sim = torch.einsum("rtc,rsc->rts", za, zv) / temp  # [R,T,T]
                            tgt = torch.arange(Tt, device=za.device).expand(R, Tt)
                            ct_loss = F.cross_entropy(sim.reshape(R * Tt, Tt), tgt.reshape(R * Tt))
                            loss = loss + ct_w * ct_loss
                            comp["nce"] = float(ct_loss.detach())

                # DANN domain-adversarial loss (multi-source / pseudo-domain DG).
                # Requires attn["domain_labels"] and attn["temporal_context"].
                dann_w = float(tr_cfg.get("dann_weight", 0.0))
                if train and dann_w > 0.0:
                    domain_labels = attn.get("domain_labels")
                    temporal_ctx  = attn.get("temporal_context")
                    domain_clf    = getattr(self, "_domain_clf", None)
                    if domain_labels is not None and temporal_ctx is not None and domain_clf is not None:
                        d_logits = domain_clf(temporal_ctx)
                        dann_loss = F.cross_entropy(d_logits, domain_labels.long())
                        loss = loss + dann_w * dann_loss
                        comp["dann"] = float(dann_loss.detach())

                loss = loss / self.accum_steps

            # Expose the per-component breakdown for run_epoch to accumulate.
            comp["total"] = float((loss * self.accum_steps).detach())
            self._step_components = comp

            if train:
                self.scaler.scale(loss).backward()
                self.accum_counter += 1
                if self.accum_counter % self.accum_steps == 0:
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg["training"]["gradient_clip"]
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

        return float(loss.detach().cpu().item() * self.accum_steps)

    def run_epoch(self, loader, train: bool = True) -> float:
        self.model.train(mode=train)
        losses = []
        self.accum_counter = 0
        comp_sums: Dict[str, float] = {}
        comp_n = 0

        # tqdm redraws a single line with carriage returns — great on an
        # interactive terminal, but through a pipe (e.g. the ablation runner or a
        # redirected log) every redraw becomes a NEW line, flooding the output.
        # So: live bar only on a real TTY; otherwise disable it and print a
        # compact progress line a few times per epoch.
        interactive = bool(getattr(sys.stdout, "isatty", lambda: False)()) \
            and os.environ.get("TALVAD_NO_TQDM") != "1"
        n_batches = len(loader)
        phase = "train" if train else "val"
        pbar = tqdm(loader, leave=True, disable=not interactive)
        next_report = 0.0
        for i, batch in enumerate(pbar):
            loss = self._step(batch, train=train)
            losses.append(loss)
            # Accumulate the per-component loss breakdown exposed by _step.
            step_comp = getattr(self, "_step_components", None)
            if step_comp:
                for k, v in step_comp.items():
                    comp_sums[k] = comp_sums.get(k, 0.0) + v
                comp_n += 1
            comp_txt = ""
            if step_comp:
                comp_txt = " [" + " ".join(
                    f"{k}={step_comp[k]:.3f}" for k in step_comp if k != "total"
                ) + "]"
            if interactive:
                pbar.set_description(f"{phase} loss={loss:.4f}{comp_txt}")
            else:
                # ~5 checkpoints per epoch (0/25/50/75/100%) instead of a flood.
                frac = (i + 1) / max(n_batches, 1)
                if frac >= next_report:
                    print(f"   {phase} {int(frac * 100):3d}% · batch {i + 1}/{n_batches}"
                          f" · loss={loss:.4f}{comp_txt}", flush=True)
                    next_report += 0.25
        # Mean of each loss component over the epoch — read by fit() for the box.
        self.last_components = {k: v / max(comp_n, 1) for k, v in comp_sums.items()} if comp_n else {}
        # Final optimizer step if there are remaining accumulated gradients
        if train and self.accum_counter % self.accum_steps != 0:
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["training"]["gradient_clip"])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        return float(sum(losses) / max(len(losses), 1))

    def predict_probabilities(self, loader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        y_true = []
        y_prob = []
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0], batch[1]
                x = x.to(self.device)
                audio_x, video_x = self._split_modal(x)
                logits, _, _ = self.model(audio_x, video_x)
                prob = torch.sigmoid(logits)
                y_true.extend(y.numpy().astype(int).tolist())
                y_prob.extend(prob.squeeze(-1).cpu().numpy().tolist())
        return np.asarray(y_true, dtype=np.int64), np.asarray(y_prob, dtype=np.float32)

    def validate(self, loader) -> Tuple[float, np.ndarray, np.ndarray]:
        """Single pass over the validation set returning (mean_bce_loss, y_true, y_prob).

        Replaces the previous two-pass approach (run_epoch + predict_probabilities),
        halving per-epoch validation cost. The loss equals the BCE term only — the
        auxiliary localisation / TDFS / A-V / MAA losses are training-only, so this
        matches what run_epoch reported for the validation split.
        """
        self.model.eval()
        losses, y_true, y_prob = [], [], []
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0], batch[1]
                x = x.to(self.device, non_blocking=True)
                y_dev = y.to(self.device, non_blocking=True).unsqueeze(-1).float()
                audio_x, video_x = self._split_modal(x)
                logits, _, _ = self.model(audio_x, video_x)
                losses.append(float(self.criterion(logits, y_dev).item()))
                prob = torch.sigmoid(logits)
                y_true.extend(y.numpy().astype(int).tolist())
                y_prob.extend(prob.squeeze(-1).cpu().numpy().tolist())
        mean_loss = float(sum(losses) / max(len(losses), 1))
        return mean_loss, np.asarray(y_true, dtype=np.int64), np.asarray(y_prob, dtype=np.float32)

    def evaluate_accuracy(self, loader, threshold: float = 0.5) -> float:
        y_true, y_prob = self.predict_probabilities(loader)
        y_pred = (y_prob >= threshold).astype(int)
        return float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0

    def _write_validation_outputs(self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float, metrics: Dict) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        rows = ["label,probability,prediction"]
        for label, prob in zip(y_true, y_prob):
            rows.append(f"{int(label)},{float(prob):.8f},{int(prob >= threshold)}")
        (self.log_dir / "validation_predictions.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        serializable = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in metrics.items()}
        (self.log_dir / "validation_threshold.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    def _update_swa(self) -> None:
        """Fold the current weights into the running SWA average (float tensors
        only; non-float buffers keep their latest value)."""
        sd = self.model.state_dict()
        if self._swa_state is None:
            self._swa_state = {k: v.detach().clone().float() for k, v in sd.items()}
            self._swa_n = 1
            return
        self._swa_n += 1
        inv = 1.0 / self._swa_n
        for k, v in sd.items():
            if torch.is_floating_point(v):
                self._swa_state[k].mul_(1.0 - inv).add_(v.detach().float(), alpha=inv)
            else:
                self._swa_state[k] = v.detach().clone()

    def _finalize_swa(self, val_loader, state: "TrainState") -> "TrainState":
        """Load the SWA-averaged weights into the model, re-tune the threshold on
        val, and persist them as best_model.pt (the artifact eval/cross-dataset
        consume). The early-stop best is preserved as best_model_earlystop.pt."""
        if not self._swa_enabled or self._swa_n == 0 or self._swa_state is None:
            if self._swa_enabled:
                print("  [SWA] no snapshots collected (training ended before the SWA window) — keeping best_model.pt as-is")
            return state
        cur = self.model.state_dict()
        merged = {k: (self._swa_state[k].to(v.dtype) if (k in self._swa_state and torch.is_floating_point(v)) else v)
                  for k, v in cur.items()}
        self.model.load_state_dict(merged)

        val_loss, vy, vp = self.validate(val_loader)
        if self.cfg["training"].get("tune_threshold", False):
            metric = self.cfg["training"].get("threshold_metric", "balanced_accuracy")
            thr, tm = find_best_threshold(vy, vp, metric=metric)
        else:
            thr = float(self.cfg["evaluation"].get("threshold", 0.5))
            tm = compute_binary_metrics(vy, vp, threshold=thr)

        best_path = self.ckpt_dir / "best_model.pt"
        if best_path.exists():
            import shutil
            shutil.copyfile(best_path, self.ckpt_dir / "best_model_earlystop.pt")
        save_checkpoint(best_path, {
            "epoch": -1, "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(), "best_val_loss": float(val_loss),
            "checkpoint_metric": "swa", "checkpoint_score": float(tm.get("auc", 0.0)),
            "threshold": float(thr),
            "threshold_metrics": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in tm.items()},
        })
        state.best_threshold = float(thr)
        print(f"  [SWA] averaged {self._swa_n} epochs → val AUC={tm.get('auc', 0):.4f} "
              f"bal_acc={tm.get('balanced_accuracy', 0):.4f} thr={thr:.3f}  "
              f"(saved best_model.pt; early-stop → best_model_earlystop.pt)")
        return state

    def fit(self, train_loader, val_loader, resume_from: str | None = None):
        state = TrainState()
        start_epoch = 0
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        fit_start = time.perf_counter()
        stopped_early = False
        last_train_loss = 0.0
        last_val_loss = 0.0
        last_val_acc = 0.0
        last_bal_acc = 0.0
        last_auc = 0.0

        if resume_from:
            cp = load_checkpoint(resume_from, map_location=self.device)
            self.model.load_state_dict(cp["model"])
            self.optimizer.load_state_dict(cp["optimizer"])
            start_epoch = cp["epoch"] + 1
            state.best_val_loss = cp.get("best_val_loss", state.best_val_loss)

        epochs = self.cfg["training"]["epochs"]
        patience = self.cfg["training"]["early_stopping_patience"]
        self._print_training_start_banner(train_loader, val_loader)
        swa_start_epoch = int(epochs * self._swa_start_frac) if self._swa_enabled else epochs + 1
        if self._swa_enabled:
            print(f"  [SWA] enabled — averaging weights every epoch from epoch "
                  f"{swa_start_epoch + 1}/{epochs} (early stopping suppressed)")

        for epoch in range(start_epoch, epochs):
            epoch_start = time.perf_counter()
            self._epoch = epoch
            # DANN gradient-reversal strength ramps 0 -> λ_max over training
            # (Ganin schedule) so the domain-adversarial signal turns on gently.
            if self._domain_clf is not None:
                from models.domain_adv import dann_lambda
                self._domain_clf.set_lambda(
                    dann_lambda(epoch, epochs, float(self.cfg["training"].get("dann_lambda_max", 1.0)))
                )
            lr_now = self.optimizer.param_groups[0]["lr"]
            print("──────────────────────────────────────")
            print(f" Epoch {epoch + 1:>2} / {epochs}   lr = {lr_now:.2e}")
            print("──────────────────────────────────────")
            train_loss = self.run_epoch(train_loader, train=True)
            epoch_components = dict(getattr(self, "last_components", {}) or {})
            # Single validation pass yields both the loss and the probabilities
            # used for threshold tuning / metrics (was previously two full passes).
            # Find threshold on val first so train_acc and val_acc use the same cut.
            val_loss, val_y, val_prob = self.validate(val_loader)
            threshold = float(self.cfg["evaluation"].get("threshold", 0.5))
            threshold_metrics = None
            if self.cfg["training"].get("tune_threshold", False):
                metric = self.cfg["training"].get("threshold_metric", "balanced_accuracy")
                threshold, threshold_metrics = find_best_threshold(val_y, val_prob, metric=metric)
            train_acc = self.evaluate_accuracy(train_loader, threshold=threshold)
            val_acc = float(accuracy_score(val_y, (val_prob >= threshold).astype(int))) if len(val_y) else 0.0
            if self._scheduler_type == "cosine":
                self.scheduler.step()
            else:
                self.scheduler.step(val_loss)

            lr_now = self.optimizer.param_groups[0]["lr"]
            display_metrics = threshold_metrics or compute_binary_metrics(val_y, val_prob, threshold=threshold)
            bal_acc = float(display_metrics.get("balanced_accuracy", val_acc))
            auc_val = float(display_metrics.get("auc", 0.0))
            f1_val = float(display_metrics.get("f1", 0.0))

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            last_train_loss = train_loss
            last_val_loss = val_loss
            last_val_acc = val_acc
            last_bal_acc = bal_acc
            last_auc = auc_val

            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val_loss, epoch)
            self.writer.add_scalar("acc/train", train_acc, epoch)
            self.writer.add_scalar("acc/val", val_acc, epoch)
            self.writer.add_scalar("threshold/val", threshold, epoch)
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)
            if threshold_metrics:
                self.writer.add_scalar("metrics/balanced_accuracy", float(threshold_metrics.get("balanced_accuracy", 0)), epoch)
                self.writer.add_scalar("metrics/auc", float(threshold_metrics.get("auc", 0)), epoch)
                self.writer.add_scalar("metrics/f1", float(threshold_metrics.get("f1", 0)), epoch)

            checkpoint_metric = self.cfg["training"].get("checkpoint_metric", "val_loss")
            if checkpoint_metric == "val_loss":
                current_score = -val_loss
            elif checkpoint_metric == "val_accuracy":
                current_score = val_acc
            else:
                if threshold_metrics is None:
                    threshold_metrics = compute_binary_metrics(val_y, val_prob, threshold=threshold)
                    threshold_metrics["threshold"] = threshold
                    threshold_metrics["threshold_metric"] = "fixed"
                current_score = float(threshold_metrics.get(checkpoint_metric, val_acc))

            is_new_best = current_score > state.best_score
            if is_new_best:
                state.best_score = current_score
                state.best_val_loss = val_loss
                state.best_epoch = epoch
                state.patience_counter = 0
                state.best_threshold = threshold
                if threshold_metrics is None:
                    threshold_metrics = compute_binary_metrics(val_y, val_prob, threshold=threshold)
                    threshold_metrics["threshold"] = threshold
                    threshold_metrics["threshold_metric"] = "fixed"
                self._write_validation_outputs(val_y, val_prob, threshold, threshold_metrics)
                save_checkpoint(
                    self.ckpt_dir / "best_model.pt",
                    {
                        "epoch": epoch,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_val_loss": state.best_val_loss,
                        "checkpoint_metric": checkpoint_metric,
                        "checkpoint_score": current_score,
                        "threshold": threshold,
                        "threshold_metrics": {
                            k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in threshold_metrics.items()
                        },
                    },
                )
            else:
                state.patience_counter += 1

            epoch_secs = time.perf_counter() - epoch_start
            epochs_left = epochs - (epoch + 1)
            eta_secs = epoch_secs * epochs_left
            title = f"Epoch {epoch + 1} / {epochs}" + ("  [NEW BEST ★]" if is_new_best else "")
            metric_lines = [
                f"train_loss : {train_loss:.4f}   val_loss : {val_loss:.4f}",
                f"train_acc  : {train_acc:.4f}   val_acc  : {val_acc:.4f}",
                f"bal_acc    : {bal_acc:.4f}   auc      : {auc_val:.4f}",
                f"threshold  : {threshold:.3f}   f1       : {f1_val:.4f}",
                f"lr         : {lr_now:.2e}",
            ]
            # Per-component loss breakdown — shows which objectives are live and
            # how much each contributes (e.g. whether CMIS/InfoNCE are non-trivial).
            order = ["bce", "tflh", "cmis", "nce", "av", "maa", "tdfs", "dann"]
            comp_items = [(k, epoch_components[k]) for k in order if k in epoch_components]
            comp_items += [(k, v) for k, v in epoch_components.items() if k not in order and k != "total"]
            if comp_items:
                metric_lines.append("loss terms :")
                for i in range(0, len(comp_items), 2):
                    pair = comp_items[i:i + 2]
                    metric_lines.append("  " + "   ".join(f"{k}={v:.4f}" for k, v in pair))
            metric_lines.append(
                f"epoch time : {epoch_secs:5.1f}s   ETA : {eta_secs/60:5.1f}m"
            )
            best_name, best_value = self._best_metric_display(state)
            metric_lines.append(
                f"best so far: {best_name}={best_value:.4f} @ ep{state.best_epoch + 1}"
                f" (patience {state.patience_counter}/{patience})"
            )
            gpu_mem = self._gpu_memory_str()
            if gpu_mem is not None:
                metric_lines.append(f"GPU mem    : {gpu_mem}")
            print(self._render_box(title, metric_lines, width=58))

            self._write_training_status(
                status="training",
                current_epoch=epoch + 1,
                total_epochs=epochs,
                state=state,
                train_loss=train_loss,
                val_loss=val_loss,
                val_acc=val_acc,
                bal_acc=bal_acc,
                auc=auc_val,
            )

            if self._swa_enabled and epoch >= swa_start_epoch:
                self._update_swa()

            early_stopping_enabled = self.cfg["training"].get("early_stopping", True) and not self._swa_enabled
            if early_stopping_enabled and state.patience_counter >= patience:
                stopped_early = True
                best_name, best_value = self._best_metric_display(state)
                print(f"⚠  Early stopping triggered (patience={patience}, no improvement since epoch {state.best_epoch + 1})")
                print(f"   Best epoch : {state.best_epoch + 1}   best {best_name} : {best_value:.4f}")
                self._write_training_status(
                    status="stopped_early",
                    current_epoch=epoch + 1,
                    total_epochs=epochs,
                    state=state,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    bal_acc=bal_acc,
                    auc=auc_val,
                )
                break

        state = self._finalize_swa(val_loader, state)

        self.writer.flush()
        plot_training_curves(history, self.log_dir / "training_curves.png")
        total_time = time.perf_counter() - fit_start
        best_name, best_value = self._best_metric_display(state)
        print("============================================================")
        print("  Training complete")
        print(f"  Best epoch  : {state.best_epoch + 1}")
        print(f"  Best metric : {best_name} = {best_value:.4f}")
        print(f"  Threshold   : {state.best_threshold:.3f}")
        print(f"  Checkpoint  : {self.ckpt_dir / 'best_model.pt'}")
        print(f"  Total time  : {self._format_duration(total_time)}")
        print("============================================================")

        if not stopped_early:
            self._write_training_status(
                status="complete",
                current_epoch=epochs,
                total_epochs=epochs,
                state=state,
                train_loss=last_train_loss,
                val_loss=last_val_loss,
                val_acc=last_val_acc,
                bal_acc=last_bal_acc,
                auc=last_auc,
            )
        return state
