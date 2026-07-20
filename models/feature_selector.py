from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TemporalDifferentiableFeatureSelector(nn.Module):
    """
    TDFS — Temporal Differentiable Feature Selector.

    Replaces GBWSO's static binary mask with a learned, per-timestep soft mask
    trained end-to-end alongside the detector.  Solves the proxy mismatch in
    GBWSO (RandomForest on time-averaged features) by producing a different
    importance vector at every timestep.

    GBWSO integration: call init_from_prior_mask(best_mask) after GBWSO runs
    to seed the importance head bias so selected features start with high
    attention and unselected with low.  The network then refines this prior
    temporally during backprop.

    Args:
        n_features:    Total feature dimension (audio_dim + video_dim).
        selector_dim:  Hidden size of the internal BiGRU context net.
        temperature:   Sigmoid temperature — lower = sharper binary-like masks.
        l1_weight:     Sparsity regularisation weight returned in forward for
                       the caller to add to the training loss.
    """

    def __init__(
        self,
        n_features: int,
        selector_dim: int = 64,
        temperature: float = 0.5,
        l1_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        self.n_features  = n_features
        self.temperature = temperature
        self.l1_weight   = l1_weight

        # Bidirectional GRU reads the full sequence for context at each step
        self.context_net = nn.GRU(
            n_features, selector_dim,
            batch_first=True, bidirectional=True,
        )
        # Projects context to per-feature importance logits
        self.importance_head = nn.Linear(selector_dim * 2, n_features)

        # Initialise importance head to neutral (sigmoid(0) = 0.5)
        nn.init.zeros_(self.importance_head.weight)
        nn.init.zeros_(self.importance_head.bias)

    # ------------------------------------------------------------------
    def init_from_prior_mask(self, prior_mask: np.ndarray) -> None:
        """
        Seed importance_head bias from a GBWSO binary mask.

        Selected features (mask=1) get logit +2  → sigmoid ≈ 0.88  (high)
        Rejected features (mask=0) get logit -2  → sigmoid ≈ 0.12  (low)

        The network will refine this temporally during training.
        """
        with torch.no_grad():
            bias = torch.from_numpy(prior_mask.astype(np.float32)) * 4.0 - 2.0
            self.importance_head.bias.data.copy_(bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                          # [B, T, D]
    ):
        """
        Returns:
            x_masked   [B, T, D]  — soft-masked features
            soft_mask  [B, T, D]  — importance weights in (0, 1)
            l1_loss    scalar     — sparsity penalty (add to total loss)
        """
        context, _ = self.context_net(x)                          # [B, T, 2*sel]
        logits     = self.importance_head(context)                 # [B, T, D]
        soft_mask  = torch.sigmoid(logits / self.temperature)     # [B, T, D]

        x_masked  = x * soft_mask
        l1_loss   = self.l1_weight * soft_mask.mean()

        return x_masked, soft_mask, l1_loss
