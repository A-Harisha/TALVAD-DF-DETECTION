"""
Domain-adversarial (DANN) head for TALVAD-X generalization.

Purpose: train representations that are discriminative for FAKE/REAL but
invariant to *which dataset* they came from. A gradient-reversal layer (GRL)
before a domain classifier makes features that fool the classifier — i.e.,
they carry no dataset-identity information — while the main detector still
maximises forgery signal. Combined with CMIS-2's generator-agnostic consistency
gate, this is the primary mechanism for killing the cross-dataset inversion.

Design:
  - GradientReversalLayer: standard Ganin/Lempitsky 2016 GRL. lambda controls
    the reversal strength (annealed from 0→lambda_max during training).
  - DomainClassifier: reads the time-pooled temporal hidden state [B, H2*2] from
    the BiGRU stack, outputs [B, n_domains] domain logits.
  - Used by the trainer when cfg['training']['dann_weight'] > 0.

For single-source training (LAV-DF only), *pseudo-domains* are created from the
augmentation regime (one forward pass uses feature noise + heavy visual dropout;
another uses mild augmentation). The domain classifier is then trained to tell
them apart — forcing representations that are robust to augmentation variation,
which generalises to codec/distribution shifts in target datasets.

For multi-source (LAV-DF + FakeAVCeleb + ...), domain = dataset index.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lam * grad_output, None


class GradientReversalLayer(nn.Module):
    """Passes x unchanged in forward; negates × lam in backward."""
    def __init__(self, lam: float = 1.0) -> None:
        super().__init__()
        self.lam = lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lam)


class DomainClassifier(nn.Module):
    def __init__(self, in_dim: int, n_domains: int, hidden: int = 64) -> None:
        super().__init__()
        self.grl = GradientReversalLayer()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_domains),
        )

    def set_lambda(self, lam: float) -> None:
        self.grl.lam = float(lam)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, in_dim] (time-pooled BiGRU output from TemporalAttention)."""
        return self.net(self.grl(h))


def dann_lambda(epoch: int, total_epochs: int, lambda_max: float = 1.0) -> float:
    """Standard DANN lambda schedule: ramp from 0 to lambda_max over training."""
    p = epoch / max(total_epochs, 1)
    return lambda_max * (2.0 / (1.0 + float(torch.exp(torch.tensor(-10.0 * p)))) - 1.0)


class InstanceFeatureNorm(nn.Module):
    """
    Test-time domain-shift mitigation: standardise each sample's features
    independently (mean-0, std-1) along the feature dimension.

    Why: the cross-dataset AUC inversion (0.33 on FakeAVCeleb) is caused by the
    LAV-DF-fit global StandardScaler pushing FakeAVCeleb features into OOD
    regions. Instance norm is label-free (no target data required) and brings
    each sample to a canonical distribution regardless of codec/loudness.

    Usage: apply to x [B, T, D] before the model forward pass during inference
    on a target dataset (or optionally during training as an augmentation).
    """
    def forward(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # Normalise per-sample, across both time and feature dims.
        mu = x.mean(dim=(-2, -1), keepdim=True)
        sigma = x.std(dim=(-2, -1), keepdim=True).clamp_min(eps)
        return (x - mu) / sigma


if __name__ == "__main__":
    B, H = 8, 128
    dc = DomainClassifier(H, n_domains=3)
    dc.set_lambda(0.5)
    h = torch.randn(B, H, requires_grad=True)
    logits = dc(h)
    assert logits.shape == (B, 3)
    logits.sum().backward()
    assert h.grad is not None
    print(f"[OK] DomainClassifier: logits {logits.shape}  GRL reverses grad")

    inst = InstanceFeatureNorm()
    x = torch.randn(B, 48, 128) * 5 + 3  # mean≠0, std≠1
    xn = inst(x)
    print(f"[OK] InstanceNorm: mean={xn.mean():.4f} std={xn.std():.4f} (should be ~0, ~1)")
    assert abs(float(xn.mean())) < 0.1 and abs(float(xn.std()) - 1.0) < 0.1

    lams = [dann_lambda(e, 40) for e in [0, 10, 20, 40]]
    print(f"[OK] DANN lambda schedule: {[round(l,3) for l in lams]} (0 -> 1)")
    print("[OK] domain_adv.py self-test passed")
