"""
training/trainer_loc.py — trainer for the TALVAD-L localizer.

Separate from training/trainer.py (the detector trainer) so the working path is
untouched. Multi-task objective:

    L = w_clip * BCE(clip)                 # detection — keeps AUC
      + w_cls  * Focal(actionness)         # dense per-step forgery-ness
      + w_reg  * DIoU(boundary regression) # anchor-free localization (the AP lever)
      + w_cbm  * BCE(CBM boundary energy)  # correspondence change-point novelty
      + w_recon* Recon(reals)              # CMRD cross-modal reconstruction
      + w_nce  * InfoNCE(reals)            # CMIS-2 A/V correspondence

Probability / cosine losses (CBM, recon) and the IoU math run in fp32 with
autocast disabled (the autocast-blocklist lesson from training/trainer.py).
Dev model selection is on AP@0.5 via scripts/eval_localization.evaluate_localization.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from utils.io import save_checkpoint  # noqa: E402
from eval_localization import evaluate_localization  # noqa: E402


# ── losses ────────────────────────────────────────────────────────────────────

def focal_bce(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1.0 - p) * (1.0 - target)
    return ((1.0 - p_t).clamp_min(0.0) ** gamma * ce).mean()


def dfl_loss(logits: torch.Tensor, target: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Distribution Focal Loss: push softmax mass onto the two bins straddling the
    continuous target (in level-stride units). logits [P,2,reg_max+1], target [P,2]."""
    t = target.clamp(0, reg_max - 1e-3)
    lo = t.floor().long()
    hi = (lo + 1).clamp(max=reg_max)
    w_hi = t - lo.float()
    w_lo = 1.0 - w_hi
    lp = F.log_softmax(logits, dim=-1)
    loss_lo = -lp.gather(-1, lo.unsqueeze(-1)).squeeze(-1) * w_lo
    loss_hi = -lp.gather(-1, hi.unsqueeze(-1)).squeeze(-1) * w_hi
    return (loss_lo + loss_hi).mean()


def boundary_contrastive_loss(
    h: torch.Tensor,        # [B,T,D] level-0 features
    frame: torch.Tensor,    # [B,T] in {0,1}  (fake-region indicator)
    max_offset: int = 6,
    neg_margin: float = 0.2,
) -> torch.Tensor:
    """Sharpen the backbone at forgery boundaries (the redirected correspondence
    novelty). For small temporal offsets d, neighbouring features on the SAME side
    of a boundary are pulled together (smooth within a region) and features
    STRADDLING a boundary (label flips within d steps) are pushed apart below a
    cosine margin. This turns every real<->fake edge into a feature discontinuity
    the heads can localize tightly -> targets AP@0.95. UniCaCLF won the high-IoU
    tail with contrastive learning; this is the audio-visual-correspondence version."""
    h = F.normalize(h.float(), dim=-1)
    terms = []
    for d in range(1, max_offset + 1):
        a, b = h[:, :-d], h[:, d:]                       # [B,T-d,D]
        sim = (a * b).sum(-1)                            # [B,T-d] cosine
        same = (frame[:, :-d] == frame[:, d:]).float()
        diff = 1.0 - same
        pos = (same * (1.0 - sim)).sum() / same.sum().clamp_min(1.0)
        neg = (diff * F.relu(sim - neg_margin)).sum() / diff.sum().clamp_min(1.0)
        terms.append(pos + neg)
    return torch.stack(terms).mean()


def diou_1d_loss(ps, pe, ts, te, eps: float = 1e-7) -> torch.Tensor:
    inter = (torch.min(pe, te) - torch.max(ps, ts)).clamp(min=0.0)
    union = (pe - ps) + (te - ts) - inter
    iou = inter / union.clamp(min=eps)
    c = (torch.max(pe, te) - torch.min(ps, ts)).clamp(min=eps)
    pc, tc = 0.5 * (ps + pe), 0.5 * (ts + te)
    d = (pc - tc) ** 2
    diou = iou - d / (c * c + eps)
    return (1.0 - diou).mean()


class TrainerL:
    def __init__(self, model: nn.Module, cfg: Dict, offsets: Dict, device: str,
                 dev_eval_fn=None) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.tr = cfg["training"]
        self.offsets = offsets
        self.device = device
        # Optional injected dev evaluator (model -> metrics dict). Used by the DF1M
        # windowed path for sliding-window eval; default None keeps the LAV-DF
        # duration-normalized evaluate_localization path unchanged.
        self.dev_eval_fn = dev_eval_fn
        self.seq_len = int(self.tr["sequence_length"])
        self.dcfg = cfg["localization"]["decode"]

        self.opt = AdamW(model.parameters(), lr=self.tr["lr"], weight_decay=self.tr["weight_decay"])
        epochs = int(self.tr["epochs"])
        warm = int(self.tr.get("warmup_epochs", 0))
        if warm > 0:
            self.sched = SequentialLR(
                self.opt,
                [LinearLR(self.opt, 0.01, 1.0, total_iters=warm),
                 CosineAnnealingLR(self.opt, T_max=max(epochs - warm, 1), eta_min=self.tr.get("lr_min", 1e-6))],
                milestones=[warm],
            )
        else:
            self.sched = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=self.tr.get("lr_min", 1e-6))

        self.amp = bool(self.tr.get("mixed_precision", True)) and "cuda" in device
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.ckpt_dir = Path(self.tr["checkpoint_dir"]); self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._epoch = 0

        self.w = {
            "clip": float(self.tr.get("clip_loss_weight", 1.0)),
            "cls": float(self.tr.get("cls_loss_weight", 1.0)),
            "reg": float(self.tr.get("reg_loss_weight", 1.0)),
            "cbm": float(self.tr.get("cbm_loss_weight", 0.5)),
            "recon": float(self.tr.get("recon_loss_weight", 0.5)),
            "nce": float(self.tr.get("contrastive_loss_weight", 0.1)),
            "qual": float(self.tr.get("qual_loss_weight", 1.0)),
            "dfl": float(self.tr.get("dfl_loss_weight", 0.5)),
            "bcl": float(self.tr.get("bcl_loss_weight", 0.0)),
            "reg_coarse": float(self.tr.get("reg_coarse_loss_weight", 0.5)),
        }
        self.gamma = float(self.tr.get("focal_gamma", 2.0))
        self.nce_temp = float(self.tr.get("contrastive_temperature", 0.07))
        self.nce_warm = int(self.tr.get("contrastive_warmup_epochs", 0))

    def _slice(self, x):
        o = self.offsets
        return (x[..., o["audio"][0]:o["audio"][1]], x[..., o["artic"][0]:o["artic"][1]],
                x[..., o["appearance"][0]:o["appearance"][1]], x[..., o["spectral"][0]:o["spectral"][1]])

    def _losses(self, out, batch) -> Dict[str, torch.Tensor]:
        dev = self.device
        y = batch["y"].to(dev)
        cls_t = [c.to(dev) for c in batch["cls"]]
        reg_t = [r.to(dev) for r in batch["reg"]]
        cbm_t = batch["cbm_target"].to(dev)
        frame = batch["frame_label"].to(dev)
        comp: Dict[str, torch.Tensor] = {}

        comp["clip"] = F.binary_cross_entropy_with_logits(out["clip_logits"].squeeze(-1).float(), y)

        # focal actionness across levels
        cls_losses = [focal_bce(out["level_cls"][l].squeeze(-1).float(), cls_t[l], self.gamma)
                      for l in range(len(cls_t))]
        comp["cls"] = torch.stack(cls_losses).mean()

        # DIoU regression on positives, all levels (frame-unit intervals)
        ps_a, pe_a, ts_a, te_a = [], [], [], []
        for l in range(len(reg_t)):
            ctr = out["level_centers"][l].float().unsqueeze(0)         # [1,Tl]
            pr = out["level_reg"][l].float()                           # [B,Tl,2]
            tg = reg_t[l]                                              # [B,Tl,2]
            pos = cls_t[l] > 0.5
            if pos.any():
                ps_a.append((ctr - pr[..., 0])[pos]); pe_a.append((ctr + pr[..., 1])[pos])
                ts_a.append((ctr - tg[..., 0])[pos]); te_a.append((ctr + tg[..., 1])[pos])
        if ps_a:
            comp["reg"] = diou_1d_loss(torch.cat(ps_a), torch.cat(pe_a), torch.cat(ts_a), torch.cat(te_a))
        else:
            comp["reg"] = torch.zeros((), device=dev)

        # Deep-supervision DIoU on the PRE-REFINE (coarse) boundaries so the refine
        # head learns a genuine correction rather than the coarse head degrading.
        if out.get("level_reg_coarse") and out["level_reg_coarse"][0] is not None:
            psc, pec, tsc, tec = [], [], [], []
            for l in range(len(reg_t)):
                ctr = out["level_centers"][l].float().unsqueeze(0)
                pr = out["level_reg_coarse"][l].float()
                tg = reg_t[l]
                pos = cls_t[l] > 0.5
                if pos.any():
                    psc.append((ctr - pr[..., 0])[pos]); pec.append((ctr + pr[..., 1])[pos])
                    tsc.append((ctr - tg[..., 0])[pos]); tec.append((ctr + tg[..., 1])[pos])
            if psc:
                comp["reg_coarse"] = diou_1d_loss(torch.cat(psc), torch.cat(pec), torch.cat(tsc), torch.cat(tec))

        # DFL on the boundary distributions (sub-step precision), positives only.
        if out.get("level_reg_logits") and out["level_reg_logits"][0] is not None:
            reg_max = out["level_reg_logits"][0].shape[-1] - 1
            strides = out["level_strides"]
            dfl_terms = []
            for l in range(len(reg_t)):
                pos = cls_t[l] > 0.5
                if pos.any():
                    lg = out["level_reg_logits"][l].float()[pos]              # [P,2,reg_max+1]
                    tg = (reg_t[l][pos] / float(strides[l]))                  # [P,2] stride units
                    dfl_terms.append(dfl_loss(lg, tg, reg_max))
            if dfl_terms:
                comp["dfl"] = torch.stack(dfl_terms).mean()

        # Centerness / quality: BCE to sqrt(min(ds,de)/max(ds,de)) on positives, so
        # the decoder can rank tight (center) predictions above boundary ones.
        if "level_qual" in out:
            ql, qt = [], []
            for l in range(len(reg_t)):
                pos = cls_t[l] > 0.5
                if pos.any():
                    ds, de = reg_t[l][..., 0][pos], reg_t[l][..., 1][pos]
                    ctr = torch.sqrt((torch.min(ds, de) / torch.max(ds, de).clamp_min(1e-6)).clamp(0, 1))
                    ql.append(out["level_qual"][l].squeeze(-1).float()[pos]); qt.append(ctr)
            if ql:
                comp["qual"] = F.binary_cross_entropy_with_logits(torch.cat(ql), torch.cat(qt))

        # CBM boundary energy (prob in (0,1)) — fp32 BCE
        if "cbm_b" in out:
            comp["cbm"] = F.binary_cross_entropy(out["cbm_b"].float().clamp(1e-6, 1 - 1e-6), cbm_t)

        # CMRD reconstruction on REAL frames only (minimise error where content is real)
        if "recon_err" in out:
            real_mask = (1.0 - frame)                                  # [B,T0]
            denom = real_mask.sum().clamp_min(1.0)
            comp["recon"] = (out["recon_err"].float() * real_mask).sum() / denom

        # Boundary-contrastive (the redirected novelty) — sharpen feat0 at edges.
        if self.w["bcl"] > 0 and "feat0" in out:
            comp["bcl"] = boundary_contrastive_loss(
                out["feat0"], frame,
                max_offset=int(self.tr.get("bcl_max_offset", 6)),
                neg_margin=float(self.tr.get("bcl_neg_margin", 0.2)),
            )

        # InfoNCE A/V correspondence on REAL clips (CMIS-2 z_a/z_v)
        nce_w = self.w["nce"]
        if self.nce_warm > 0:
            nce_w *= min(1.0, (self._epoch + 1) / self.nce_warm)
        if nce_w > 0 and "z_a" in out and "z_v" in out:
            real = (y < 0.5)
            if bool(real.any()):
                za, zv = out["z_a"][real].float(), out["z_v"][real].float()
                R, T, _ = za.shape
                sim = torch.einsum("rtc,rsc->rts", za, zv) / self.nce_temp
                tgt = torch.arange(T, device=dev).expand(R, T)
                comp["nce"] = F.cross_entropy(sim.reshape(R * T, T), tgt.reshape(R * T))
        return comp

    def _step(self, batch, train: bool):
        x = batch["x"].to(self.device)
        a, ar, ap, sp = self._slice(x)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", enabled=self.amp):
                out = self.model(a, ar, ap, sp)
            with torch.autocast(device_type="cuda", enabled=False):
                comp = self._losses(out, batch)
                total = sum(self.w.get(k, 1.0) * v for k, v in comp.items())
            if train:
                self.scaler.scale(total).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tr.get("gradient_clip", 1.0))
                self.scaler.step(self.opt); self.scaler.update(); self.opt.zero_grad(set_to_none=True)
        return {k: float(v.detach()) for k, v in comp.items()}, float(total.detach())

    def _dev_eval(self, val_loader) -> Dict:
        """Dev metrics for checkpoint selection. Default = LAV-DF duration-normalized
        evaluator; DF1M injects a sliding-window evaluator via `dev_eval_fn`."""
        if self.dev_eval_fn is not None:
            return self.dev_eval_fn(self.model)
        return evaluate_localization(self.model, val_loader, self.offsets, self.seq_len,
                                     self.dcfg, self.device)

    def fit(self, train_loader, val_loader):
        epochs = int(self.tr["epochs"])
        patience = int(self.tr.get("early_stopping_patience", 8))
        sel_key = {"map": "mAP", "ap50": "AP@50", "ap75": "AP@75"}.get(
            str(self.tr.get("checkpoint_metric", "map")).lower(), "mAP")
        best_ap, best_ep, bad = -1.0, -1, 0
        from tqdm import tqdm
        interactive = bool(getattr(sys.stdout, "isatty", lambda: False)())
        for epoch in range(epochs):
            self._epoch = epoch
            self.model.train()
            t0 = time.perf_counter()
            csum: Dict[str, float] = {}; n = 0
            pbar = tqdm(train_loader, disable=not interactive, leave=False)
            for batch in pbar:
                comp, tot = self._step(batch, True)
                for k, v in comp.items():
                    csum[k] = csum.get(k, 0.0) + v
                n += 1
                if interactive:
                    pbar.set_description(f"ep{epoch+1} loss={tot:.3f}")
            self.sched.step()
            cmean = {k: v / max(n, 1) for k, v in csum.items()}

            m = self._dev_eval(val_loader)
            ap50, ap75, mAP = m.get("AP@50", 0.0), m.get("AP@75", 0.0), m.get("mAP", 0.0)
            auc = m.get("clip_auc", float("nan"))
            dt = time.perf_counter() - t0
            comp_txt = " ".join(f"{k}={cmean[k]:.3f}" for k in cmean)
            star = ""
            sel_val = float(m.get(sel_key, 0.0))
            if sel_val > best_ap:
                best_ap, best_ep, bad = sel_val, epoch, 0
                star = "  ★"
                save_checkpoint(self.ckpt_dir / "best_model.pt", {
                    "epoch": epoch, "model": self.model.state_dict(),
                    "ap50": float(ap50), "ap75": float(ap75), "mAP": float(mAP), "clip_auc": float(auc),
                    "decode": self.dcfg, "sel_metric": sel_key,
                })
            else:
                bad += 1
            print(f"  ep{epoch+1:02d}/{epochs}  [{comp_txt}]  "
                  f"dev AP@50={ap50:.4f} AP@75={ap75:.4f} mAP={mAP:.4f} clipAUC={auc:.4f}  "
                  f"{dt:.0f}s{star}", flush=True)
            if self.tr.get("early_stopping", True) and bad >= patience:
                print(f"  early stop (no dev {sel_key} gain since ep{best_ep+1})")
                break
        print(f"  [DONE] best dev {sel_key}={best_ap:.4f} @ ep{best_ep+1}  -> {self.ckpt_dir/'best_model.pt'}")
        return best_ap
