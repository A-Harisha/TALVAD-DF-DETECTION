"""
datasets/multisource.py — unified multi-source dataset for TALVAD-X DG training.

Combines LAV-DF, FakeAVCeleb, (DFDC-visual) and DeepfakeTIMIT into a single
Dataset with per-domain balanced sampling and Leave-One-Dataset-Out (LODO)
support for the domain-generalization ablation ladder.

Domain mapping (int):
  0 = LAV-DF
  1 = FakeAVCeleb
  2 = DeepfakeTIMIT
  3 = DFDC-visual (face-only, no audio manipulation)

The dataset assumes features are already extracted with the TALVAD-X pipeline
(scripts/extract_talvad_x.py) into compatible packed/ directories, or plain
final/<id>_sequence.npy files, all with the same feature layout and seq_len.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


DOMAIN_IDS = {
    "LAV-DF": 0,
    "FakeAVCeleb": 1,
    "DeepfakeTIMIT": 2,
    "DFDC": 3,
}


class MultiSourceDataset(Dataset):
    """
    Concatenated multi-domain deepfake detection dataset.

    Each sample returns:
        x           [T, D] float32
        label       scalar float32 (0/1)
        domain      scalar int64  (for DANN)
        frame_label [T] float32   (all-zeros when unavailable)
    """

    def __init__(
        self,
        sources: List[Dict],
        sequence_length: int,
        hold_out_domain: Optional[str] = None,
    ) -> None:
        """
        sources: list of dicts with keys:
          - "name": dataset name matching DOMAIN_IDS
          - "features": np.ndarray [N, T, D] float32 (or float16)
          - "labels": np.ndarray [N] int64
          - "frame_labels": np.ndarray [N, T] float32  (optional; else zeros)
        hold_out_domain: if set, exclude this domain (LODO protocol).
        """
        self.seqs: List[np.ndarray] = []
        self.labels: List[int] = []
        self.domains: List[int] = []
        self.frame_labels: List[np.ndarray] = []

        T = sequence_length
        for src in sources:
            dom = src["name"]
            if hold_out_domain and dom == hold_out_domain:
                continue
            dom_id = DOMAIN_IDS.get(dom, 99)
            feats = np.asarray(src["features"], dtype=np.float32)
            lbls = np.asarray(src["labels"], dtype=np.int64)
            frm = src.get("frame_labels")
            for i in range(len(feats)):
                self.seqs.append(feats[i])
                self.labels.append(int(lbls[i]))
                self.domains.append(dom_id)
                if frm is not None:
                    self.frame_labels.append(np.asarray(frm[i], dtype=np.float32))
                else:
                    self.frame_labels.append(np.zeros(T, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.seqs[idx].copy()).float()
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        d = torch.tensor(self.domains[idx], dtype=torch.int64)
        fl = torch.from_numpy(self.frame_labels[idx]).float()
        return x, y, d, fl

    def make_balanced_sampler(self) -> WeightedRandomSampler:
        """Weight each sample so that (1) real/fake and (2) domains are balanced."""
        labels = np.array(self.labels)
        domains = np.array(self.domains)
        n = len(labels)
        weights = np.ones(n, dtype=np.float64)

        # Per-domain class balance
        for dom_id in np.unique(domains):
            dm = domains == dom_id
            n_real = int((labels[dm] == 0).sum())
            n_fake = int((labels[dm] == 1).sum())
            if n_real > 0 and n_fake > 0:
                w_real = 1.0 / n_real
                w_fake = 1.0 / n_fake
                weights[dm & (labels == 0)] = w_real
                weights[dm & (labels == 1)] = w_fake

        # Additionally balance across domains (each domain gets equal total weight)
        for dom_id in np.unique(domains):
            dm = domains == dom_id
            weights[dm] /= weights[dm].sum()
        weights /= weights.sum()

        return WeightedRandomSampler(
            weights=torch.from_numpy(weights).float(),
            num_samples=n,
            replacement=True,
        )


if __name__ == "__main__":
    T, D, N = 48, 666, 100
    rng = np.random.default_rng(0)
    sources = [
        {"name": "LAV-DF",       "features": rng.random((N, T, D)).astype(np.float32),
         "labels": rng.integers(0, 2, N)},
        {"name": "FakeAVCeleb",  "features": rng.random((N//2, T, D)).astype(np.float32),
         "labels": rng.integers(0, 2, N//2)},
    ]
    # Full dataset
    ds = MultiSourceDataset(sources, sequence_length=T)
    x, y, d, fl = ds[0]
    assert x.shape == (T, D) and d.dtype == torch.int64
    print(f"[OK] full: n={len(ds)} x={x.shape} domain={d.item()}")

    # LODO: hold out FakeAVCeleb
    ds_lodo = MultiSourceDataset(sources, T, hold_out_domain="FakeAVCeleb")
    assert len(ds_lodo) == N, f"LODO should keep {N} samples (LAV-DF only)"
    print(f"[OK] LODO (excl. FakeAVCeleb): n={len(ds_lodo)}")

    sampler = ds.make_balanced_sampler()
    print(f"[OK] balanced sampler: {len(sampler)} samples")
    print("[OK] multisource.py self-test passed")
