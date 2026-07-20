"""
scripts/train_talvad_l.py — train the TALVAD-L localizer on packed TALVAD-L
features (produced by scripts/extract_talvad_l.py).

Builds LocalizationPackedDataset splits from packed/splits.json, fits the model
dims from the pack manifest (so dim tuning = edit the config + re-pack), and runs
training.trainer_loc.TrainerL (focal + DIoU + CBM + recon + InfoNCE + clip).

Usage:
  .venv\\Scripts\\python.exe scripts\\train_talvad_l.py --config configs/lavdf_talvad_l.yaml
  .venv\\Scripts\\python.exe scripts\\train_talvad_l.py --config configs/lavdf_talvad_l.yaml --epochs 2 --tag smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402

from datasets.sequence_dataset import LocalizationPackedDataset  # noqa: E402
from models.talvad_l import TALVADLLocalizer  # noqa: E402
from training.trainer_loc import TrainerL  # noqa: E402
from utils.config import load_config  # noqa: E402


def _force_utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main():
    _force_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config)
    for kv in args.set:
        k, v = kv.split("=", 1)
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            v = float(v) if ("." in v or "e" in v.lower()) else int(v)
        except ValueError:
            v = v.lower() == "true" if v.lower() in ("true", "false") else v
        node[parts[-1]] = v
    if args.epochs > 0:
        cfg["training"]["epochs"] = args.epochs
    if args.tag:
        cfg["experiment_name"] += f"_{args.tag}"
        cfg["training"]["checkpoint_dir"] += f"_{args.tag}"

    seq_len = int(cfg["training"]["sequence_length"])
    fps = int(cfg["preprocessing"]["fps"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(cfg.get("seed", 42))); np.random.seed(int(cfg.get("seed", 42)))

    pack = REPO / "data" / "talvad_l" / f"fps{fps}_sl{seq_len}" / "packed"
    feats = np.load(pack / "features.npy", mmap_mode="r")        # lazy — never loads 50GB whole
    labels = np.load(pack / "labels.npy")
    ids = json.loads((pack / "ids.json").read_text())
    splits = np.array(json.loads((pack / "splits.json").read_text()))
    manifest = json.loads((pack / "manifest.json").read_text())

    # Standardization fit on a SAMPLE of TRAIN rows (bounded RAM); applied PER-SAMPLE
    # in the dataset. Scales to full LAV-DF without materializing the whole array.
    tr_mask = splits == "train"
    from sklearn.preprocessing import StandardScaler
    tr_idx = np.where(tr_mask)[0]
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    samp = np.sort(rng.choice(tr_idx, size=min(3000, len(tr_idx)), replace=False))
    scaler = StandardScaler().fit(np.asarray(feats[samp]).reshape(-1, feats.shape[-1]))
    Path(cfg["training"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    from joblib import dump
    dump(scaler, Path(cfg["training"]["checkpoint_dir"]) / "scaler.joblib")

    lc = cfg["localization"]
    strides = [2 ** i for i in range(int(cfg["model"]["n_levels"]))]
    ranges = [tuple(r) for r in lc["regression_ranges"]]

    def make_ds(mask):
        idxs = np.where(mask)[0]
        return LocalizationPackedDataset(
            feats, labels[idxs], [ids[i] for i in idxs],   # feats = full mmap; index_map selects
            sequence_length=seq_len, annotation_json=cfg["dataset"]["annotation_json"],
            strides=strides, ranges=ranges, center_radius=float(lc["center_radius"]),
            cbm_sigma=float(cfg["training"].get("cbm_sigma", 1.0)), scaler=scaler,
            index_map=idxs,
        )
    va_mask = splits == "dev"
    if va_mask.sum() == 0:                       # smoke packs may lack dev
        idx = np.where(tr_mask)[0][: max(1, int(tr_mask.sum()) // 5)]
        va_mask = np.zeros_like(tr_mask); va_mask[idx] = True; tr_mask = tr_mask & ~va_mask
    train_ds, val_ds = make_ds(tr_mask), make_ds(va_mask)
    bs = int(cfg["training"]["batch_size"])
    if cfg["training"].get("balanced_sampler", False):
        tl = labels[tr_mask]
        nr, nf = int((tl == 0).sum()), int((tl == 1).sum())
        w = np.where(tl == 1, 1.0 / max(nf, 1), 1.0 / max(nr, 1)).astype(np.float64)
        sampler = WeightedRandomSampler(torch.from_numpy(w).double(), len(w), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["inference"]["batch_size"]), shuffle=False)

    mcfg = dict(cfg["model"])
    mcfg["dims"] = {"audio": manifest["audio_dim"], "artic": manifest["artic_dim"],
                    "appearance": manifest["appearance_dim"], "spectral": manifest["spectral_dim"]}
    model = TALVADLLocalizer(mcfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 64)
    print(f"  TALVAD-L training  exp={cfg['experiment_name']}")
    print(f"  device={device}  params={n_params/1e6:.2f}M  seq_len={seq_len}")
    print(f"  dims: audio={manifest['audio_dim']} artic={manifest['artic_dim']} "
          f"appear={manifest['appearance_dim']} spec={manifest['spectral_dim']}")
    print(f"  splits: train={int(tr_mask.sum())} dev={int(va_mask.sum())}  (test held out)")
    print(f"  cmis warm_init={cfg['model']['cmis'].get('warm_init')}  cmrd={cfg['model']['cmrd'].get('enabled')} "
          f"cbm={cfg['model']['cbm'].get('enabled')}")
    print("=" * 64, flush=True)

    trainer = TrainerL(model, cfg, manifest["offsets"], device)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
