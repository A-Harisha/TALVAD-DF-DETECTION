"""
scripts/train_talvad_x.py — train the TALVAD-X detector on packed TALVAD-X
features (produced by scripts/extract_talvad_x.py).

Reuses training.trainer.Trainer (BCE + TFLH + CMIS-2 inconsistency BCE + InfoNCE)
and datasets.sequence_dataset.PackedSequenceDataset. Splits the packed clips into
train/dev by their LAV-DF official split directory (test held out for eval).

CMIS-2 ablation override (for the A0..A5 ladder), dotted keys:
  --set model.cmis.enabled=false  --set model.cmis.coupling=false  ...

Usage:
  python scripts/train_talvad_x.py --config configs/lavdf_talvad_x.yaml
  python scripts/train_talvad_x.py --config configs/lavdf_talvad_x.yaml --epochs 2 --tag smoke
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
from torch.utils.data import DataLoader  # noqa: E402

from datasets.metadata import load_metadata  # noqa: E402
from datasets.sequence_dataset import PackedSequenceDataset  # noqa: E402
from evaluation.metrics import compute_binary_metrics  # noqa: E402
from models.talvad_x import TALVADXDetector  # noqa: E402
from training.trainer import Trainer  # noqa: E402
from utils.config import load_config  # noqa: E402


def _split_of(path: str) -> str:
    p = str(path).replace("\\", "/")
    for s in ("train", "dev", "test"):
        if f"/{s}/" in p:
            return s
    return "train"


def _apply_overrides(cfg: dict, sets: list[str]) -> None:
    for kv in sets or []:
        key, val = kv.split("=", 1)
        v: object = val
        if val.lower() in ("true", "false"):
            v = val.lower() == "true"
        else:
            try:
                v = int(val) if val.isdigit() else float(val)
            except ValueError:
                v = val
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v


def _print_setup_banner(cfg, model, feats, labels, splits, ids, audio_dim, video_dim,
                        manifest, pack, train_loader, seed, device):
    """Super-informative pre-training banner: data, splits, dims, model, losses."""
    import torch.nn.functional as F

    def bal(name):
        m = splits == name
        n = int(m.sum())
        fake = int((labels[m] == 1).sum())
        real = n - fake
        pct = (100.0 * fake / n) if n else 0.0
        return n, real, fake, pct

    feat_gb = pack.joinpath("features.npy").stat().st_size / 1e9
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mcfg, tcfg, cm = cfg["model"], cfg["training"], cfg["model"]["cmis"]
    dims = manifest["dims"]
    bs = int(tcfg["batch_size"])
    n_tr = int((splits == "train").sum())
    batches = max(1, -(-n_tr // bs))  # ceil

    dev_label = device
    if "cuda" in device:
        try:
            dev_label = f"cuda — {torch.cuda.get_device_name(0)} " \
                        f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)"
        except Exception:
            pass

    gate_txt = "n/a (CMIS off)"
    if model.use_cmis:
        a = float(F.softplus(model.cmis.alpha_raw).detach())
        t = float(F.softplus(model.cmis.tgate_raw).detach())
        gate_txt = f"coupling α={a:.3f}  temporal τ={t:.3f}  (warm_init={cm.get('warm_init')})"

    bar = "═" * 64
    print(bar)
    print("  TALVAD-X — TRAINING SETUP")
    print(bar)
    print(f"  experiment   : {cfg['experiment_name']}   seed={seed}")
    print(f"  device       : {dev_label}")
    print(f"  feature cache: {pack}")
    print(f"                 features.npy = {feat_gb:.2f} GB  shape={tuple(feats.shape)}")
    print(f"  feature dims : {feats.shape[-1]} = MFCC {dims['mfcc']} | w2v {dims['w2v_pca']} "
          f"| eff {dims['eff_pca']} | spec {dims['spectral']}  (audio={audio_dim} video={video_dim})")
    print("  ── data splits ──────────────────────────────────────────────")
    for nm in ("train", "dev", "test"):
        n, real, fake, pct = bal(nm)
        tag = "(held out for eval)" if nm == "test" else ""
        print(f"   {nm:5s}: {n:7d} clips  | real {real:6d}  fake {fake:6d}  ({pct:4.1f}% fake) {tag}")
    print(f"   sampler: {'balanced (WeightedRandomSampler)' if tcfg.get('balanced_sampler') else 'shuffle (no rebalancing)'}")
    print("  ── model ────────────────────────────────────────────────────")
    print(f"   {model.__class__.__name__}: {n_params/1e6:.2f}M params ({n_train/1e6:.2f}M trainable)")
    print(f"   CMIS-2 enabled={model.use_cmis}  spectral={cm.get('spectral_branch')} "
          f"consistency={cm.get('consistency_branch')} coupling={cm.get('coupling')} "
          f"contrastive={cm.get('contrastive')}")
    print(f"   CMIS-2 gates : {gate_txt}")
    print("  ── losses (weights) ─────────────────────────────────────────")
    print(f"   BCE=1.0  TFLH={tcfg.get('localization_loss_weight',0)}  "
          f"CMIS-incon={tcfg.get('cmis_incon_loss_weight',0)}  "
          f"InfoNCE={tcfg.get('contrastive_loss_weight',0)} (τ={tcfg.get('contrastive_temperature','-')})")
    print("  ── schedule ─────────────────────────────────────────────────")
    print(f"   epochs={tcfg['epochs']}  batch={bs}  ~{batches} batches/epoch  "
          f"lr={tcfg['lr']:.0e}→{tcfg.get('lr_min',0):.0e}  warmup={tcfg.get('warmup_epochs',0)}  "
          f"early-stop patience={tcfg.get('early_stopping_patience','-')}")
    print(f"   regularisers: dropout={mcfg.get('dropout')}  noise={tcfg.get('feature_noise_std')}  "
          f"temporal_mask={tcfg.get('temporal_mask_steps')}  modality_drop={tcfg.get('modality_dropout_p')}  "
          f"amp={tcfg.get('mixed_precision')}")
    print(bar, flush=True)


def _force_utf8_stdout() -> None:
    """Windows consoles default to cp1252, which cannot encode the box-drawing
    characters in the progress output (raises UnicodeEncodeError mid-training).
    Reconfigure stdout/stderr to UTF-8 so the rich logging never crashes a run."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass


def main():
    _force_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=0, help="override training.epochs (0=use config)")
    ap.add_argument("--tag", default="", help="suffix for experiment_name / checkpoint dir")
    ap.add_argument("--set", action="append", default=[], help="dotted override key=value (repeatable)")
    ap.add_argument("--seed", type=int, default=0, help="0 = use config seed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    _apply_overrides(cfg, args.set)
    if args.epochs > 0:
        cfg["training"]["epochs"] = args.epochs
    seed = args.seed or int(cfg.get("seed", 42))
    if args.tag:
        cfg["experiment_name"] = f"{cfg['experiment_name']}_{args.tag}"
        cfg["training"]["checkpoint_dir"] = f"{cfg['training']['checkpoint_dir']}_{args.tag}"
    torch.manual_seed(seed); np.random.seed(seed)

    seq_len = int(cfg["training"]["sequence_length"])
    fps = int(cfg["preprocessing"]["fps"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pack = REPO / "data" / "talvad_x" / f"fps{fps}_sl{seq_len}" / "packed"
    feats = np.load(pack / "features.npy").astype(np.float32)        # [N,T,D]
    labels = np.load(pack / "labels.npy").astype(np.int64)
    ids = json.loads((pack / "ids.json").read_text(encoding="utf-8"))
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    audio_dim = int(manifest["audio_dim"])
    video_dim = int(manifest["appearance_dim"] + manifest["spectral_dim"])
    assert feats.shape[-1] == audio_dim + video_dim, (feats.shape, audio_dim, video_dim)

    # id -> official split (from metadata path)
    ds = cfg["dataset"]
    meta = load_metadata(ds["metadata_file"], ds["root_dir"], ds["name"])
    split_by_id = {str(r["id"]): _split_of(r["video_path"]) for _, r in meta.iterrows()}
    splits = np.array([split_by_id.get(str(i), "train") for i in ids])

    tr = splits == "train"; va = splits == "dev"
    if va.sum() == 0:  # tiny smoke sets may lack a dev split — carve one off train
        idx = np.where(tr)[0]
        va_idx = idx[: max(1, len(idx) // 5)]
        tr[va_idx] = False; va = np.zeros_like(tr); va[va_idx] = True

    # Global standardisation fit on TRAIN only.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(feats[tr].reshape(-1, feats.shape[-1]))
    feats = scaler.transform(feats.reshape(-1, feats.shape[-1])).reshape(feats.shape).astype(np.float32)
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    from joblib import dump
    dump(scaler, ckpt_dir / "scaler.joblib")

    ann = ds.get("annotation_json")
    def make_ds(mask):
        return PackedSequenceDataset(
            feats[mask], labels[mask], [ids[i] for i in np.where(mask)[0]],
            sequence_length=seq_len, annotation_json=ann, fps=fps,
        )
    train_ds, val_ds = make_ds(tr), make_ds(va)
    bs = int(cfg["training"]["batch_size"])
    if cfg["training"].get("balanced_sampler", False):
        # The full LAV-DF pack is ~2.7:1 fake-heavy; inverse-frequency weighting
        # gives the model an even real/fake diet (helps calibration + transfer).
        from torch.utils.data import WeightedRandomSampler
        tr_labels = labels[tr]
        n_real, n_fake = int((tr_labels == 0).sum()), int((tr_labels == 1).sum())
        w = np.where(tr_labels == 1, 1.0 / max(n_fake, 1), 1.0 / max(n_real, 1)).astype(np.float64)
        sampler = WeightedRandomSampler(torch.from_numpy(w).double(), num_samples=len(w), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, drop_last=False)
        print(f"  [balanced_sampler] real={n_real} fake={n_fake} -> inverse-frequency weighting")
    else:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    model = TALVADXDetector(cfg["model"])
    # contrastive weight only matters if the model exposes z_a/z_v (cmis enabled)
    if not model.use_cmis or not cfg["model"]["cmis"].get("contrastive", True):
        cfg["training"]["contrastive_loss_weight"] = 0.0
    if not model.use_cmis:
        cfg["training"]["cmis_incon_loss_weight"] = 0.0

    _print_setup_banner(cfg, model, feats, labels, splits, ids, audio_dim, video_dim,
                        manifest, pack, train_loader, seed, device)

    trainer = Trainer(model, cfg, audio_dim=audio_dim, video_dim=video_dim, device=device)
    state = trainer.fit(train_loader, val_loader)

    # Final dev metrics at the tuned threshold.
    _, y, p = trainer.validate(val_loader)
    m = compute_binary_metrics(y, p, threshold=state.best_threshold)
    print(f"  [RESULT] {cfg['experiment_name']} dev AUC={m.get('auc'):.4f} "
          f"bal_acc={m.get('balanced_accuracy'):.4f} f1={m.get('f1'):.4f} (best_epoch={state.best_epoch+1})")
    return m


if __name__ == "__main__":
    main()
