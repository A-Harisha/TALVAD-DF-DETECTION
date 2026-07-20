"""
scripts/train_talvad_l_df1m.py — train TALVAD-L on AV-Deepfake1M with the fixed-fps
sliding-window pipeline.

Builds WindowedLocalizationDataset splits from a DF1M fixed-fps pack
(data/talvad_l_df1m/fps25, produced by extract_talvad_l_df1m.py) + the annotation
index, fits a per-sample scaler on a sample of TRAIN clips, optionally WARM-STARTS
from the LAV-DF SOTA checkpoint (train.init_from), and runs training.trainer_loc.TrainerL
with an injected SLIDING-WINDOW dev evaluator (so dev mAP is the real stitched metric).

  .venv\\Scripts\\python.exe scripts\\train_talvad_l_df1m.py --config configs/avdeepfake1m_talvad_l.yaml
  .venv\\Scripts\\python.exe scripts\\train_talvad_l_df1m.py --config configs/avdeepfake1m_talvad_l.yaml --epochs 2 --tag smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from datasets.windowed_localization_dataset import WindowedLocalizationDataset, load_pack, load_clip_feats  # noqa: E402
from models.talvad_l import TALVADLLocalizer  # noqa: E402
from training.trainer_loc import TrainerL  # noqa: E402
from utils.config import load_config  # noqa: E402
from eval_localization_df1m import make_dev_eval_fn  # noqa: E402


def _force_utf8():
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass


def main():
    _force_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--pack", default=None, help="override pack dir (default data/talvad_l_df1m/fps<fps>)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs > 0:
        cfg["training"]["epochs"] = args.epochs
    if args.tag:
        cfg["experiment_name"] += f"_{args.tag}"
        cfg["training"]["checkpoint_dir"] += f"_{args.tag}"

    fps = int(cfg["preprocessing"]["fps"])
    win = int(cfg["window"]["win_frames"]); hop = int(cfg["window"]["hop_frames"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(cfg.get("seed", 42))); np.random.seed(int(cfg.get("seed", 42)))

    _base = (cfg.get("output") or {}).get("root") or str(REPO / "data" / "talvad_l_df1m")
    pack = Path(args.pack) if args.pack else (Path(_base) / f"fps{fps}")
    manifest, ids_by_split, frames, _ = load_pack(pack)
    index = json.loads(Path(cfg["dataset"]["index_json"]).read_text())
    feats_dir = pack / "feats"
    offsets = manifest["offsets"]
    strides = [2 ** i for i in range(int(cfg["model"]["n_levels"]))]
    ranges = [tuple(r) for r in cfg["localization"]["regression_ranges"]]

    tr_ids = ids_by_split.get("train", [])
    dev_ids = ids_by_split.get("dev", []) or tr_ids[: max(1, len(tr_ids) // 10)]
    if not tr_ids:
        raise RuntimeError(f"No train clips in pack {pack}.")

    # scaler on a sample of TRAIN clips' frames (bounded RAM)
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    samp_ids = [tr_ids[i] for i in rng.permutation(len(tr_ids))[:min(200, len(tr_ids))]]
    samp = np.concatenate([load_clip_feats(feats_dir, c) for c in samp_ids], 0)
    scaler = StandardScaler().fit(samp)
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    from joblib import dump
    dump(scaler, ckpt_dir / "scaler.joblib")

    def mk(ids, mode):
        return WindowedLocalizationDataset(
            feats_dir, index, ids, frames, fps, win, hop, strides, ranges,
            center_radius=float(cfg["localization"]["center_radius"]),
            cbm_sigma=float(cfg["training"].get("cbm_sigma", 1.0)), scaler=scaler, mode=mode,
            neg_ratio=float(cfg["window"].get("neg_ratio", 1.0)),
            min_overlap_frames=float(cfg["window"].get("min_overlap_frames", 0.5)),
            seed=int(cfg.get("seed", 42)))

    train_ds = mk(tr_ids, "train")
    train_loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)

    mcfg = dict(cfg["model"])
    mcfg["dims"] = {"audio": manifest["audio_dim"], "artic": manifest["artic_dim"],
                    "appearance": manifest["appearance_dim"], "spectral": manifest["spectral_dim"]}
    model = TALVADLLocalizer(mcfg)
    init_from = str(cfg["training"].get("init_from", "")).strip()
    if init_from and Path(init_from).exists():
        state = torch.load(init_from, map_location="cpu")
        sd = state["model"] if "model" in state else state
        # drop keys whose shapes differ (e.g. resized reg_head when reg_max changes,
        # or new refine_head absent in the source) — load_state_dict raises on shape
        # mismatch even with strict=False, so filter first for a clean partial warm-start.
        msd = model.state_dict()
        sd = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"  warm-start from {init_from}  (loaded={len(sd)} missing={len(miss)} unexpected={len(unexp)})")
    elif init_from:
        print(f"  WARNING: init_from not found ({init_from}) — training fresh.")

    dev_fn = make_dev_eval_fn(feats_dir, index, dev_ids, frames, fps, win, hop, strides, offsets,
                              cfg["localization"]["decode"], scaler, device,
                              max_clips=int(cfg["window"].get("eval_max_clips", 0)))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 64)
    print(f"  TALVAD-L DF1M training  exp={cfg['experiment_name']}")
    print(f"  device={device} params={n_params/1e6:.2f}M  win={win} hop={hop} fps={fps}")
    print(f"  clips: train={len(tr_ids)} dev={len(dev_ids)}  | train windows={len(train_ds)} "
          f"(pos={train_ds.n_pos} neg={train_ds.n_neg})")
    print("=" * 64, flush=True)

    trainer = TrainerL(model, cfg, offsets, device, dev_eval_fn=dev_fn)
    trainer.fit(train_loader, train_loader)        # val_loader ignored (dev_eval_fn set)


if __name__ == "__main__":
    main()
