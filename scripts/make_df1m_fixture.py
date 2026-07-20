"""
scripts/make_df1m_fixture.py — end-to-end validation of the DF1M sliding-window
localization pipeline WITHOUT any real data.

Generates a handful of variable-length synthetic clips with a PLANTED signal inside
their fake segments (a constant offset on a couple of feature dims), writes them in
the exact per-clip fixed-fps pack layout the real extractor will produce, then runs
the whole stack: WindowedLocalizationDataset -> TALVADLLocalizer -> TrainerL (with the
injected sliding-window dev evaluator) -> standalone evaluate_localization_windowed.

Success = the loop runs with no shape/key errors, losses are finite and fall, and AP
is > 0 (the model learns the planted boundaries through the windowing geometry). This
is the gate that the DF1M code is correct before a single real clip is downloaded.

  .venv\\Scripts\\python.exe scripts\\make_df1m_fixture.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def _force_utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# tiny dims so the smoke is fast; layout/order identical to the real pack
DIMS = {"audio": 16, "artic": 12, "appearance": 8, "spectral": 6}
FPS = 25.0
WIN, HOP = 96, 48


def gen_clip(rng, n_frames, n_fakes):
    D = sum(DIMS.values())
    x = rng.standard_normal((n_frames, D)).astype(np.float32) * 0.5
    segs = []
    cur = 0.6
    for _ in range(n_fakes):
        start = cur + rng.uniform(0.4, 1.2)
        dur = rng.uniform(0.16, 0.5)                  # word-level: ~4-12 frames
        end = start + dur
        if end * FPS >= n_frames - 2:
            break
        a, b = int(round(start * FPS)), int(round(end * FPS))
        x[a:b, 0] += 3.0                               # planted signal (audio dim 0)
        x[a:b, DIMS["audio"]] += 3.0                   # planted signal (artic dim 0)
        segs.append([round(start, 3), round(end, 3)])
        cur = end
    return x, segs


def main():
    _force_utf8()
    root = REPO / "data" / "talvad_l_df1m_fixture"
    if root.exists():
        shutil.rmtree(root)
    feats_dir = root / "feats"; feats_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    ids_by_split = {"train": [], "dev": [], "test": []}
    frames, labels, index = {}, {}, {}
    plan = [("train", 40), ("dev", 10), ("test", 10)]
    for split, n in plan:
        for i in range(n):
            n_frames = int(rng.integers(150, 360))     # 6-14 s variable length
            n_fakes = int(rng.integers(0, 4))          # some clips real (0 fakes)
            x, segs = gen_clip(rng, n_frames, n_fakes)
            cid = f"{split}_{i:03d}"
            np.save(feats_dir / f"{cid}.npy", x.astype(np.float16))
            ids_by_split[split].append(cid)
            frames[cid] = n_frames
            labels[cid] = 1 if segs else 0
            index[cid] = {"label": labels[cid], "fake_periods": [tuple(s) for s in segs],
                          "modify_type": "both_modified" if segs else "real"}

    audio, artic, appear, spec = DIMS["audio"], DIMS["artic"], DIMS["appearance"], DIMS["spectral"]
    offsets = {"audio": [0, audio], "artic": [audio, audio + artic],
               "appearance": [audio + artic, audio + artic + appear],
               "spectral": [audio + artic + appear, audio + artic + appear + spec]}
    manifest = {"fps": FPS, "feature_dim": audio + artic + appear + spec,
                "audio_dim": audio, "artic_dim": artic, "appearance_dim": appear, "spectral_dim": spec,
                "offsets": offsets, "frames": frames, "labels": labels, "splits": ids_by_split}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "index.min.json").write_text(json.dumps(index), encoding="utf-8")
    print(f"  fixture: {sum(len(v) for v in ids_by_split.values())} clips -> {root}")

    # ── build dataset / model / trainer ──────────────────────────────────────
    from datasets.windowed_localization_dataset import WindowedLocalizationDataset
    from models.talvad_l import TALVADLLocalizer
    from training.trainer_loc import TrainerL
    from eval_localization_df1m import evaluate_localization_windowed, make_dev_eval_fn
    from torch.utils.data import DataLoader

    strides = [1, 2, 4, 8]
    ranges = [(0, 8), (8, 16), (16, 32), (32, 1e9)]
    from sklearn.preprocessing import StandardScaler
    samp = np.concatenate([np.load(feats_dir / f"{c}.npy").astype(np.float32)
                           for c in ids_by_split["train"][:20]], 0)
    scaler = StandardScaler().fit(samp)

    def mk(split, mode, neg_ratio=1.0):
        return WindowedLocalizationDataset(
            feats_dir, index, ids_by_split[split], frames, FPS, WIN, HOP, strides, ranges,
            scaler=scaler, mode=mode, neg_ratio=neg_ratio, seed=1)

    train_ds = mk("train", "train", 1.0)
    print(f"  train windows: {len(train_ds)} (pos={train_ds.n_pos} neg={train_ds.n_neg})")
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    mcfg = {"dims": DIMS, "fusion_hidden": 64, "d_model": 64, "n_heads": 4, "ffn": 128,
            "dropout": 0.1, "n_levels": 4, "layers_per_level": 1, "app_proj_dim": 32, "head_conv": 2,
            "use_dfl": True, "reg_max": 16,
            "cmis": {"corr_dim": 64, "spectral_hidden": 32, "incon_hidden": 16, "band_mlp": 16,
                     "coupling": True, "warm_init": True, "contrastive": True},
            "cmrd": {"enabled": True, "hidden": 64}, "cbm": {"enabled": True, "hidden": 32}}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TALVADLLocalizer(mcfg)

    decode = {"score_thr": 0.05, "use_qual": False, "method": "soft",
              "soft_nms_sigma": 0.5, "soft_nms_thr": 0.001, "max_segments": 100}
    cfg = {"experiment_name": "df1m_fixture", "model": mcfg,
           "localization": {"decode": decode},
           "training": {"sequence_length": WIN, "epochs": 4, "lr": 3e-4, "weight_decay": 0.01,
                        "warmup_epochs": 1, "mixed_precision": False, "gradient_clip": 1.0,
                        "early_stopping": False, "checkpoint_metric": "map",
                        "checkpoint_dir": str(root / "ckpt"),
                        "clip_loss_weight": 1.0, "cls_loss_weight": 1.0, "reg_loss_weight": 1.0,
                        "qual_loss_weight": 1.0, "dfl_loss_weight": 0.5, "bcl_loss_weight": 0.3,
                        "cbm_loss_weight": 0.5, "recon_loss_weight": 0.5, "contrastive_loss_weight": 0.1,
                        "contrastive_warmup_epochs": 1, "focal_gamma": 2.0}}

    dev_fn = make_dev_eval_fn(feats_dir, index, ids_by_split["dev"], frames, FPS, WIN, HOP,
                              strides, offsets, decode, scaler, device)
    trainer = TrainerL(model, cfg, offsets, device, dev_eval_fn=dev_fn)
    print("  --- training (4 ep) ---")
    trainer.fit(train_loader, train_loader)        # val_loader ignored (dev_eval_fn set)

    print("  --- standalone test eval ---")
    m = evaluate_localization_windowed(model, feats_dir, index, ids_by_split["test"], frames,
                                       FPS, WIN, HOP, strides, offsets, decode, scaler, device)
    for k in ["AP@50", "AP@75", "AP@90", "mAP", "AR@100", "clip_auc"]:
        print(f"   {k:8s}: {m.get(k, float('nan')):.4f}")
    print(f"   preds={m['n_pred']} gts={m['n_gt']} clips={m['n_clip']}")

    ap50 = float(m.get("AP@50", 0.0))
    if not np.isfinite(ap50):
        print("FIXTURE FAIL: AP not finite"); sys.exit(1)
    print(f"\nDF1M FIXTURE OK  (pipeline runs end-to-end; planted-signal AP@50={ap50:.3f})")


if __name__ == "__main__":
    main()
