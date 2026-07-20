"""
scripts/train_correspondence.py — train the Phase-2 self-supervised correspondence
PRIOR on LAV-DF REAL clips only (zero-shot-safe: never sees fakes, never sees the
cross-dataset targets). Saves the frozen prior + its scaler to checkpoints/correspondence/.

  python scripts/train_correspondence.py --config configs/lavdf_talvad_x.yaml --epochs 20
  python scripts/train_correspondence.py --config configs/lavdf_talvad_x.yaml --epochs 20 --tag m1 --no_motion
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
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from datasets.metadata import load_metadata  # noqa: E402
from models.correspondence import CrossModalCorrespondence  # noqa: E402
from utils.config import load_config  # noqa: E402


def _split_of(path: str) -> str:
    p = str(path).replace("\\", "/")
    for s in ("train", "dev", "test"):
        if f"/{s}/" in p:
            return s
    return "train"


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--recon_weight", type=float, default=0.5)
    ap.add_argument("--no_motion", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool_real_targets", nargs="*", default=[],
                    help="Refinement A: cross-dataset pack names whose REAL clips are pooled "
                         "in for domain diversity. NEVER pass the eval target (that leaks).")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = int(cfg["training"]["sequence_length"]); fps = int(cfg["preprocessing"]["fps"])

    pack = REPO / "data" / "talvad_x" / f"fps{fps}_sl{seq_len}" / "packed"
    feats = np.load(pack / "features.npy", mmap_mode="r")  # memmap: don't pull 8.3GB into RAM
    labels = np.load(pack / "labels.npy").astype(np.int64)
    ids = json.loads((pack / "ids.json").read_text(encoding="utf-8"))
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    audio_dim = int(manifest["audio_dim"]); app_dim = int(manifest["appearance_dim"])

    ds = cfg["dataset"]
    meta = load_metadata(ds["metadata_file"], ds["root_dir"], ds["name"])
    split_by_id = {str(r["id"]): _split_of(r["video_path"]) for _, r in meta.iterrows()}
    splits = np.array([split_by_id.get(str(i), "train") for i in ids])

    # REAL + TRAIN only — the whole point of the zero-shot-safe prior.
    mask = (splits == "train") & (labels == 0)
    n = int(mask.sum())
    print(f"  correspondence prior: training on {n:,} LAV-DF REAL train clips "
          f"(of {len(labels):,} total)  audio_dim={audio_dim} app_dim={app_dim}")

    from sklearn.preprocessing import StandardScaler
    # Materialize ONLY the REAL+TRAIN subset from the memmap (boolean index copies),
    # then fit+transform just that — avoids touching the full 8.3GB pack twice.
    Xreal = np.ascontiguousarray(feats[mask], dtype=np.float32)   # [n, T, D]
    # Refinement A (domain-adaptive, REAL-only): pool REAL clips from other domains so
    # the prior's "real" model covers more domains and stops treating OOD reals as
    # anomalous (the cause of cross-domain inversion). Same 666-d / saved PCA bases =>
    # directly concatenable. NEVER pool the eval target's reals (would leak).
    for tgt in args.pool_real_targets:
        tp = REPO / "data" / "talvad_x_crossdataset" / tgt / "packed"
        tf = np.load(tp / "features.npy"); tl = np.load(tp / "labels.npy")
        treal = np.ascontiguousarray(tf[tl == 0], dtype=np.float32)
        if treal.shape[-1] != Xreal.shape[-1]:
            raise ValueError(f"{tgt} feature dim {treal.shape[-1]} != LAV-DF {Xreal.shape[-1]}")
        Xreal = np.concatenate([Xreal, treal], axis=0)
        print(f"   pooled +{len(treal):,} REAL clips from {tgt}  (pooled total {len(Xreal):,})")
    D = Xreal.shape[-1]
    scaler = StandardScaler().fit(Xreal.reshape(-1, D))
    Xreal = scaler.transform(Xreal.reshape(-1, D)).reshape(Xreal.shape).astype(np.float32)

    # audio = [:audio_dim]; appearance = [audio_dim : audio_dim+app_dim]  (spectral EXCLUDED)
    X = torch.from_numpy(Xreal)
    loader = DataLoader(TensorDataset(X), batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = CrossModalCorrespondence(audio_dim=audio_dim, visual_dim=app_dim, emb_dim=args.emb_dim,
                                     motion=not args.no_motion, recon_weight=args.recon_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params/1e6:.2f}M  motion={not args.no_motion}  emb={args.emb_dim}  device={device}")

    model.train()
    for ep in range(args.epochs):
        tot, comp = 0.0, {"nce": 0.0, "recon": 0.0}; nb = 0
        for (x,) in loader:
            x = x.to(device)
            a = x[..., :audio_dim]; v = x[..., audio_dim:audio_dim + app_dim]
            loss, c = model.loss(a, v)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
            for k in comp: comp[k] += c[k]
        print(f"   epoch {ep+1:2d}/{args.epochs}  loss={tot/max(nb,1):.4f}  "
              f"nce={comp['nce']/max(nb,1):.4f}  recon={comp['recon']/max(nb,1):.4f}", flush=True)

    out = REPO / "checkpoints" / (f"correspondence_{args.tag}" if args.tag else "correspondence")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "cfg": {"audio_dim": audio_dim, "app_dim": app_dim, "emb_dim": args.emb_dim,
                        "motion": not args.no_motion, "recon_weight": args.recon_weight}},
               out / "prior.pt")
    from joblib import dump
    dump(scaler, out / "scaler.joblib")
    print(f"  [RESULT] correspondence prior saved -> {out / 'prior.pt'}  (trained on {n} real clips)")


if __name__ == "__main__":
    main()
