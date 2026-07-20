"""
scripts/extract_talvad_x.py  —  TALVAD-X feature extraction (upgraded pipeline).

Per clip, on a fixed seq_len temporal grid (frame i <-> time i/seq_len * duration):
    [ MFCC+d+dd (120) | wav2vec2 (768->PCA) | EfficientNet-B0 face (1280->PCA) | spectral (S) ]

Stages (all resumable — re-running skips finished work):
  1. raw   : extract raw per-clip features  -> raw/<id>.npy   [seq_len, 120+768+1280+S] fp16
  2. pca   : fit IncrementalPCA on the wav2vec2 and EfficientNet blocks (streamed
             from raw/) -> pca/pca_w2v.joblib, pca/pca_eff.joblib  (saved for parity)
  3. final : apply PCA, concat -> final/<id>.npy  [seq_len, 120+256+256+S] fp16
  4. pack  : stack final/ + labels + ids -> packed/{features.npy,labels.npy,ids.json,manifest.json}

Kept entirely separate from scripts/extract_all.py (the working v8 pipeline).
Cache root: data/talvad_x/fps{fps}_sl{seq_len}/

Usage (dev subset):
  python scripts/extract_talvad_x.py --config configs/lavdf_talvad_x.yaml \
      --n_train 2000 --n_val 500 --n_test 500
Smoke:
  python scripts/extract_talvad_x.py --config configs/lavdf_talvad_x.yaml \
      --n_train 8 --n_val 4 --n_test 4
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from datasets.metadata import load_metadata  # noqa: E402
from features.appearance import EFFICIENTNET_B0_DIM, extract_efficientnet_features  # noqa: E402
from features.audio_features import extract_mfcc_with_deltas  # noqa: E402
from features.audio_ssl import extract_wav2vec2_features, wav2vec2_feature_dim  # noqa: E402
from features.face_crop import FaceCropper  # noqa: E402
from features.spectral import extract_spectral_features, spectral_feature_dim  # noqa: E402
from preprocessing.audio import extract_audio_wav, preprocess_audio_waveform  # noqa: E402
from utils.config import load_config  # noqa: E402

MFCC_DIM = 120
W2V_DIM = wav2vec2_feature_dim("WAV2VEC2_BASE")  # 768


# ── helpers ────────────────────────────────────────────────────────────────
def _resample_to(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """Linearly resample [T, C] -> [seq_len, C] along time."""
    arr = np.asarray(arr, dtype=np.float32)
    t = arr.shape[0]
    if t == seq_len:
        return arr
    if t == 0:
        return np.zeros((seq_len, arr.shape[1]), dtype=np.float32)
    idx = np.linspace(0, t - 1, seq_len)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    fr = (idx - lo)[:, None].astype(np.float32)
    return (arr[lo] * (1 - fr) + arr[hi] * fr).astype(np.float32)


def _read_frames_uniform(path: str, seq_len: int, max_frames: int = 4000):
    """Read up to max_frames, return seq_len uniformly-spaced RGB frames (native res)."""
    import cv2

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        if len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        return None
    idx = np.linspace(0, len(frames) - 1, seq_len).round().astype(int)
    return [frames[i] for i in idx]


def _spec_dim(cfg) -> int:
    s = cfg["features"]["spectral"]
    return spectral_feature_dim(s["n_radial"], s["n_angular"], s["hf_residual"], s["phase"])


def extract_raw_one(sample_id, video_path, raw_dir, cfg, seq_len, cropper, device):
    out = raw_dir / f"{sample_id}_raw.npy"
    if out.exists():
        return sample_id, True, "cached"
    try:
        sp = cfg["features"]["spectral"]
        a_pre = cfg["preprocessing"]["audio"]
        a_feat = cfg["features"]["audio"]

        frames = _read_frames_uniform(video_path, seq_len)
        if frames is None:
            return sample_id, False, "no frames"
        crops = cropper.crop_frames(frames)                       # [seq_len,224,224,3]
        eff = extract_efficientnet_features(crops, device=device, batch_size=64)  # [seq_len,1280]
        spec = extract_spectral_features(
            list(crops), sp["n_radial"], sp["n_angular"], sp["hf_residual"], sp["phase"]
        )                                                         # [seq_len, S]

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / f"{sample_id}.wav"
            extract_audio_wav(video_path, wav_path, a_pre["sample_rate"])
            y, sr = preprocess_audio_waveform(
                wav_path, sample_rate=a_pre["sample_rate"],
                preemph=a_pre.get("preemphasis", 0.97),
                wavelet_denoise=a_pre.get("wavelet_denoise", False),
            )
        mfcc = extract_mfcc_with_deltas(y, sr, n_mfcc=a_feat["n_mfcc"],
                                        n_fft=a_feat["n_fft"], hop_length=a_feat["hop_length"])
        mfcc = _resample_to(mfcc, seq_len)                        # [seq_len,120]
        w2v = extract_wav2vec2_features(y, sr, target_len=seq_len, device=device)  # [seq_len,768]

        raw = np.concatenate([mfcc, w2v, eff, _resample_to(spec, seq_len)], axis=-1).astype(np.float16)
        np.save(out, raw)
        return sample_id, True, "extracted"
    except Exception:
        return sample_id, False, traceback.format_exc()


def fit_pca(raw_dir, pca_dir, ids, pca_dim, fit_n, seed=42):
    """Fit PCA (in-memory; the fit sample fits comfortably in RAM) on the
    wav2vec2 and EfficientNet blocks, saved next to the cache for parity."""
    from joblib import dump
    from sklearn.decomposition import PCA

    pca_dir.mkdir(parents=True, exist_ok=True)
    w2v_path, eff_path = pca_dir / "pca_w2v.joblib", pca_dir / "pca_eff.joblib"
    if w2v_path.exists() and eff_path.exists():
        print("  PCA already fit — skipping.")
        return
    rng = np.random.default_rng(seed)
    avail = [i for i in ids if (raw_dir / f"{i}_raw.npy").exists()]
    if not avail:
        raise RuntimeError("no raw feature files available to fit PCA")
    sample = list(rng.permutation(avail)[: min(fit_n, len(avail))])

    W, E = [], []
    for i in sample:
        raw = np.load(raw_dir / f"{i}_raw.npy").astype(np.float32)
        W.append(raw[:, MFCC_DIM:MFCC_DIM + W2V_DIM])
        E.append(raw[:, MFCC_DIM + W2V_DIM:MFCC_DIM + W2V_DIM + EFFICIENTNET_B0_DIM])
    W, E = np.concatenate(W, 0), np.concatenate(E, 0)
    w_dim = min(pca_dim, *W.shape)
    e_dim = min(pca_dim, *E.shape)
    print(f"  Fitting PCA on {len(sample)} clips ({W.shape[0]} frames): w2v->{w_dim} eff->{e_dim}")
    pca_w = PCA(n_components=w_dim, random_state=seed).fit(W)
    pca_e = PCA(n_components=e_dim, random_state=seed).fit(E)
    dump(pca_w, w2v_path)
    dump(pca_e, eff_path)
    print(f"  PCA explained var: w2v={pca_w.explained_variance_ratio_.sum():.3f} "
          f"eff={pca_e.explained_variance_ratio_.sum():.3f}")


def make_final(raw_dir, final_dir, pca_dir, ids, spec_dim):
    from joblib import load

    final_dir.mkdir(parents=True, exist_ok=True)
    ipca_w = load(pca_dir / "pca_w2v.joblib")
    ipca_e = load(pca_dir / "pca_eff.joblib")
    n = 0
    for i in ids:
        rp = raw_dir / f"{i}_raw.npy"
        fp = final_dir / f"{i}_sequence.npy"
        if fp.exists() or not rp.exists():
            continue
        raw = np.load(rp).astype(np.float32)
        mfcc = raw[:, :MFCC_DIM]
        w2v = ipca_w.transform(raw[:, MFCC_DIM:MFCC_DIM + W2V_DIM])
        eff = ipca_e.transform(raw[:, MFCC_DIM + W2V_DIM:MFCC_DIM + W2V_DIM + EFFICIENTNET_B0_DIM])
        spec = raw[:, MFCC_DIM + W2V_DIM + EFFICIENTNET_B0_DIM:]
        final = np.concatenate([mfcc, w2v, eff, spec], axis=-1).astype(np.float16)
        np.save(fp, final)
        n += 1
    print(f"  Wrote {n} final feature files.")


def _load_target_rows(csv_path, root_dir):
    """Read an arbitrary cross-dataset CSV (video_path,label) into
    [(uid, path, label)]. Mirrors eval_crossdataset._load_target_df, including the
    md5-uid fix for datasets that reuse filenames across identity folders."""
    import hashlib

    import pandas as pd

    df = pd.read_csv(csv_path)
    if "video_path" not in df.columns:
        for alt in ["path", "filepath", "file", "video"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "video_path"}); break
    if "video_path" not in df.columns:
        raise ValueError(f"{csv_path} must contain a 'video_path' column")
    if "label" not in df.columns:
        for alt in ["target", "is_fake", "class"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "label"}); break
    if "label" not in df.columns:
        df["label"] = 0
    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().map({"real": 0, "fake": 1}).fillna(df["label"]).astype(int)
    df["video_path"] = df["video_path"].astype(str)
    if root_dir:
        df["video_path"] = df["video_path"].apply(
            lambda p: str(Path(root_dir) / p) if not Path(p).is_absolute() else p)
    exists = df["video_path"].map(lambda p: Path(p).exists())
    miss = int((~exists).sum())
    if miss:
        print(f"  WARNING: {miss} rows reference missing files and are dropped.")
    df = df.loc[exists].reset_index(drop=True)
    df["id"] = df["video_path"].map(
        lambda p: f"{Path(p).stem}_{hashlib.md5(str(p).encode('utf-8')).hexdigest()[:12]}")
    assert df["id"].nunique() == len(df), "clip ids not unique after hashing"
    return [(str(r["id"]), str(r["video_path"]), int(r["label"])) for _, r in df.iterrows()]


def extract_target(cfg, target_csv, target_name, root_dir, pca_dir, device, limit=0):
    """Zero-shot cross-dataset extraction: same raw/final pipeline as LAV-DF but
    with the SAVED PCA bases (no refit) → feature parity with the trained model.
    Writes data/talvad_x_crossdataset/<target_name>/packed/."""
    seq_len = int(cfg["training"]["sequence_length"])
    fps = int(cfg["preprocessing"]["fps"])
    spec_dim = _spec_dim(cfg)
    pca_dim = int(cfg["features"]["pca_dim"])

    pca_dir = Path(pca_dir) if pca_dir else (REPO / "data" / "talvad_x" / f"fps{fps}_sl{seq_len}" / "pca")
    if not (pca_dir / "pca_w2v.joblib").exists() or not (pca_dir / "pca_eff.joblib").exists():
        raise FileNotFoundError(
            f"Saved PCA not found in {pca_dir}. Run the LAV-DF extraction first so the "
            f"target is projected with the SAME bases the model was trained on.")

    root = REPO / "data" / "talvad_x_crossdataset" / target_name
    raw_dir, final_dir, pack_dir = root / "raw", root / "final", root / "packed"
    for d in (raw_dir, final_dir, pack_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows = _load_target_rows(target_csv, root_dir)
    if limit and limit > 0:
        rng = np.random.default_rng(42)
        rows = [rows[i] for i in rng.permutation(len(rows))[:limit]]
    print(f"  TALVAD-X target extract '{target_name}': {len(rows)} clips | "
          f"seq_len={seq_len} fps={fps} dims=[120|{pca_dim}|{pca_dim}|{spec_dim}] pca_dir={pca_dir}")

    cropper = FaceCropper(size=224, margin=float(cfg["features"]["face"]["margin"]),
                          device=device, min_conf=float(cfg["features"]["face"]["min_conf"]))
    from tqdm import tqdm
    n_err = 0
    for sid, vp, _ in tqdm(rows, desc=f"  raw[{target_name}]", unit="clip"):
        _, ok, msg = extract_raw_one(sid, vp, raw_dir, cfg, seq_len, cropper, device)
        if not ok:
            n_err += 1
            if n_err <= 3:
                print(f"   [err] {sid}: {msg.splitlines()[-1] if msg else '?'}")
    print(f"  raw done (errors={n_err})")

    ids = [r[0] for r in rows]
    make_final(raw_dir, final_dir, pca_dir, ids, spec_dim)

    feats, labels, kept = [], [], []
    label_by_id = {r[0]: r[2] for r in rows}
    for i in ids:
        fp = final_dir / f"{i}_sequence.npy"
        if fp.exists():
            feats.append(np.load(fp)); labels.append(label_by_id[i]); kept.append(i)
    if not feats:
        raise RuntimeError("No target features extracted — check video paths / codecs.")
    feats = np.stack(feats, 0).astype(np.float16)
    np.save(pack_dir / "features.npy", feats)
    np.save(pack_dir / "labels.npy", np.asarray(labels, dtype=np.int64))
    (pack_dir / "ids.json").write_text(json.dumps(kept), encoding="utf-8")
    (pack_dir / "manifest.json").write_text(json.dumps({
        "seq_len": seq_len, "fps": fps, "feature_dim": int(feats.shape[-1]),
        "dims": {"mfcc": MFCC_DIM, "w2v_pca": pca_dim, "eff_pca": pca_dim, "spectral": spec_dim},
        "audio_dim": MFCC_DIM + pca_dim, "appearance_dim": pca_dim, "spectral_dim": spec_dim,
        "n": len(kept), "target": target_name,
    }, indent=2), encoding="utf-8")
    print(f"  packed target: features {feats.shape} -> {pack_dir}")
    print("  DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--fit_pca_n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    # Cross-dataset target mode (zero-shot): extract an arbitrary CSV with the
    # SAVED PCA bases instead of fitting a new split.
    ap.add_argument("--target_csv", default=None, help="cross-dataset CSV (video_path,label)")
    ap.add_argument("--target_name", default=None, help="name for the target pack dir")
    ap.add_argument("--root_dir", default=None, help="prefix for relative target video_path entries")
    ap.add_argument("--pca_dir", default=None, help="dir with saved pca_w2v/pca_eff (default: LAV-DF cache)")
    ap.add_argument("--limit", type=int, default=0, help="subsample N target clips (0=all)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seq_len = int(cfg["training"]["sequence_length"])
    fps = int(cfg["preprocessing"]["fps"])
    pca_dim = int(cfg["features"]["pca_dim"])
    spec_dim = _spec_dim(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Cross-dataset target extraction (zero-shot) ────────────────────────
    if args.target_csv:
        assert args.target_name, "--target_name is required with --target_csv"
        extract_target(cfg, args.target_csv, args.target_name, args.root_dir,
                        args.pca_dir, device, args.limit)
        return

    root = REPO / "data" / "talvad_x" / f"fps{fps}_sl{seq_len}"
    raw_dir, final_dir, pca_dir, pack_dir = (root / "raw", root / "final", root / "pca", root / "packed")
    for d in (raw_dir, final_dir, pca_dir, pack_dir):
        d.mkdir(parents=True, exist_ok=True)

    ds = cfg["dataset"]
    df = load_metadata(ds["metadata_file"], ds["root_dir"], ds["name"])
    df["split"] = df["video_path"].astype(str).apply(
        lambda p: "train" if "/train/" in p.replace("\\", "/") else
                  ("dev" if "/dev/" in p.replace("\\", "/") else
                   ("test" if "/test/" in p.replace("\\", "/") else "train")))
    rng = np.random.default_rng(args.seed)
    picks = []
    for split, n in [("train", args.n_train), ("dev", args.n_val), ("test", args.n_test)]:
        sub = df[df.split == split]
        if len(sub) and n > 0:
            picks.append(sub.iloc[rng.permutation(len(sub))[:min(n, len(sub))]])
    sel = (picks[0] if len(picks) == 1 else __import__("pandas").concat(picks)).reset_index(drop=True)
    rows = [(str(r["id"]), str(r["video_path"]), int(r["label"])) for _, r in sel.iterrows()]
    print(f"  TALVAD-X extract: {len(rows)} clips | seq_len={seq_len} fps={fps} "
          f"dims=[120|{pca_dim}|{pca_dim}|{spec_dim}] device={device}")

    # ── Phase 1: raw (sequential GPU-batched) ──────────────────────────────
    cropper = FaceCropper(size=224, margin=float(cfg["features"]["face"]["margin"]),
                          device=device, min_conf=float(cfg["features"]["face"]["min_conf"]))
    from tqdm import tqdm
    n_err = 0
    for sid, vp, _ in tqdm(rows, desc="  raw", unit="clip"):
        _, ok, msg = extract_raw_one(sid, vp, raw_dir, cfg, seq_len, cropper, device)
        if not ok:
            n_err += 1
            if n_err <= 3:
                print(f"   [err] {sid}: {msg.splitlines()[-1] if msg else '?'}")
    print(f"  raw done (errors={n_err})")

    ids = [r[0] for r in rows]
    # ── Phase 2: PCA fit ───────────────────────────────────────────────────
    fit_pca(raw_dir, pca_dir, ids, pca_dim, args.fit_pca_n, args.seed)
    # ── Phase 3: final ─────────────────────────────────────────────────────
    make_final(raw_dir, final_dir, pca_dir, ids, spec_dim)
    # ── Phase 4: pack ──────────────────────────────────────────────────────
    feats, labels, kept = [], [], []
    label_by_id = {r[0]: r[2] for r in rows}
    for i in ids:
        fp = final_dir / f"{i}_sequence.npy"
        if fp.exists():
            feats.append(np.load(fp)); labels.append(label_by_id[i]); kept.append(i)
    if feats:
        feats = np.stack(feats, 0).astype(np.float16)
        np.save(pack_dir / "features.npy", feats)
        np.save(pack_dir / "labels.npy", np.asarray(labels, dtype=np.int64))
        (pack_dir / "ids.json").write_text(json.dumps(kept), encoding="utf-8")
        (pack_dir / "manifest.json").write_text(json.dumps({
            "seq_len": seq_len, "fps": fps, "feature_dim": int(feats.shape[-1]),
            "dims": {"mfcc": MFCC_DIM, "w2v_pca": pca_dim, "eff_pca": pca_dim, "spectral": spec_dim},
            "audio_dim": MFCC_DIM + pca_dim, "appearance_dim": pca_dim, "spectral_dim": spec_dim,
            "n": len(kept),
        }, indent=2), encoding="utf-8")
        print(f"  packed: features {feats.shape} -> {pack_dir}")
    print("  DONE")


if __name__ == "__main__":
    main()
