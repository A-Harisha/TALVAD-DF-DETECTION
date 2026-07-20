"""
scripts/extract_talvad_l.py — TALVAD-L feature extraction (localizer stack).

Two-stage because the encoders live in different venvs:
  * AVSR-lip (auto_avsr Conformer) needs mediapipe  -> run under .venv-lip
  * WavLM / MFCC / EfficientNet / spectral          -> run under .venv

Final per-clip layout on the duration-normalized seq_len grid (frame i <-> time
i/seq_len * duration), contiguous so the dataset can slice by manifest offsets:

    [ MFCC(120) | WavLM-large->PCA | AVSR-lip->PCA | EfficientNet->PCA | spectral(S) ]
    |<--------- audio --------->|<-- artic -->|<-- appearance -->|<-- spectral -->|

Stages (resumable — re-running skips finished work):
  avsr : mouth-crop + AVSR encoder -> raw_avsr/<id>.npy [seq_len,768]   (.venv-lip)
  main : MFCC+WavLM+EfficientNet+spectral -> raw_main/<id>.npy          (.venv)
  pack : fit PCA on TRAIN ids, apply, concat, stack -> packed/          (.venv)
  all  : main + pack (avsr must be run separately under .venv-lip first)

Cache root: data/talvad_l/fps{fps}_sl{seq_len}/

Usage:
  # 1) AVSR features (mediapipe env)
  .venv-lip\\Scripts\\python.exe scripts\\extract_talvad_l.py --config configs/lavdf_talvad_l.yaml \
      --stage avsr --n_train 2000 --n_val 500 --n_test 500
  # 2) the rest + pack (main env)
  .venv\\Scripts\\python.exe scripts\\extract_talvad_l.py --config configs/lavdf_talvad_l.yaml \
      --stage main --n_train 2000 --n_val 500 --n_test 500
  .venv\\Scripts\\python.exe scripts\\extract_talvad_l.py --config configs/lavdf_talvad_l.yaml --stage pack
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.config import load_config  # noqa: E402

MFCC_DIM = 120
WAVLM_DIM = 1024
EFF_DIM = 1280


# ── shared helpers (no heavy deps) ────────────────────────────────────────────
def _resample_to(arr: np.ndarray, seq_len: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    t = arr.shape[0]
    if t == seq_len:
        return arr
    if t == 0:
        return np.zeros((seq_len, arr.shape[1] if arr.ndim > 1 else 1), dtype=np.float32)
    idx = np.linspace(0, t - 1, seq_len)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    fr = (idx - lo)[:, None].astype(np.float32)
    return (arr[lo] * (1 - fr) + arr[hi] * fr).astype(np.float32)


def _spec_dim(cfg) -> int:
    from features.spectral import spectral_feature_dim
    s = cfg["features"]["spectral"]
    return spectral_feature_dim(s["n_radial"], s["n_angular"], s["hf_residual"], s["phase"])


def _select_rows(cfg, n_train, n_val, n_test, seed):
    """Deterministic clip list shared across stages: [(id, path, label, split)]."""
    from datasets.metadata import load_metadata
    ds = cfg["dataset"]
    df = load_metadata(ds["metadata_file"], ds["root_dir"], ds["name"])
    df["split"] = df["video_path"].astype(str).apply(
        lambda p: "train" if "/train/" in p.replace("\\", "/") else
                  ("dev" if "/dev/" in p.replace("\\", "/") else
                   ("test" if "/test/" in p.replace("\\", "/") else "train")))
    rng = np.random.default_rng(seed)
    picks = []
    for split, n in [("train", n_train), ("dev", n_val), ("test", n_test)]:
        sub = df[df.split == split]
        if len(sub) and n != 0:
            take = len(sub) if n < 0 else min(n, len(sub))
            picks.append(sub.iloc[rng.permutation(len(sub))[:take]])
    import pandas as pd
    sel = (picks[0] if len(picks) == 1 else pd.concat(picks)).reset_index(drop=True)
    return [(str(r["id"]), str(r["video_path"]), int(r["label"]), r["split"]) for _, r in sel.iterrows()]


def _paths(cfg, target_name=None):
    seq_len = int(cfg["training"]["sequence_length"])
    fps = int(cfg["preprocessing"]["fps"])
    if target_name:
        root = REPO / "data" / "talvad_l_crossdataset" / target_name
    else:
        root = REPO / "data" / "talvad_l" / f"fps{fps}_sl{seq_len}"
    return root, seq_len, fps


def _load_target_rows(csv_path, root_dir):
    """Read an arbitrary cross-dataset CSV (video_path,label) into
    [(uid, path, label, 'test')]. Mirrors extract_talvad_x._load_target_rows,
    including the md5-uid fix for datasets that reuse filenames across folders."""
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
    exists = df["video_path"].map(lambda p: Path(p).exists()).astype(bool)
    miss = int((~exists).sum()) if len(df) else 0
    if miss:
        print(f"  WARNING: {miss} rows reference missing files and are dropped.")
    df = df.loc[exists].reset_index(drop=True)
    df["id"] = df["video_path"].map(
        lambda p: f"{Path(p).stem}_{hashlib.md5(str(p).encode('utf-8')).hexdigest()[:12]}")
    assert df["id"].nunique() == len(df), "clip ids not unique after hashing"
    return [(str(r["id"]), str(r["video_path"]), int(r["label"]), "test") for _, r in df.iterrows()]


# ── stage: avsr (.venv-lip) ───────────────────────────────────────────────────
def stage_avsr(cfg, rows, seq_len, target_name=None):
    sys.path.insert(0, str(REPO / "third_party" / "auto_avsr"))
    import torch
    from preparation.detectors.mediapipe.detector import LandmarksDetector
    from preparation.detectors.mediapipe.video_process import VideoProcess
    from models.avsr_encoder import AVSRVideoEncoder
    import cv2

    root, _, _ = _paths(cfg, target_name)
    out_dir = root / "raw_avsr"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = cfg["features"]["avsr"]["ckpt"]
    enc = AVSRVideoEncoder(ckpt_path=ckpt, device=device)
    ld = LandmarksDetector()
    vp = VideoProcess()
    FPS = 25

    def read_frames_25fps(video_path):
        cap = cv2.VideoCapture(video_path)
        native = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            return np.zeros((0, 0, 0, 3), dtype=np.uint8)
        frames = np.stack(frames, 0)
        if abs(native - FPS) > 0.1:
            dur = len(frames) / native
            idx = np.clip((np.arange(0, dur, 1.0 / FPS) * native).astype(int), 0, len(frames) - 1)
            frames = frames[idx]
        return frames

    def preprocess_mouth(crops):
        x = torch.from_numpy(crops.astype(np.float32)) / 255.0
        x = x[:, 4:92, 4:92]
        x = (x - 0.421) / 0.165
        return x.unsqueeze(1)

    from tqdm import tqdm
    ok = bad = cached = 0
    for sid, vp_path, _, _ in tqdm(rows, desc="  avsr", unit="clip"):
        out = out_dir / f"{sid}.npy"
        if out.exists():
            cached += 1
            continue
        try:
            frames = read_frames_25fps(vp_path)
            if len(frames) < 8:
                bad += 1
                continue
            with torch.no_grad():
                landmarks = ld(frames)
                crops = vp(frames, landmarks)
                if crops is None or len(crops) < 8:
                    bad += 1
                    continue
                x = preprocess_mouth(np.asarray(crops)).unsqueeze(0).to(device)
                feat = enc(x).squeeze(0).cpu().numpy()          # [T_native,768]
            np.save(out, _resample_to(feat, seq_len).astype(np.float16))
            ok += 1
        except Exception:
            bad += 1
            if bad <= 3:
                print(f"   [err] {sid}: {traceback.format_exc().splitlines()[-1]}")
    print(f"  avsr done: ok={ok} cached={cached} bad={bad} -> {out_dir}")


# ── stage: main (.venv) ───────────────────────────────────────────────────────
def stage_main(cfg, rows, seq_len, target_name=None):
    import torch
    from features.appearance import extract_efficientnet_features
    from features.audio_features import extract_mfcc_with_deltas
    from features.audio_ssl import extract_wav2vec2_features
    from features.face_crop import FaceCropper
    from features.spectral import extract_spectral_features
    from preprocessing.audio import extract_audio_wav, preprocess_audio_waveform
    import tempfile

    root, _, _ = _paths(cfg, target_name)
    out_dir = root / "raw_main"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = cfg["features"]["spectral"]
    a_pre = cfg["preprocessing"]["audio"]
    a_feat = cfg["features"]["audio"]
    wavlm_bundle = cfg["features"]["wavlm"]["bundle"]
    wavlm_layer = int(cfg["features"]["wavlm"].get("layer", -1))
    cropper = FaceCropper(size=224, margin=float(cfg["features"]["face"]["margin"]),
                          device=device, min_conf=float(cfg["features"]["face"]["min_conf"]))

    def read_frames_uniform(path, n, max_frames=4000):
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
        idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
        return [frames[i] for i in idx]

    from tqdm import tqdm
    ok = bad = cached = 0
    for sid, vp_path, _, _ in tqdm(rows, desc="  main", unit="clip"):
        out = out_dir / f"{sid}.npy"
        if out.exists():
            cached += 1
            continue
        try:
            frames = read_frames_uniform(vp_path, seq_len)
            if frames is None:
                bad += 1
                continue
            crops = cropper.crop_frames(frames)                            # [seq_len,224,224,3]
            eff = extract_efficientnet_features(crops, device=device, batch_size=64)  # [seq_len,1280]
            spec = extract_spectral_features(list(crops), sp["n_radial"], sp["n_angular"],
                                             sp["hf_residual"], sp["phase"])
            spec = _resample_to(spec, seq_len)
            with tempfile.TemporaryDirectory() as tmp:
                wav_path = Path(tmp) / f"{sid}.wav"
                extract_audio_wav(vp_path, wav_path, a_pre["sample_rate"])
                y, sr = preprocess_audio_waveform(wav_path, sample_rate=a_pre["sample_rate"],
                                                  preemph=a_pre.get("preemphasis", 0.97),
                                                  wavelet_denoise=a_pre.get("wavelet_denoise", False))
            mfcc = _resample_to(extract_mfcc_with_deltas(y, sr, n_mfcc=a_feat["n_mfcc"],
                                n_fft=a_feat["n_fft"], hop_length=a_feat["hop_length"]), seq_len)  # [seq_len,120]
            wavlm = extract_wav2vec2_features(y, sr, target_len=seq_len, model_name=wavlm_bundle,
                                              layer=wavlm_layer, device=device)                     # [seq_len,1024]
            raw = np.concatenate([mfcc, wavlm, eff, spec], axis=-1).astype(np.float16)
            np.save(out, raw)
            ok += 1
        except Exception:
            bad += 1
            if bad <= 3:
                print(f"   [err] {sid}: {traceback.format_exc().splitlines()[-1]}")
    print(f"  main done: ok={ok} cached={cached} bad={bad} -> {out_dir}")


# ── stage: pack (.venv) — STREAMING (bounded RAM, scales to full LAV-DF) ───────
def _fit_pca_stream(raw_dir, sample_ids, slc, dim, name, seed=42):
    """Fit PCA on a SAMPLE of clips, loading each from disk (never all in RAM)."""
    from sklearn.decomposition import PCA
    X = []
    for sid in sample_ids:
        arr = np.load(raw_dir / f"{sid}.npy").astype(np.float32)
        X.append(arr[:, slc[0]:slc[1]])
    X = np.concatenate(X, 0)
    d = min(dim, X.shape[0], X.shape[1])
    pca = PCA(n_components=d, random_state=seed).fit(X)
    print(f"   PCA {name}: {slc[1]-slc[0]}->{d}  (fit on {len(sample_ids)} clips)  "
          f"explvar={pca.explained_variance_ratio_.sum():.3f}", flush=True)
    return pca


def stage_pack(cfg, rows, seq_len, seed):
    import numpy.lib.format as npf
    from joblib import dump
    root, _, fps = _paths(cfg)
    raw_main, raw_avsr, pca_dir, pack_dir = (root / "raw_main", root / "raw_avsr",
                                             root / "pca", root / "packed")
    pca_dir.mkdir(parents=True, exist_ok=True)
    pack_dir.mkdir(parents=True, exist_ok=True)
    S = _spec_dim(cfg)
    wavlm_pca = int(cfg["features"]["wavlm"]["pca_dim"])
    avsr_pca = int(cfg["features"]["avsr"]["pca_dim"])
    eff_pca = int(cfg["features"]["face"]["eff_pca_dim"])
    fit_n = int(cfg["features"].get("pca_fit_n", 2000))

    # kept = clips with BOTH raw files (iterate; never load all into RAM)
    kept = [(sid, path, label, split) for sid, path, label, split in rows
            if (raw_main / f"{sid}.npy").exists() and (raw_avsr / f"{sid}.npy").exists()]
    print(f"  clips with both streams: {len(kept)}/{len(rows)}", flush=True)
    if not kept:
        raise RuntimeError("No clips have both raw_main and raw_avsr — run both stages first.")

    sl_wavlm = (MFCC_DIM, MFCC_DIM + WAVLM_DIM)
    sl_eff = (MFCC_DIM + WAVLM_DIM, MFCC_DIM + WAVLM_DIM + EFF_DIM)
    rng = np.random.default_rng(seed)
    train_ids = [sid for sid, _, _, sp in kept if sp == "train"] or [k[0] for k in kept]
    sample = list(rng.permutation(np.array(train_ids))[:min(fit_n, len(train_ids))])
    pca_w = _fit_pca_stream(raw_main, sample, sl_wavlm, wavlm_pca, "wavlm", seed)
    pca_e = _fit_pca_stream(raw_main, sample, sl_eff, eff_pca, "eff", seed)
    pca_a = _fit_pca_stream(raw_avsr, sample, (0, 768), avsr_pca, "avsr", seed)
    dump(pca_w, pca_dir / "pca_wavlm.joblib")
    dump(pca_e, pca_dir / "pca_eff.joblib")
    dump(pca_a, pca_dir / "pca_avsr.joblib")

    audio_dim, artic_dim, appear_dim = MFCC_DIM + wavlm_pca, avsr_pca, eff_pca
    D = audio_dim + artic_dim + appear_dim + S
    N = len(kept)
    # memory-mapped output — one clip in RAM at a time (peak ~ one clip + PCA models)
    feats = npf.open_memmap(pack_dir / "features.npy", mode="w+", dtype=np.float16, shape=(N, seq_len, D))
    labels, ids, splits = [], [], []
    for i, (sid, path, label, split) in enumerate(kept):
        m = np.load(raw_main / f"{sid}.npy").astype(np.float32)
        a = np.load(raw_avsr / f"{sid}.npy").astype(np.float32)
        mfcc = m[:, :MFCC_DIM]
        wavlm = pca_w.transform(m[:, sl_wavlm[0]:sl_wavlm[1]])
        eff = pca_e.transform(m[:, sl_eff[0]:sl_eff[1]])
        spec = m[:, MFCC_DIM + WAVLM_DIM + EFF_DIM:]
        avsr = pca_a.transform(a)
        feats[i] = np.concatenate([mfcc, wavlm, avsr, eff, spec], axis=-1).astype(np.float16)
        labels.append(label); ids.append(sid); splits.append(split)
        if (i + 1) % 2000 == 0:
            feats.flush(); print(f"  packed {i+1}/{N}", flush=True)
    feats.flush(); del feats                                     # close the memmap
    np.save(pack_dir / "labels.npy", np.asarray(labels, np.int64))
    (pack_dir / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    (pack_dir / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    manifest = {
        "seq_len": seq_len, "fps": fps, "feature_dim": int(feats.shape[-1]),
        "audio_dim": audio_dim, "artic_dim": artic_dim, "appearance_dim": appear_dim, "spectral_dim": S,
        "offsets": {"audio": [0, audio_dim],
                    "artic": [audio_dim, audio_dim + artic_dim],
                    "appearance": [audio_dim + artic_dim, audio_dim + artic_dim + appear_dim],
                    "spectral": [audio_dim + artic_dim + appear_dim, audio_dim + artic_dim + appear_dim + S]},
        "blocks": {"mfcc": MFCC_DIM, "wavlm_pca": wavlm_pca, "avsr_pca": avsr_pca,
                   "eff_pca": eff_pca, "spectral": S},
        "n": len(ids),
    }
    (pack_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  packed: features {feats.shape}  dims=[audio {audio_dim}|artic {artic_dim}|"
          f"appear {appear_dim}|spec {S}] -> {pack_dir}")


# ── stage: pack TARGET (.venv) — zero-shot: LOAD saved LAV-DF PCA bases ────────
def stage_pack_target(cfg, rows, seq_len, target_name, src_pca_dir=None):
    """Pack a cross-dataset target with the SAVED LAV-DF PCA bases (no refit) so the
    feature space matches what the model was trained on. All clips -> split 'test'."""
    import numpy.lib.format as npf
    from joblib import load as jload
    root, _, fps = _paths(cfg, target_name)
    raw_main, raw_avsr, pack_dir = root / "raw_main", root / "raw_avsr", root / "packed"
    pack_dir.mkdir(parents=True, exist_ok=True)
    S = _spec_dim(cfg)
    wavlm_pca = int(cfg["features"]["wavlm"]["pca_dim"])
    avsr_pca = int(cfg["features"]["avsr"]["pca_dim"])
    eff_pca = int(cfg["features"]["face"]["eff_pca_dim"])

    src_pca = Path(src_pca_dir) if src_pca_dir else (
        REPO / "data" / "talvad_l" / f"fps{fps}_sl{seq_len}" / "pca")
    for f in ("pca_wavlm.joblib", "pca_eff.joblib", "pca_avsr.joblib"):
        if not (src_pca / f).exists():
            raise FileNotFoundError(
                f"Saved PCA {f} not found in {src_pca}. Run the LAV-DF pack first so the "
                f"target is projected with the SAME bases the model was trained on.")
    pca_w = jload(src_pca / "pca_wavlm.joblib")
    pca_e = jload(src_pca / "pca_eff.joblib")
    pca_a = jload(src_pca / "pca_avsr.joblib")
    print(f"  loaded saved PCA bases from {src_pca}", flush=True)

    kept = [(sid, path, label, split) for sid, path, label, split in rows
            if (raw_main / f"{sid}.npy").exists() and (raw_avsr / f"{sid}.npy").exists()]
    print(f"  clips with both streams: {len(kept)}/{len(rows)}", flush=True)
    if not kept:
        raise RuntimeError("No clips have both raw_main and raw_avsr — run both stages first.")

    sl_wavlm = (MFCC_DIM, MFCC_DIM + WAVLM_DIM)
    sl_eff = (MFCC_DIM + WAVLM_DIM, MFCC_DIM + WAVLM_DIM + EFF_DIM)
    audio_dim, artic_dim, appear_dim = MFCC_DIM + wavlm_pca, avsr_pca, eff_pca
    D = audio_dim + artic_dim + appear_dim + S
    N = len(kept)
    feats = npf.open_memmap(pack_dir / "features.npy", mode="w+", dtype=np.float16, shape=(N, seq_len, D))
    labels, ids, splits = [], [], []
    for i, (sid, path, label, split) in enumerate(kept):
        m = np.load(raw_main / f"{sid}.npy").astype(np.float32)
        a = np.load(raw_avsr / f"{sid}.npy").astype(np.float32)
        mfcc = m[:, :MFCC_DIM]
        wavlm = pca_w.transform(m[:, sl_wavlm[0]:sl_wavlm[1]])
        eff = pca_e.transform(m[:, sl_eff[0]:sl_eff[1]])
        spec = m[:, MFCC_DIM + WAVLM_DIM + EFF_DIM:]
        avsr = pca_a.transform(a)
        feats[i] = np.concatenate([mfcc, wavlm, avsr, eff, spec], axis=-1).astype(np.float16)
        labels.append(label); ids.append(sid); splits.append(split)
    feats.flush(); del feats
    np.save(pack_dir / "labels.npy", np.asarray(labels, np.int64))
    (pack_dir / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    (pack_dir / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    manifest = {
        "seq_len": seq_len, "fps": fps, "feature_dim": int(D),
        "audio_dim": audio_dim, "artic_dim": artic_dim, "appearance_dim": appear_dim, "spectral_dim": S,
        "offsets": {"audio": [0, audio_dim],
                    "artic": [audio_dim, audio_dim + artic_dim],
                    "appearance": [audio_dim + artic_dim, audio_dim + artic_dim + appear_dim],
                    "spectral": [audio_dim + artic_dim + appear_dim, audio_dim + artic_dim + appear_dim + S]},
        "blocks": {"mfcc": MFCC_DIM, "wavlm_pca": wavlm_pca, "avsr_pca": avsr_pca,
                   "eff_pca": eff_pca, "spectral": S},
        "n": len(ids), "target": target_name,
        "n_real": int((np.asarray(labels) == 0).sum()), "n_fake": int((np.asarray(labels) == 1).sum()),
    }
    (pack_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  packed target '{target_name}': features ({N}, {seq_len}, {D})  "
          f"real={manifest['n_real']} fake={manifest['n_fake']} -> {pack_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["avsr", "main", "pack", "all"])
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    # Cross-dataset target mode (zero-shot): extract an arbitrary CSV and pack with
    # the SAVED LAV-DF PCA bases. Output -> data/talvad_l_crossdataset/<name>/.
    ap.add_argument("--target_csv", default=None, help="cross-dataset CSV (video_path,label)")
    ap.add_argument("--target_name", default=None, help="name for the target pack dir")
    ap.add_argument("--root_dir", default=None, help="prefix for relative target video_path entries")
    ap.add_argument("--pca_dir", default=None, help="dir with saved pca_wavlm/eff/avsr (default: LAV-DF pack pca)")
    ap.add_argument("--limit", type=int, default=0, help="subsample N target clips (0=all)")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    cfg = load_config(args.config)
    tgt = args.target_name
    _, seq_len, fps = _paths(cfg, tgt)

    if args.target_csv:
        assert tgt, "--target_name is required with --target_csv"
        rows = _load_target_rows(args.target_csv, args.root_dir)
        if args.limit and args.limit > 0:
            rng = np.random.default_rng(args.seed)
            rows = [rows[i] for i in rng.permutation(len(rows))[:args.limit]]
        print(f"  TALVAD-L TARGET extract '{tgt}' stage={args.stage}: {len(rows)} clips "
              f"| seq_len={seq_len} fps={fps}")
    else:
        rows = _select_rows(cfg, args.n_train, args.n_val, args.n_test, args.seed)
        print(f"  TALVAD-L extract stage={args.stage}: {len(rows)} clips | seq_len={seq_len} fps={fps}")

    if args.stage == "avsr":
        stage_avsr(cfg, rows, seq_len, tgt)
    elif args.stage == "main":
        stage_main(cfg, rows, seq_len, tgt)
    elif args.stage == "pack":
        stage_pack_target(cfg, rows, seq_len, tgt, args.pca_dir) if tgt else \
            stage_pack(cfg, rows, seq_len, args.seed)
    elif args.stage == "all":
        stage_main(cfg, rows, seq_len, tgt)
        (stage_pack_target(cfg, rows, seq_len, tgt, args.pca_dir) if tgt else
         stage_pack(cfg, rows, seq_len, args.seed))
    print("  DONE")


if __name__ == "__main__":
    main()
