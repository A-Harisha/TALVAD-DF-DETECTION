"""
scripts/extract_talvad_l_df1m.py — FIXED-FPS, variable-length feature extraction
for TALVAD-L on AV-Deepfake1M (DF1M).

Difference vs extract_talvad_l.py (LAV-DF): that one squashes every clip to a
duration-normalized `seq_len` grid. Here each clip keeps its real length at a FIXED
frame rate — n_frames = round(duration * fps) — and features are written PER CLIP
(feats/<clip_id>.npy, [n_frames, D]). This is what lets the sliding-window dataset /
eval (utils.window_ops) preserve sub-second word-level boundaries.

Stages (same two-venv split as the LAV-DF extractor):
  avsr : mouth-crop + auto_avsr lip Conformer -> raw_avsr/<id>.npy [n_frames,768]   (.venv-lip)
  main : MFCC|WavLM|EfficientNet|spectral      -> raw_main/<id>.npy [n_frames,1306-cols] (.venv)
  pack : FIT PCA on a sample of DF1M TRAIN clips (fresh — we train on DF1M, not zero-shot),
         project, write feats/<id>.npy + manifest.json {frames,labels,splits,offsets,dims}

Clip ids come from datasets.avdeepfake1m_loader.df1m_clip_id (the SAME id the
annotation index uses — the guard against the label-misalignment bug). The annotation
index (build it with avdeepfake1m_loader.build_index/write_min_index) supplies labels
and the train/dev/test split membership.

  # 0) build the annotation index once:
  python -c "from datasets.avdeepfake1m_loader import build_index,write_min_index as w; \
             w(build_index(r'<root>/train_metadata.json'), 'data/df1m_index.min.json')"
  # 1) extract (subset via --limit; split column inferred from path or --split_field):
  .venv-lip\\Scripts\\python.exe scripts\\extract_talvad_l_df1m.py --config configs/avdeepfake1m_talvad_l.yaml \\
      --index data/df1m_index.min.json --root <root> --stage avsr --limit 8000
  .venv\\Scripts\\python.exe     scripts\\extract_talvad_l_df1m.py --config configs/avdeepfake1m_talvad_l.yaml \\
      --index data/df1m_index.min.json --root <root> --stage main --limit 8000
  .venv\\Scripts\\python.exe     scripts\\extract_talvad_l_df1m.py --config configs/avdeepfake1m_talvad_l.yaml \\
      --index data/df1m_index.min.json --root <root> --stage pack  --limit 8000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# CUDA allocator: curb the fragmentation-driven OOM ("reserved but unallocated
# memory is large") on long variable-length runs. NOTE: expandable_segments is
# Linux-only (Windows ignores it), so use max_split_size_mb, which IS honored on
# Windows. Must be set before torch initializes CUDA; setdefault lets env win.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.config import load_config                       # noqa: E402
from scripts.extract_talvad_l import _resample_to, _spec_dim, MFCC_DIM, WAVLM_DIM, EFF_DIM  # noqa: E402
from datasets.avdeepfake1m_loader import df1m_clip_id      # noqa: E402

FPS_DEFAULT = 25


def _paths(cfg):
    fps = int(cfg["preprocessing"]["fps"])
    base = (cfg.get("output") or {}).get("root") or str(REPO / "data" / "talvad_l_df1m")
    root = Path(base) / f"fps{fps}"
    return root, fps


def _split_of(rel: str, entry: dict) -> str:
    if entry.get("split") in ("train", "dev", "val", "test"):
        return "dev" if entry["split"] == "val" else entry["split"]
    p = rel.replace("\\", "/").lower()
    for s in ("train", "dev", "val", "test"):
        if f"/{s}/" in p or p.startswith(f"{s}/"):
            return "dev" if s == "val" else s
    return "train"


def _select_rows(cfg, index_path, root, limit, seed):
    index = json.loads(Path(index_path).read_text())
    rows = []
    for cid, e in index.items():
        rel = e["file"]
        path = str(Path(root) / rel) if not Path(rel).is_absolute() else rel
        rows.append((cid, path, int(e.get("label", 0)), _split_of(rel, e)))
    rows.sort(key=lambda r: r[0])
    if limit and limit > 0:
        rng = np.random.default_rng(seed)
        # keep split proportions while subsampling
        idx = rng.permutation(len(rows))[:limit]
        rows = [rows[i] for i in sorted(idx)]
    return rows, index


def _n_frames(path, fps):
    import cv2
    cap = cv2.VideoCapture(path)
    cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    native = cap.get(cv2.CAP_PROP_FPS) or fps
    cap.release()
    dur = (cnt / native) if native > 0 else 0.0
    return max(1, int(round(dur * fps)))


# ── stage: avsr (.venv-lip) ───────────────────────────────────────────────────
def stage_avsr(cfg, rows, fps):
    sys.path.insert(0, str(REPO / "third_party" / "auto_avsr"))
    import torch, cv2
    from preparation.detectors.mediapipe.detector import LandmarksDetector
    from preparation.detectors.mediapipe.video_process import VideoProcess
    from models.avsr_encoder import AVSRVideoEncoder

    root, _ = _paths(cfg)
    out_dir = root / "raw_avsr"; out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = AVSRVideoEncoder(ckpt_path=cfg["features"]["avsr"]["ckpt"], device=device)
    ld, vp = LandmarksDetector(), VideoProcess()

    def read_25fps(path):
        cap = cv2.VideoCapture(path); native = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ok, f = cap.read()
            if not ok: break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames: return np.zeros((0, 0, 0, 3), np.uint8)
        frames = np.stack(frames, 0)
        if abs(native - 25.0) > 0.1:
            dur = len(frames) / native
            idx = np.clip((np.arange(0, dur, 1.0 / 25.0) * native).astype(int), 0, len(frames) - 1)
            frames = frames[idx]
        return frames

    def prep(crops):
        x = torch.from_numpy(crops.astype(np.float32)) / 255.0
        x = (x[:, 4:92, 4:92] - 0.421) / 0.165
        return x.unsqueeze(1)

    from tqdm import tqdm
    ok = bad = cached = 0
    for cid, path, _, _ in tqdm(rows, desc="  df1m-avsr", unit="clip"):
        out = out_dir / f"{cid}.npy"
        if out.exists(): cached += 1; continue
        try:
            frames = read_25fps(path)
            n = max(1, len(frames))
            lms = ld(frames)
            crops = vp(frames, lms)
            feat = enc(prep(np.asarray(crops)).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()  # [t_avsr,768]
            np.save(out, _resample_to(feat, n).astype(np.float16))             # -> fixed-fps n
            ok += 1
        except Exception:
            bad += 1
            if bad <= 3: print(f"   [err] {cid}: {traceback.format_exc().splitlines()[-1]}")
    print(f"  df1m-avsr done: ok={ok} cached={cached} bad={bad} -> {out_dir}")


# ── stage: main (.venv) ───────────────────────────────────────────────────────
def stage_main(cfg, rows, fps):
    import torch, cv2, tempfile
    from features.appearance import extract_efficientnet_features
    from features.audio_features import extract_mfcc_with_deltas
    from features.audio_ssl import extract_wav2vec2_features
    from features.face_crop import FaceCropper
    from features.spectral import extract_spectral_features
    from preprocessing.audio import extract_audio_wav, preprocess_audio_waveform

    root, _ = _paths(cfg)
    out_dir = root / "raw_main"; out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = cfg["features"]["spectral"]; a_pre = cfg["preprocessing"]["audio"]; a_feat = cfg["features"]["audio"]
    wavlm_bundle = cfg["features"]["wavlm"]["bundle"]; wavlm_layer = int(cfg["features"]["wavlm"].get("layer", -1))
    # detect_every>1: run MTCNN on every Nth frame and fill boxes between (clips are
    # already face-centered). 8x faster + 5x less GPU on long clips; ~sub-3% crop shift.
    cropper = FaceCropper(size=224, margin=float(cfg["features"]["face"]["margin"]),
                          device=device, min_conf=float(cfg["features"]["face"]["min_conf"]),
                          detect_every=int(cfg["features"]["face"].get("detect_every", 5)))

    def read_all_25fps(path, fps, max_frames=8000):
        cap = cv2.VideoCapture(path); native = cap.get(cv2.CAP_PROP_FPS) or fps
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok: break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if len(frames) >= max_frames: break
        cap.release()
        if not frames: return None
        if abs(native - fps) > 0.1:
            dur = len(frames) / native
            idx = np.clip((np.arange(0, dur, 1.0 / fps) * native).astype(int), 0, len(frames) - 1)
            frames = [frames[i] for i in idx]
        return frames

    from tqdm import tqdm
    ok = bad = cached = 0
    for cid, path, _, _ in tqdm(rows, desc="  df1m-main", unit="clip"):
        out = out_dir / f"{cid}.npy"
        if out.exists(): cached += 1; continue
        try:
            frames = read_all_25fps(path, fps)
            if frames is None: bad += 1; continue
            n = len(frames)                                                    # fixed-fps target length
            crops = cropper.crop_frames(frames)
            eff = _resample_to(extract_efficientnet_features(crops, device=device, batch_size=32), n)
            spec = _resample_to(extract_spectral_features(list(crops), sp["n_radial"], sp["n_angular"],
                                                          sp["hf_residual"], sp["phase"]), n)
            with tempfile.TemporaryDirectory() as tmp:
                wav = Path(tmp) / f"{cid}.wav"
                try:
                    extract_audio_wav(path, wav, a_pre["sample_rate"])
                    y, sr = preprocess_audio_waveform(wav, sample_rate=a_pre["sample_rate"],
                                                      preemph=a_pre.get("preemphasis", 0.97),
                                                      wavelet_denoise=a_pre.get("wavelet_denoise", False))
                except Exception:
                    # silent_videos have no audio stream -> ffmpeg fails; the video is fine.
                    # A silent video's audio IS silence, so use a zero waveform sized to the clip.
                    sr = int(a_pre["sample_rate"])
                    y = np.zeros(max(1600, int(sr * n / fps)), dtype=np.float32)
            mfcc = _resample_to(extract_mfcc_with_deltas(y, sr, n_mfcc=a_feat["n_mfcc"],
                                n_fft=a_feat["n_fft"], hop_length=a_feat["hop_length"]), n)
            wavlm = _resample_to(extract_wav2vec2_features(y, sr, target_len=n, model_name=wavlm_bundle,
                                                           layer=wavlm_layer, device=device), n)
            raw = np.concatenate([mfcc, wavlm, eff, spec], axis=-1).astype(np.float16)
            np.save(out, raw); ok += 1
        except Exception:
            bad += 1
            if bad <= 3: print(f"   [err] {cid}: {traceback.format_exc().splitlines()[-1]}")
        if device == "cuda" and (ok + bad) % 200 == 0:
            torch.cuda.empty_cache()                 # periodic flush bounds allocator fragmentation (Windows)
    print(f"  df1m-main done: ok={ok} cached={cached} bad={bad} -> {out_dir}")


# ── stage: pack (.venv) — FIT PCA on DF1M train sample, write per-clip feats ───
def stage_pack(cfg, rows, fps, seed, pca_dir=None):
    from sklearn.decomposition import PCA
    from joblib import dump, load as jload
    root, _ = _paths(cfg)
    raw_main, raw_avsr = root / "raw_main", root / "raw_avsr"
    feats_dir = root / "feats"; feats_dir.mkdir(parents=True, exist_ok=True)
    pca_out = root / "pca"; pca_out.mkdir(parents=True, exist_ok=True)
    S = _spec_dim(cfg)
    wavlm_pca = int(cfg["features"]["wavlm"]["pca_dim"]); avsr_pca = int(cfg["features"]["avsr"]["pca_dim"])
    eff_pca = int(cfg["features"]["face"]["eff_pca_dim"]); fit_n = int(cfg["features"].get("pca_fit_n", 2000))

    kept = [(cid, lbl, split) for cid, _, lbl, split in rows
            if (raw_main / f"{cid}.npy").exists() and (raw_avsr / f"{cid}.npy").exists()]
    print(f"  clips with both streams: {len(kept)}/{len(rows)}", flush=True)
    if not kept:
        raise RuntimeError("No clips have both raw streams — run avsr+main first.")
    sl_w = (MFCC_DIM, MFCC_DIM + WAVLM_DIM); sl_e = (MFCC_DIM + WAVLM_DIM, MFCC_DIM + WAVLM_DIM + EFF_DIM)

    if pca_dir:                                                # reuse saved bases (e.g. LAV-DF parity)
        pca_w = jload(Path(pca_dir) / "pca_wavlm.joblib"); pca_e = jload(Path(pca_dir) / "pca_eff.joblib")
        pca_a = jload(Path(pca_dir) / "pca_avsr.joblib"); print(f"  loaded PCA bases from {pca_dir}")
    else:                                                      # FIT fresh on DF1M train
        rng = np.random.default_rng(seed)
        train_ids = [c for c, _, sp in kept if sp == "train"] or [k[0] for k in kept]
        sample = list(rng.permutation(np.array(train_ids))[:min(fit_n, len(train_ids))])
        def fit(slc, dim, src, name):
            X = np.concatenate([np.load(src / f"{c}.npy").astype(np.float32)[:, slc[0]:slc[1]] for c in sample], 0)
            d = min(dim, X.shape[0], X.shape[1]); pca = PCA(n_components=d, random_state=seed).fit(X)
            print(f"   PCA {name}: {slc[1]-slc[0]}->{d} explvar={pca.explained_variance_ratio_.sum():.3f}")
            return pca
        pca_w = fit(sl_w, wavlm_pca, raw_main, "wavlm"); pca_e = fit(sl_e, eff_pca, raw_main, "eff")
        pca_a = fit((0, 768), avsr_pca, raw_avsr, "avsr")
        dump(pca_w, pca_out / "pca_wavlm.joblib"); dump(pca_e, pca_out / "pca_eff.joblib"); dump(pca_a, pca_out / "pca_avsr.joblib")

    audio_dim, artic_dim, appear_dim = MFCC_DIM + wavlm_pca, avsr_pca, eff_pca
    D = audio_dim + artic_dim + appear_dim + S
    frames, labels, splits = {}, {}, {"train": [], "dev": [], "test": []}
    packed = bad_pack = 0
    for cid, lbl, split in kept:
        try:
            m = np.load(raw_main / f"{cid}.npy").astype(np.float32); a = np.load(raw_avsr / f"{cid}.npy").astype(np.float32)
            if a.shape[0] != m.shape[0]:                 # avsr uncapped vs main's 8000-frame cap -> align to main
                a = _resample_to(a, m.shape[0])
            feat = np.concatenate([m[:, :MFCC_DIM], pca_w.transform(m[:, sl_w[0]:sl_w[1]]),
                                   pca_a.transform(a), pca_e.transform(m[:, sl_e[0]:sl_e[1]]),
                                   m[:, MFCC_DIM + WAVLM_DIM + EFF_DIM:]], axis=-1).astype(np.float16)
            np.save(feats_dir / f"{cid}.npy", feat)
            frames[cid] = int(feat.shape[0]); labels[cid] = int(lbl)
            splits.setdefault(split, []).append(cid)
            packed += 1
        except Exception:
            bad_pack += 1
            if bad_pack <= 3: print(f"   [pack err] {cid}: {traceback.format_exc().splitlines()[-1]}")
    print(f"  packed={packed} bad={bad_pack}")
    off = {"audio": [0, audio_dim], "artic": [audio_dim, audio_dim + artic_dim],
           "appearance": [audio_dim + artic_dim, audio_dim + artic_dim + appear_dim],
           "spectral": [audio_dim + artic_dim + appear_dim, audio_dim + artic_dim + appear_dim + S]}
    manifest = {"fps": fps, "feature_dim": D, "audio_dim": audio_dim, "artic_dim": artic_dim,
                "appearance_dim": appear_dim, "spectral_dim": S, "offsets": off,
                "frames": frames, "labels": labels, "splits": splits, "n": len(frames)}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    print(f"  df1m pack: {len(frames)} clips D={D} -> {feats_dir}  "
          f"(train={len(splits.get('train',[]))} dev={len(splits.get('dev',[]))} test={len(splits.get('test',[]))})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", required=True, help="DF1M annotation min-index json")
    ap.add_argument("--root", required=True, help="dataset root for relative file paths")
    ap.add_argument("--stage", required=True, choices=["avsr", "main", "pack"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pca_dir", default=None, help="reuse saved PCA bases instead of fitting (parity)")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

    cfg = load_config(args.config)
    _, fps = _paths(cfg)
    rows, _ = _select_rows(cfg, args.index, args.root, args.limit, args.seed)
    print(f"  DF1M extract stage={args.stage}: {len(rows)} clips | fps={fps}")
    if args.stage == "avsr": stage_avsr(cfg, rows, fps)
    elif args.stage == "main": stage_main(cfg, rows, fps)
    elif args.stage == "pack": stage_pack(cfg, rows, fps, args.seed, args.pca_dir)
    print("  DONE")


if __name__ == "__main__":
    main()
