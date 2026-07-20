"""
scripts/extract_syncnet.py — extract frozen SyncNet lip-sync features for cross-dataset
deepfake detection (TALVAD-X v2, AV-sync backbone).

For every clip we reproduce SyncNet's exact input pipeline:
  * frames decoded and resampled to 25 fps, face-cropped to 224x224 (reusing
    features.face_crop.FaceCropper), fed as BGR float 0-255 (NO /255 norm);
  * audio via ffmpeg -> 16 kHz mono wav -> python_speech_features MFCC (numcep=13);
  * sliding 5-frame windows; window v uses frames[v:v+5] and mfcc[:, v*4:v*4+20].

Per clip we cache:
  * sync_conf  : joonson confidence = median(shift-dists) - min(shift-dists). HIGH = real.
  * sync_off   : best audio-visual offset (frames).
  * min_dist   : best-aligned L2 distance (LOW = good sync).
  * mean_dist  : mean aligned-window L2 distance.
  * lip_mean / aud_mean : mean-pooled 1024-d embeddings (for training a head later).
Plus id, label, modify_video, modify_audio for per-fake-type analysis.

Output: data/talvad_x_crossdataset/<target>/syncnet/{features.npz, manifest.json}

Run (you run this so you can watch progress):
  python scripts/extract_syncnet.py --csv data/fakeavceleb_balanced_metadata.csv \
      --root datasets/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2 --target fakeavceleb_bal
  # smoke test first:
  python scripts/extract_syncnet.py --csv data/fakeavceleb_balanced_metadata.csv \
      --root datasets/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2 --target smoke --limit 3
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from features.face_crop import FaceCropper  # noqa: E402
from models.syncnet import load_syncnet  # noqa: E402

FPS = 25
SR = 16000
SAMPLES_PER_FRAME = SR // FPS  # 640


def _read_frames_25fps(video_path: str) -> np.ndarray:
    """Decode all frames (BGR), resample to 25 fps. Returns [T,H,W,3] uint8 RGB."""
    import cv2
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
        return np.zeros((0, 224, 224, 3), dtype=np.uint8)
    frames = np.stack(frames, 0)
    if abs(native - FPS) > 0.1:  # resample by nearest timestamp
        dur = len(frames) / native
        idx = np.clip((np.arange(0, dur, 1.0 / FPS) * native).astype(int), 0, len(frames) - 1)
        frames = frames[idx]
    return frames


def _read_audio_16k(video_path: str) -> np.ndarray:
    """ffmpeg -> 16 kHz mono wav -> raw int16 samples as float (joonson convention)."""
    with tempfile.TemporaryDirectory() as td:
        wav = str(Path(td) / "a.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(SR), wav],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if not Path(wav).exists():
            return np.zeros((0,), dtype=np.float64)
        _, a = wavfile.read(wav)
    if a.ndim > 1:
        a = a[:, 0]
    return a.astype(np.float64)  # keep int16 scale (do NOT normalise)


def _calc_conf(lip: torch.Tensor, aud: torch.Tensor, vshift: int = 15):
    """joonson sync confidence. lip/aud: [Nwin,1024]. Returns (conf, offset, min_dist, mean_dist)."""
    n = lip.shape[0]
    win = vshift * 2 + 1
    audp = torch.nn.functional.pad(aud, (0, 0, vshift, vshift))  # pad time dim
    dists = []
    for i in range(n):
        d = torch.nn.functional.pairwise_distance(lip[[i]].repeat(win, 1), audp[i:i + win])
        dists.append(d)
    mdist = torch.mean(torch.stack(dists, 1), 1)        # [win]
    minval, minidx = torch.min(mdist, 0)
    conf = torch.median(mdist) - minval
    aligned = torch.nn.functional.pairwise_distance(lip, aud).mean()  # offset-0 mean dist
    return float(conf), int(vshift - int(minidx)), float(minval), float(aligned)


@torch.no_grad()
def _embed(model, lip_win, aud_win, device, bs=64):
    """lip_win [N,3,5,224,224] float, aud_win [N,1,13,20] float -> embeddings on CPU."""
    le, ae = [], []
    for i in range(0, len(lip_win), bs):
        le.append(model.forward_lip(lip_win[i:i + bs].to(device)).cpu())
        ae.append(model.forward_aud(aud_win[i:i + bs].to(device)).cpu())
    return torch.cat(le), torch.cat(ae)


def process_clip(model, cropper, video_path: str, device: str):
    import python_speech_features as psf
    frames_rgb = _read_frames_25fps(video_path)
    audio = _read_audio_16k(video_path)
    if len(frames_rgb) < 6 or len(audio) < SAMPLES_PER_FRAME * 6:
        return None
    crops_rgb = cropper.crop_frames(frames_rgb)          # [T,224,224,3] RGB
    crops_bgr = crops_rgb[..., ::-1].astype(np.float32)  # SyncNet wants BGR 0-255
    mfcc = psf.mfcc(audio, SR).T                          # [13, Tm], Tm ~ 100fps

    min_len = min(len(crops_bgr), mfcc.shape[1] // 4)
    n_win = min_len - 5
    if n_win < 1 or mfcc.shape[1] < (n_win - 1) * 4 + 20:
        return None

    vid = torch.from_numpy(np.transpose(crops_bgr, (3, 0, 1, 2)))  # [3,T,224,224]
    lip_win = torch.stack([vid[:, v:v + 5] for v in range(n_win)])  # [N,3,5,224,224]
    cc = torch.from_numpy(mfcc.astype(np.float32))
    aud_win = torch.stack([cc[:, v * 4:v * 4 + 20].unsqueeze(0) for v in range(n_win)])  # [N,1,13,20]

    lip_emb, aud_emb = _embed(model, lip_win, aud_win, device)
    conf, off, mind, meand = _calc_conf(lip_emb, aud_emb)
    return {
        "sync_conf": conf, "sync_off": off, "min_dist": mind, "mean_dist": meand,
        "lip_mean": lip_emb.mean(0).numpy(), "aud_mean": aud_emb.mean(0).numpy(),
        "n_win": int(n_win),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--root", required=True, help="dataset root the csv paths are relative to")
    ap.add_argument("--target", required=True, help="output name under data/talvad_x_crossdataset/")
    ap.add_argument("--limit", type=int, default=0, help="process only first N clips (smoke test)")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_syncnet(device=device)
    cropper = FaceCropper(size=224, margin=0.4, device=device, detect_every=1)
    root = Path(args.root)
    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]
    print(f"  SyncNet extract: {len(rows)} clips  root={root}  device={device}", flush=True)

    out = {k: [] for k in ("id", "label", "modify_video", "modify_audio",
                           "sync_conf", "sync_off", "min_dist", "mean_dist", "n_win")}
    lip_means, aud_means = [], []
    t0 = time.time(); ok = 0; bad = 0
    for i, r in enumerate(rows):
        vp = root / r["video_path"]
        try:
            res = process_clip(model, cropper, str(vp), device) if vp.exists() else None
        except Exception as e:
            res = None
            if bad < 5: print(f"   [warn] {r['video_path']}: {e}", flush=True)
        if res is None:
            bad += 1
        else:
            ok += 1
            out["id"].append(Path(r["video_path"]).stem)
            out["label"].append(int(r["label"]))
            out["modify_video"].append(int(r.get("modify_video", 0)))
            out["modify_audio"].append(int(r.get("modify_audio", 0)))
            for k in ("sync_conf", "sync_off", "min_dist", "mean_dist", "n_win"):
                out[k].append(res[k])
            lip_means.append(res["lip_mean"]); aud_means.append(res["aud_mean"])
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            r_ = ok / max(time.time() - t0, 1e-6)
            print(f"   {i+1:4d}/{len(rows)}  ok={ok} bad={bad}  {r_:.2f} clips/s", flush=True)

    outdir = REPO / "data" / "talvad_x_crossdataset" / args.target / "syncnet"
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / "features.npz",
             lip_mean=np.asarray(lip_means, np.float32),
             aud_mean=np.asarray(aud_means, np.float32),
             **{k: np.asarray(v) for k, v in out.items()})
    (outdir / "manifest.json").write_text(json.dumps(
        {"target": args.target, "n_ok": ok, "n_bad": bad, "emb_dim": 1024,
         "csv": args.csv, "root": str(root)}, indent=2), encoding="utf-8")
    print(f"\n  [RESULT] {ok} clips -> {outdir/'features.npz'}  ({bad} skipped)", flush=True)


if __name__ == "__main__":
    main()
