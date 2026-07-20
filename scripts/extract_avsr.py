"""
scripts/extract_avsr.py — extract frozen auto_avsr VSR encoder features (AV-HuBERT-class,
768-d per frame) for cross-dataset deepfake detection (TALVAD-X v2).

Pipeline per clip (faithful to auto_avsr):
  frames @25fps RGB -> mediapipe LandmarksDetector (4 keypoints) -> VideoProcess affine-
  align to mean face + crop 96x96 grayscale mouth -> /255, CenterCrop(88), Normalize(0.421,
  0.165) -> AVSRVideoEncoder (ResNet frontend + 12-block Conformer) -> [T,768] features.

Per clip we cache pooled stats (mean+std over time -> 1536-d) plus n_frames, id, label,
modify_video, modify_audio. (Pooled is enough for the first supervised cross-dataset test;
sequences can be added for an audio-visual correspondence head later.)

MUST be run with the Py3.11 GPU env:
  .\.venv-lip\Scripts\python.exe scripts/extract_avsr.py --csv data/fakeavceleb_balanced_metadata.csv \
      --root datasets/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2 --target fakeavceleb_bal
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "auto_avsr"))

import torch  # noqa: E402

from preparation.detectors.mediapipe.detector import LandmarksDetector  # noqa: E402
from preparation.detectors.mediapipe.video_process import VideoProcess  # noqa: E402
from models.avsr_encoder import AVSRVideoEncoder  # noqa: E402

FPS = 25


def read_frames_25fps(video_path: str) -> np.ndarray:
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
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    frames = np.stack(frames, 0)
    if abs(native - FPS) > 0.1:
        dur = len(frames) / native
        idx = np.clip((np.arange(0, dur, 1.0 / FPS) * native).astype(int), 0, len(frames) - 1)
        frames = frames[idx]
    return frames


def preprocess_mouth(crops: np.ndarray) -> torch.Tensor:
    """crops [T,96,96] uint8 gray -> [T,1,88,88] normalized (matches val VideoTransform)."""
    x = torch.from_numpy(crops.astype(np.float32)) / 255.0   # [T,96,96]
    x = x[:, 4:92, 4:92]                                      # CenterCrop(88)
    x = (x - 0.421) / 0.165                                   # Normalize
    return x.unsqueeze(1)                                     # [T,1,88,88]


@torch.no_grad()
def process_clip(enc, ld, vp, video_path, device):
    frames = read_frames_25fps(video_path)
    if len(frames) < 8:
        return None
    landmarks = ld(frames)                       # list of [4,2] or None
    crops = vp(frames, landmarks)                # [T,96,96] gray
    if crops is None or len(crops) < 8:
        return None
    x = preprocess_mouth(crops).unsqueeze(0).to(device)   # [1,T,1,88,88]
    feat = enc(x).squeeze(0)                      # [T,768]
    return feat.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = AVSRVideoEncoder(device=device)
    ld = LandmarksDetector()
    vp = VideoProcess()
    root = Path(args.root)
    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]
    print(f"  AVSR extract: {len(rows)} clips  root={root}  device={device}", flush=True)

    pooled, meta = [], {k: [] for k in ("id", "label", "modify_video", "modify_audio", "n_frames")}
    t0 = time.time(); ok = bad = 0
    for i, r in enumerate(rows):
        vp_path = root / r["video_path"]
        try:
            feat = process_clip(enc, ld, vp, str(vp_path), device) if vp_path.exists() else None
        except Exception as e:
            feat = None
            if bad < 5: print(f"   [warn] {r['video_path']}: {e}", flush=True)
        if feat is None:
            bad += 1
        else:
            ok += 1
            pooled.append(np.concatenate([feat.mean(0), feat.std(0)]).astype(np.float32))  # 1536
            meta["id"].append(Path(r["video_path"]).stem)
            meta["label"].append(int(r["label"]))
            meta["modify_video"].append(int(r.get("modify_video", 0)))
            meta["modify_audio"].append(int(r.get("modify_audio", 0)))
            meta["n_frames"].append(int(feat.shape[0]))
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"   {i+1:4d}/{len(rows)}  ok={ok} bad={bad}  {ok/max(time.time()-t0,1e-6):.2f} clips/s", flush=True)

    outdir = REPO / "data" / "talvad_x_crossdataset" / args.target / "avsr"
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / "features.npz", pooled=np.asarray(pooled, np.float32),
             **{k: np.asarray(v) for k, v in meta.items()})
    (outdir / "manifest.json").write_text(json.dumps(
        {"target": args.target, "n_ok": ok, "n_bad": bad, "feat_dim": 768, "pooled_dim": 1536,
         "encoder": "auto_avsr vsr_trlrs3vox2_base", "csv": args.csv}, indent=2), encoding="utf-8")
    print(f"\n  [RESULT] {ok} clips -> {outdir/'features.npz'}  ({bad} skipped)", flush=True)


if __name__ == "__main__":
    main()
