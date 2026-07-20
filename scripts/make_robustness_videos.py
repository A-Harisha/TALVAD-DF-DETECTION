"""
scripts/make_robustness_videos.py — build distorted LAV-DF variants for the
AuViRe-style robustness analysis (paper Fig. robustness).

Applies 7 visual + 4 audio distortions at 5 intensity levels to a fixed subset
of unseen LAV-DF test clips, mirroring AuViRe's robustness protocol. Outputs:

    datasets/robustness/<distortion>/L<level>/<stem>.mp4
    datasets/robustness/csv/<distortion>_L<level>.csv   (video_path,label — target-CSV format)
    datasets/robustness/csv/clean.csv                    (original clips, level 0)
    datasets/robustness/gt.json                          {stem: {label, duration, fake_periods}}

Distortions (ffmpeg; block-wise uses OpenCV): saturation, contrast, blockwise,
vnoise (Gaussian video noise), blur, jpeg, vcompress | anoise (white noise mix),
pitch, reverb, acompress. Levels are nominal monotone intensities (documented
below), not calibrated SNRs.

Usage (any python with pandas; ffmpeg must be on PATH):
  python scripts/make_robustness_videos.py --config configs/lavdf_talvad_l.yaml \
      --n_clips 200 --distortions all --levels 1,2,3,4,5
  # smoke: --n_clips 5 --distortions vnoise,anoise --levels 1,5
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.config import load_config  # noqa: E402

VISUAL = ["saturation", "contrast", "blockwise", "vnoise", "blur", "jpeg", "vcompress"]
AUDIO = ["anoise", "pitch", "reverb", "acompress"]
ALL = VISUAL + AUDIO

# level -> parameter (index 0 == level 1)
P = {
    "saturation": [1.4, 1.8, 2.2, 2.6, 3.0],          # eq=saturation
    "contrast":   [1.2, 1.4, 1.6, 1.8, 2.0],          # eq=contrast
    "blockwise":  [4, 8, 12, 16, 20],                 # gray 32px blocks / frame (cv2)
    "vnoise":     [6, 12, 18, 24, 30],                # noise=alls
    "blur":       [0.8, 1.6, 2.4, 3.2, 4.0],          # gblur sigma
    "jpeg":       [6, 12, 18, 24, 31],                # mjpeg -q:v (2 best .. 31 worst)
    "vcompress":  [30, 35, 40, 45, 50],               # x264 CRF
    "anoise":     [0.005, 0.01, 0.02, 0.045, 0.09],   # anoisesrc amplitude
    "pitch":      [0.5, 1.0, 1.5, 2.0, 3.0],          # semitones up
    "reverb":     [(20, 0.25), (40, 0.30), (60, 0.35), (90, 0.40), (120, 0.45)],  # aecho delay,decay
    "acompress":  ["48k", "32k", "24k", "16k", "12k"],  # aac bitrate
}


def _ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError("ffmpeg not on PATH and imageio-ffmpeg unavailable; "
                           "pip install imageio-ffmpeg or add ffmpeg to PATH") from e


FFMPEG = _ffmpeg_bin()


def _run(cmd: list[str]) -> None:
    if cmd and cmd[0] == "ffmpeg":
        cmd = [FFMPEG] + cmd[1:]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({' '.join(cmd[:6])}...):\n{r.stderr[-1500:]}")


def _ffmpeg_variant(src: Path, dst: Path, dist: str, lvl: int) -> None:
    p = P[dist][lvl - 1]
    base = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]
    venc = ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
    aenc = ["-c:a", "aac", "-b:a", "128k"]
    if dist == "saturation":
        cmd = base + ["-vf", f"eq=saturation={p}"] + venc + ["-c:a", "copy", str(dst)]
    elif dist == "contrast":
        cmd = base + ["-vf", f"eq=contrast={p}"] + venc + ["-c:a", "copy", str(dst)]
    elif dist == "vnoise":
        cmd = base + ["-vf", f"noise=alls={p}:allf=t+u"] + venc + ["-c:a", "copy", str(dst)]
    elif dist == "blur":
        cmd = base + ["-vf", f"gblur=sigma={p}"] + venc + ["-c:a", "copy", str(dst)]
    elif dist == "jpeg":
        # per-frame JPEG artifacts: mjpeg intermediate at quality p, then h264 wrap
        tmp = dst.with_suffix(".mjpeg.avi")
        _run(base + ["-c:v", "mjpeg", "-q:v", str(p), "-c:a", "copy", str(tmp)])
        _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp)] + venc + aenc + [str(dst)])
        tmp.unlink(missing_ok=True)
        return
    elif dist == "vcompress":
        cmd = base + ["-c:v", "libx264", "-crf", str(p), "-preset", "veryfast",
                      "-pix_fmt", "yuv420p", "-c:a", "copy", str(dst)]
    elif dist == "anoise":
        cmd = base + ["-filter_complex",
                      f"anoisesrc=colour=white:amplitude={p}[n];"
                      f"[0:a][n]amix=inputs=2:duration=first:normalize=0[a]",
                      "-map", "0:v", "-map", "[a]", "-c:v", "copy"] + aenc + [str(dst)]
    elif dist == "pitch":
        f = 2.0 ** (p / 12.0)          # semitones -> ratio
        sr0 = 16000                    # normalize, then shift by a numeric rate
        new = int(round(sr0 * f))
        cmd = base + ["-af",
                      f"aresample={sr0},asetrate={new},aresample={sr0},atempo={1.0 / f:.6f}",
                      "-c:v", "copy"] + aenc + [str(dst)]
    elif dist == "reverb":
        delay, decay = p
        cmd = base + ["-af", f"aecho=0.8:0.88:{delay}:{decay}", "-c:v", "copy"] + aenc + [str(dst)]
    elif dist == "acompress":
        cmd = base + ["-c:v", "copy", "-c:a", "aac", "-b:a", str(p), str(dst)]
    else:
        raise ValueError(dist)
    _run(cmd)


def _blockwise(src: Path, dst: Path, n_blocks: int, seed: int) -> None:
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = dst.with_suffix(".noaudio.mp4")
    out = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(seed)
    bs = 32
    ok, frame = cap.read()
    while ok:
        for _ in range(n_blocks):
            x = int(rng.integers(0, max(w - bs, 1)))
            y = int(rng.integers(0, max(h - bs, 1)))
            frame[y:y + bs, x:x + bs] = 128
        out.write(frame)
        ok, frame = cap.read()
    cap.release(); out.release()
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-i", str(src),
          "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "18",
          "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "copy", str(dst)])
    tmp.unlink(missing_ok=True)


def _resolve_src(raw: str, root_dir: Path, name: str) -> Path | None:
    """Resolve a clip path robustly: the metadata CSV may carry stale absolute
    paths from another machine (e.g. C:\\Users\\HP\\...). Try as-is, then remap
    the '<name>/<split>/<file>' tail onto the local dataset root, then basename."""
    p = Path(str(raw).replace("\\", "/"))
    if p.exists():
        return p
    parts = p.parts
    if name in parts:
        tail = Path(*parts[parts.index(name) + 1:])   # e.g. test/000001.mp4
        cand = root_dir / tail
        if cand.exists():
            return cand
    for sub in ("test", "dev", "train", ""):
        cand = root_dir / sub / p.name
        if cand.exists():
            return cand
    return None


def _load_gt(annotation_json: Path) -> dict:
    """LAV-DF metadata.min.json -> {stem: {label, fake_periods, duration?}}."""
    meta = json.loads(annotation_json.read_text(encoding="utf-8"))
    out = {}
    for e in meta:
        stem = Path(e.get("file", e.get("path", ""))).stem
        periods = e.get("fake_periods") or []
        out[stem] = {"label": int(bool(periods) or bool(e.get("n_fakes", 0))),
                     "fake_periods": periods,
                     "duration": e.get("duration") or e.get("video_length")}
    return out


def _duration(src: Path) -> float:
    """Clip duration via OpenCV (ffprobe is not bundled with imageio-ffmpeg)."""
    import cv2
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(n / fps) if fps else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_clips", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_root", default="datasets/robustness")
    ap.add_argument("--distortions", default="all")
    ap.add_argument("--levels", default="1,2,3,4,5")
    ap.add_argument("--jobs", type=int, default=12, help="parallel ffmpeg workers (CPU-bound)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    cfg = load_config(args.config)
    ds = cfg["dataset"]
    df = pd.read_csv(ds["metadata_file"])
    df["video_path"] = df["video_path"].astype(str)
    test = df[df["video_path"].str.replace("\\", "/").str.contains("/test/")].reset_index(drop=True)
    rng = random.Random(args.seed)
    idx = list(range(len(test)))
    rng.shuffle(idx)
    rows = test.iloc[idx[:args.n_clips]]
    print(f"selected {len(rows)} unseen test clips (seed {args.seed})")

    root_dir = Path(ds["root_dir"])
    if not root_dir.is_absolute():
        root_dir = REPO / root_dir
    out_root = REPO / args.out_root
    (out_root / "csv").mkdir(parents=True, exist_ok=True)
    gt_all = _load_gt(REPO / ds["annotation_json"])

    dists = ALL if args.distortions == "all" else [d.strip() for d in args.distortions.split(",")]
    levels = [int(x) for x in args.levels.split(",")]
    for d in dists:
        assert d in ALL, f"unknown distortion {d}; choose from {ALL}"

    # clean.csv references originals directly (level 0 baseline; no copy)
    gt_out, clean_rows = {}, []
    missing = 0
    for _, r in rows.iterrows():
        src = _resolve_src(r["video_path"], root_dir, ds["name"])
        if src is None:
            missing += 1
            if missing <= 3:
                print(f"  WARNING: cannot resolve {r['video_path']} under {root_dir}")
            continue
        stem = src.stem
        g = gt_all.get(stem, {})
        label = int(r.get("label", g.get("label", 0)))
        dur = g.get("duration") or _duration(src)
        gt_out[stem] = {"label": label, "duration": float(dur),
                        "fake_periods": g.get("fake_periods", []), "src": str(src)}
        clean_rows.append((str(src), label))
    (out_root / "gt.json").write_text(json.dumps(gt_out, indent=1), encoding="utf-8")
    pd.DataFrame(clean_rows, columns=["video_path", "label"]).to_csv(out_root / "csv" / "clean.csv", index=False)
    print(f"gt.json + clean.csv written ({len(clean_rows)} clips)")

    # Build the full task list (dist, lvl, stem, src, dst, label) and run it
    # through a thread pool — every distortion is a subprocess (ffmpeg) or a
    # GIL-releasing cv2 loop, so threads parallelize cleanly across CPU cores.
    tasks = []
    for dist in dists:
        for lvl in levels:
            (out_root / dist / f"L{lvl}").mkdir(parents=True, exist_ok=True)
            for stem, g in gt_out.items():
                dst = out_root / dist / f"L{lvl}" / f"{stem}.mp4"
                tasks.append((dist, lvl, stem, Path(g["src"]), dst, g["label"]))

    def _gen_one(t):
        dist, lvl, stem, src, dst, label = t
        if dst.exists() and not args.overwrite:
            return (dist, lvl, str(dst), label, "cached")
        try:
            if dist == "blockwise":
                _blockwise(src, dst, P[dist][lvl - 1], seed=hash(stem) & 0xFFFF)
            else:
                _ffmpeg_variant(src, dst, dist, lvl)
            return (dist, lvl, str(dst), label, "made")
        except Exception as ex:
            return (dist, lvl, None, label, f"FAIL {dist} L{lvl} {stem}: {ex}")

    from collections import defaultdict
    rows_by_variant = defaultdict(list)
    made = cached = failed = 0
    total = len(tasks)
    print(f"generating {total} variant clips with {args.jobs} workers ...")
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for i, (dist, lvl, path, label, status) in enumerate(
                ex.map(_gen_one, tasks), 1):
            if path is not None:
                rows_by_variant[(dist, lvl)].append((path, label))
                if status == "made":
                    made += 1
                else:
                    cached += 1
            else:
                failed += 1
                if failed <= 20:
                    print(f"  {status}")
            if i % 250 == 0 or i == total:
                print(f"  ...{i}/{total}  (made={made} cached={cached} fail={failed})",
                      flush=True)

    for (dist, lvl), rows in rows_by_variant.items():
        pd.DataFrame(rows, columns=["video_path", "label"]).to_csv(
            out_root / "csv" / f"{dist}_L{lvl}.csv", index=False)
    print(f"done: {made} made, {cached} cached, {failed} failed; "
          f"{len(rows_by_variant)} variant CSVs written")


if __name__ == "__main__":
    main()
