"""
scripts/fetch_vidtimit_reals.py

Download the real (VidTIMIT) videos that correspond to the DeepfakeTIMIT fakes,
and mux them into playable .mp4 clips so the TALVAD-X pipeline can read them
(it expects a real video container: cv2.VideoCapture + a muxed audio stream).

Why this is needed
------------------
DeepfakeTIMIT ships only the *fake* face-swapped .avi clips. Their real
counterparts come from VidTIMIT, where each recording is stored as a numbered
sequence of JPEG frames (512x384) plus a separate 32 kHz mono WAV — i.e. NOT a
video file. This script downloads the per-subject VidTIMIT zips from Zenodo
(record 158963), then muxes each sentence's frames + WAV into an .mp4 at 25 fps
(matching the DeepfakeTIMIT fakes) under <root>/real/<subject>/<sentence>.mp4.

Only the 32 subjects present in DeepfakeTIMIT/higher_quality are fetched, and
only sentences that have BOTH a frame folder and a matching audio WAV are muxed
(the `head`/`head2`/`head3` head-rotation clips have no audio and are skipped).

Resumable & disk-frugal
-----------------------
Per subject: download zip -> unzip -> mux all sentences -> (default) delete the
unzipped frames and the zip. Subjects whose real/<subject>/ already holds the
expected number of mp4s are skipped. Peak extra disk stays ~one subject (~160 MB)
plus the small muxed outputs.

Usage:
  .venv/Scripts/python.exe scripts/fetch_vidtimit_reals.py \
      --root "datasets/Deepfake TIMIT/DeepfakeTIMIT"
  # keep the downloaded zips / unzipped frames for inspection:
  ... --keep_zips --keep_unzipped
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ZENODO_BASE = "https://zenodo.org/records/158963/files"
FPS = 25  # VidTIMIT native rate; also the DeepfakeTIMIT fake rate (25/1, verified)


def log(msg: str) -> None:
    print(msg, flush=True)


def download_zip(subject: str, zips_dir: Path) -> Path:
    """Download <subject>.zip from Zenodo (skip if already present & valid)."""
    out = zips_dir / f"{subject}.zip"
    if out.exists() and zipfile.is_zipfile(out):
        log(f"  [{subject}] zip cached ({out.stat().st_size/1e6:.0f} MB)")
        return out
    url = f"{ZENODO_BASE}/{subject}.zip?download=1"
    log(f"  [{subject}] downloading {url}")
    # curl: -L follow redirects, -f fail on HTTP error, -s silent, -S show errors
    rc = subprocess.run(
        ["curl", "-L", "-f", "-s", "-S", "-o", str(out), url],
        capture_output=True, text=True,
    )
    if rc.returncode != 0 or not (out.exists() and zipfile.is_zipfile(out)):
        if out.exists():
            out.unlink()
        raise RuntimeError(f"download failed for {subject}: {rc.stderr.strip()}")
    log(f"  [{subject}] downloaded ({out.stat().st_size/1e6:.0f} MB)")
    return out


def unzip(subject: str, zip_path: Path, work_dir: Path) -> Path:
    """Unzip into work_dir/<subject>/ (skip if already unzipped)."""
    dest = work_dir / subject
    if (dest / "audio").exists() and (dest / "video").exists():
        return dest
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(work_dir)
    if not ((dest / "audio").exists() and (dest / "video").exists()):
        raise RuntimeError(f"unexpected zip layout for {subject} (no audio/video dirs)")
    return dest


def mux_sentence(frames_dir: Path, wav_path: Path, out_mp4: Path) -> None:
    """Mux a numbered JPEG sequence (001,002,...) + WAV into an h264/aac mp4."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(FPS), "-f", "image2", "-start_number", "1",
        "-i", str(frames_dir / "%03d"),
        "-i", str(wav_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(out_mp4),
    ]
    rc = subprocess.run(cmd, capture_output=True, text=True)
    if rc.returncode != 0 or not out_mp4.exists():
        raise RuntimeError(f"ffmpeg mux failed for {out_mp4.name}: {rc.stderr.strip()}")


def process_subject(subject: str, root: Path, zips_dir: Path, work_dir: Path,
                    keep_zips: bool, keep_unzipped: bool) -> tuple[int, int]:
    """Returns (muxed, expected) sentence counts for the subject."""
    real_dir = root / "real" / subject
    zip_path = download_zip(subject, zips_dir)
    src = unzip(subject, zip_path, work_dir)

    audio_dir, video_dir = src / "audio", src / "video"
    sentences = sorted(w.stem for w in audio_dir.glob("*.wav")
                       if (video_dir / w.stem).is_dir())
    muxed = 0
    for s in sentences:
        out_mp4 = real_dir / f"{s}.mp4"
        if out_mp4.exists():
            muxed += 1
            continue
        frames = sorted((video_dir / s).iterdir())
        if not frames:
            log(f"  [{subject}] WARN: no frames for {s}, skipping")
            continue
        mux_sentence(video_dir / s, audio_dir / f"{s}.wav", out_mp4)
        muxed += 1
    log(f"  [{subject}] muxed {muxed}/{len(sentences)} sentences -> {real_dir}")

    if not keep_unzipped:
        shutil.rmtree(src, ignore_errors=True)
    if not keep_zips:
        zip_path.unlink(missing_ok=True)
    return muxed, len(sentences)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/Deepfake TIMIT/DeepfakeTIMIT",
                    help="DeepfakeTIMIT root (must contain higher_quality/)")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="explicit subject ids; default = all under higher_quality/")
    ap.add_argument("--stage_dir", default="data/vidtimit",
                    help="staging dir for zips + temporary unzipped frames")
    ap.add_argument("--keep_zips", action="store_true")
    ap.add_argument("--keep_unzipped", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    hq = root / "higher_quality"
    if not hq.is_dir():
        sys.exit(f"ERROR: {hq} not found — is --root correct?")
    subjects = args.subjects or sorted(d.name for d in hq.iterdir() if d.is_dir())

    stage = Path(args.stage_dir)
    zips_dir, work_dir = stage / "zips", stage / "unzipped"
    zips_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    log(f"VidTIMIT reals: {len(subjects)} subjects -> {root/'real'}")
    total_muxed, total_expected, failed = 0, 0, []
    for i, subj in enumerate(subjects, 1):
        real_dir = root / "real" / subj
        # quick resume: if already has >=10 mp4s, skip whole subject
        if real_dir.is_dir() and len(list(real_dir.glob("*.mp4"))) >= 10:
            n = len(list(real_dir.glob("*.mp4")))
            log(f"[{i}/{len(subjects)}] {subj}: {n} mp4 cached — skip")
            total_muxed += n
            total_expected += n
            continue
        log(f"[{i}/{len(subjects)}] {subj}")
        try:
            m, e = process_subject(subj, root, zips_dir, work_dir,
                                   args.keep_zips, args.keep_unzipped)
            total_muxed += m
            total_expected += e
        except Exception as exc:  # noqa: BLE001
            log(f"  [{subj}] FAILED: {exc}")
            failed.append(subj)

    log(f"\nDONE: muxed {total_muxed}/{total_expected} real clips across "
        f"{len(subjects)-len(failed)}/{len(subjects)} subjects.")
    if failed:
        log(f"FAILED subjects ({len(failed)}): {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
