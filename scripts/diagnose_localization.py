"""
scripts/diagnose_localization.py — Step 3 / Part B diagnostic.

Tests, on the LAV-DF test split, WHY segment-AP is ~0 despite high frame-AUC.
Checks the hypotheses from the localization plan in one pass:

  B1/B2  window-vs-duration + time-units mismatch:
         frame LABELS map frame t -> t*duration/seq_len (full-video stretch),
         but the predicted-segment decoder maps frame t -> t/fps (fixed [0, seq_len/fps]s).
         If durations differ from seq_len/fps, GT and predicted segments live on
         DIFFERENT time axes -> IoU ~0 -> AP ~0.
  B3     are GT fake_periods actually present for test clips?
  B5     fully-fake vs partial-fake distribution; GT segment durations.

Pure analysis of the annotation JSON — no model, no GPU, runs in seconds.

Usage:
  .venv\\Scripts\\python.exe scripts\\diagnose_localization.py --config configs\\lavdf_talvad_x.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.config import load_config  # noqa: E402
from datasets.annotation_loader import _load_metadata_index  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lavdf_talvad_x.yaml")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seq_len = int(cfg["training"]["sequence_length"])
    fps = int(cfg["preprocessing"]["fps"])
    ann_json = cfg["dataset"]["annotation_json"]
    window_s = seq_len / fps  # the [0, window_s] axis the decoder currently assumes

    index = _load_metadata_index(ann_json)
    # De-duplicate (index has both padded + int keys -> same entry object)
    seen = set()
    entries = []
    for e in index.values():
        key = e["file"]
        if key in seen:
            continue
        seen.add(key)
        if e.get("split") == args.split:
            entries.append(e)

    print("=" * 68)
    print(f"  LOCALIZATION DIAGNOSTIC — split='{args.split}'")
    print("=" * 68)
    print(f"  seq_len={seq_len}  fps={fps}  -> decoder time window = [0, {window_s:.2f}]s")
    print(f"  test clips in annotation: {len(entries)}")
    if not entries:
        print("  !! no entries for this split — check annotation_json/split field")
        return

    durations = np.array([float(e.get("duration", 0.0)) for e in entries])
    n_fake = sum(1 for e in entries if e.get("fake_periods"))
    print("\n  ── durations (seconds) ──")
    print(f"   min={durations.min():.2f}  median={np.median(durations):.2f}  "
          f"mean={durations.mean():.2f}  max={durations.max():.2f}")
    print(f"   clips longer than the {window_s:.2f}s decoder window: "
          f"{int((durations > window_s + 1e-6).sum())} / {len(entries)} "
          f"({100.0*(durations > window_s).mean():.1f}%)")

    # B3: GT presence
    print("\n  ── GT fake_periods (B3) ──")
    print(f"   clips with >=1 fake_period: {n_fake} / {len(entries)} ({100.0*n_fake/len(entries):.1f}%)")

    # B1/B2: how many fake_periods fall (partly) outside the decoder's [0, window_s]
    all_seg_durs = []
    n_segs = 0
    n_segs_end_after_window = 0
    n_segs_start_after_window = 0
    fully_fake = 0
    for e in entries:
        fp = e.get("fake_periods") or []
        dur = float(e.get("duration", 0.0))
        for s, t in fp:
            s, t = float(s), float(t)
            n_segs += 1
            all_seg_durs.append(t - s)
            if t > window_s + 1e-6:
                n_segs_end_after_window += 1
            if s > window_s + 1e-6:
                n_segs_start_after_window += 1
            # whole-clip fake?
            if dur > 0 and (t - s) >= 0.95 * dur:
                fully_fake += 1

    print("\n  ── GT segment vs decoder-window mismatch (B1/B2) — THE KEY TEST ──")
    print(f"   total GT segments: {n_segs}")
    if n_segs:
        print(f"   segments ENDING after {window_s:.2f}s (decoder can't reach them): "
              f"{n_segs_end_after_window} ({100.0*n_segs_end_after_window/n_segs:.1f}%)")
        print(f"   segments STARTING after {window_s:.2f}s (decoder fully blind): "
              f"{n_segs_start_after_window} ({100.0*n_segs_start_after_window/n_segs:.1f}%)")
        sd = np.array(all_seg_durs)
        print(f"\n  ── GT segment durations (B5) ──")
        print(f"   min={sd.min():.2f}  median={np.median(sd):.2f}  mean={sd.mean():.2f}  max={sd.max():.2f}")
        print(f"   whole-clip ('fully fake') segments: {fully_fake} ({100.0*fully_fake/n_segs:.1f}%)")

    print("\n" + "=" * 68)
    print("  VERDICT")
    print("=" * 68)
    pct_out = 100.0 * n_segs_end_after_window / max(n_segs, 1)
    if pct_out > 5.0:
        print(f"  >> CONFIRMED units/window bug: {pct_out:.1f}% of GT segments end past the")
        print(f"     {window_s:.2f}s axis the decoder uses. Predicted segments (frame/{fps}) and")
        print(f"     GT segments (real seconds over full duration) are on DIFFERENT time axes.")
        print(f"     FIX: decode predicted-segment times with duration/seq_len per clip,")
        print(f"     matching get_frame_labels — not a fixed fps. (Part C: eval-side patch.)")
    else:
        print(f"  >> Window mismatch is minor ({pct_out:.1f}%). Look elsewhere: decoder")
        print(f"     threshold / min-duration / IoU units. (Part C: decoder tuning.)")


if __name__ == "__main__":
    main()
