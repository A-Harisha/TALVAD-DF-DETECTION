"""
scripts/test_talvad_x.py — TALVAD-X test-set evaluation driver.

PART 1 (this script): IN-DOMAIN LAV-DF test.
  Evaluates the trained ablation checkpoints (A0..A5) on the official LAV-DF
  *test* split and aggregates one table: detection AUC/F1/BalAcc + frame-AUC +
  temporal localisation (AP@{50,75,90,95}, mAP) for the TFLH head and the CMIS-2
  inconsistency map. Each variant is loaded into the SAME architecture it was
  trained with (its CMIS flags are re-applied via eval_talvad_x --set), and the
  segment decoder is the duration-corrected one (frame -> t*duration/seq_len).

PART 2 (zero-shot cross-dataset): NOT wired here yet — it has prerequisites.
  The current scripts/eval_crossdataset.py targets the LEGACY BiGRUSoftplusDetector,
  and the only cross-dataset cache (data/crossdataset/fakeavceleb_balanced) is the
  OLD pipeline, not the 666-d TALVAD-X format. To run zero-shot on FakeAVCeleb /
  DeepfakeTIMIT with THIS model you must first:
    1. build the target CSV (scripts/build_deepfaketimit_csv.py / FakeAVCeleb CSV),
    2. extract it through scripts/extract_talvad_x.py (same 666-d pipeline → a
       packed cache), then
    3. score it with the LAV-DF-trained checkpoint + LAV-DF scaler (detection-only;
       no fake_periods GT on FakeAVCeleb). See --cross_pack below: once a target
       pack exists this driver can be extended to consume it.

Usage:
  python scripts/test_talvad_x.py --config configs/lavdf_talvad_x.yaml --seed 42
  python scripts/test_talvad_x.py --config configs/lavdf_talvad_x.yaml --variants A0_baseline A4_full
  python scripts/test_talvad_x.py --config configs/lavdf_talvad_x.yaml --limit 3000   # quick check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.ablate_talvad_x import VARIANTS          # noqa: E402  (variant -> model flags)
from scripts.eval_talvad_x import run as eval_run     # noqa: E402


def _g(d, *keys, default=float("nan")):
    """Safe nested-get: _g(res, 'localization', 'AP@50')."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return default
        cur = cur[k]
    return cur


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS.keys()))
    ap.add_argument("--limit", type=int, default=0, help="subsample N clips for a quick check")
    ap.add_argument("--ckpt_tmpl", default="checkpoints/talvad_x_{variant}_s{seed}",
                    help="checkpoint dir template")
    ap.add_argument("--out", default="logs/talvad_x_test_table.md")
    args = ap.parse_args()

    rows = []
    for v in args.variants:
        ckpt_dir = args.ckpt_tmpl.format(variant=v, seed=args.seed)
        if not (REPO / ckpt_dir / "best_model.pt").exists():
            print(f"  [skip] {v}: no checkpoint at {ckpt_dir}")
            continue
        print("\n" + "#" * 64 + f"\n#  TEST  {v}  (seed {args.seed})  ->  {ckpt_dir}\n" + "#" * 64)
        res = eval_run(
            args.config, split=args.split,
            out_dir_override=f"logs/talvad_x/test_{v}",
            limit=args.limit, ckpt_dir_override=str(REPO / ckpt_dir),
            sets=VARIANTS[v],
        )
        rows.append({
            "variant": v,
            "det_auc": _g(res, "video_metrics_05", "auc"),
            "det_f1": _g(res, "video_metrics_05", "f1"),
            "frame_auc": _g(res, "frame_auc"),
            "ap50": _g(res, "localization", "AP@50"),
            "ap75": _g(res, "localization", "AP@75"),
            "map": _g(res, "localization", "mAP"),
            "tflh_map": _g(res, "tflh_localization", "best_map"),
            "cmis_frame_auc": _g(res, "cmis_localization", "frame_auc"),
            "cmis_map": _g(res, "cmis_localization", "best_map"),
        })

    # ── aggregate table ──────────────────────────────────────────────────────
    hdr = ("| variant | det_AUC | det_F1 | frame_AUC | AP@50 | AP@75 | mAP "
           "| TFLH best_mAP | CMIS frame_AUC | CMIS best_mAP |")
    sep = "|" + "---|" * 10
    lines = [f"# TALVAD-X — {args.split} set (seed {args.seed})", "",
             "Detection + temporal localisation (duration-corrected decoder).", "",
             hdr, sep]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['det_auc']:.4f} | {r['det_f1']:.4f} | {r['frame_auc']:.4f} "
            f"| {r['ap50']:.4f} | {r['ap75']:.4f} | {r['map']:.4f} | {r['tflh_map']:.4f} "
            f"| {r['cmis_frame_auc']:.4f} | {r['cmis_map']:.4f} |"
        )
    # CMIS deltas vs baseline (honest read-off)
    by = {r["variant"]: r for r in rows}
    if "A0_baseline" in by and "A4_full" in by:
        d_map = by["A4_full"]["tflh_map"] - by["A0_baseline"]["tflh_map"]
        d_det = by["A4_full"]["det_auc"] - by["A0_baseline"]["det_auc"]
        lines += ["", f"**CMIS effect (A4 − A0):** det_AUC {d_det:+.4f} · TFLH mAP {d_map:+.4f}"]

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
