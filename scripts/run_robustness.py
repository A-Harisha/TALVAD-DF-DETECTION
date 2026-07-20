"""
scripts/run_robustness.py — orchestrate the full robustness experiment:
for every (distortion, level) variant produced by make_robustness_videos.py,
run the two-venv TALVAD-L target extraction (avsr -> main -> pack with saved
LAV-DF PCA) and then eval_robustness_l.py. Resumable: variants whose result
JSON already exists are skipped.

Order of operations:
  1) python scripts/make_robustness_videos.py --config ... --n_clips 200
  2) python scripts/run_robustness.py --config configs/lavdf_talvad_l.yaml
  3) python scripts/plot_robustness.py            -> figures/fig_robustness.pdf

Smoke first:  --distortions vnoise --levels 1 --limit 5
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VISUAL = ["saturation", "contrast", "blockwise", "vnoise", "blur", "jpeg", "vcompress"]
AUDIO = ["anoise", "pitch", "reverb", "acompress"]
ALL = VISUAL + AUDIO


def _py(venv: str) -> str:
    p = REPO / venv / "Scripts" / "python.exe"      # windows
    if not p.exists():
        p = REPO / venv / "bin" / "python"          # posix
    if not p.exists():
        raise FileNotFoundError(f"no python in {REPO / venv}")
    return str(p)


def _run(cmd: list[str], tag: str) -> bool:
    print(f"  [{tag}] {' '.join(cmd[1:])}", flush=True)
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        print(f"  [{tag}] FAILED (rc={r.returncode}) — continuing with next variant")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lavdf_talvad_l.yaml")
    ap.add_argument("--videos_root", default="datasets/robustness")
    ap.add_argument("--distortions", default="all")
    ap.add_argument("--levels", default="1,2,3,4,5")
    ap.add_argument("--limit", type=int, default=0, help="subsample N clips per variant (0=all)")
    ap.add_argument("--workers", type=int, default=4,
                    help="variants processed concurrently, sharing the GPU (24GB fits ~4-6)")
    ap.add_argument("--venv", default=".venv")
    ap.add_argument("--venv_lip", default=".venv-lip")
    ap.add_argument("--ckpt", default="checkpoints/talvad_l_full/best_model.pt",
                    help="DFL paper model; passed through to eval_robustness_l.py")
    ap.add_argument("--include_clean", action="store_true", default=True)
    args = ap.parse_args()

    py, py_lip = _py(args.venv), _py(args.venv_lip)
    extract = str(REPO / "scripts" / "extract_talvad_l.py")
    evaler = str(REPO / "scripts" / "eval_robustness_l.py")
    vroot = REPO / args.videos_root
    gt = str(vroot / "gt.json")
    out_dir = REPO / "logs" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    dists = ALL if args.distortions == "all" else [d.strip() for d in args.distortions.split(",")]
    levels = [int(x) for x in args.levels.split(",")]

    variants = []
    if args.include_clean:
        variants.append(("clean", 0, vroot / "csv" / "clean.csv"))
    for d in dists:
        for lv in levels:
            variants.append((d, lv, vroot / "csv" / f"{d}_L{lv}.csv"))

    def process_variant(dist, lvl, csv):
        name = f"rob_{dist}_L{lvl}"
        result = out_dir / f"{dist}_L{lvl}.json"
        if result.exists():
            print(f"skip {name} (result exists)", flush=True)
            return "skip"
        if not csv.exists():
            print(f"skip {name} (no CSV {csv})", flush=True)
            return "nocsv"
        print(f"=== {name} start ===", flush=True)
        common = ["--config", args.config, "--target_csv", str(csv), "--target_name", name]
        if args.limit:
            common += ["--limit", str(args.limit)]
        # avsr and main are independent; run avsr first (its own venv), then main,
        # then pack (needs both), then eval. Each variant is one pipeline; N
        # variants run concurrently, sharing the GPU.
        ok = (_run([py_lip, extract, "--stage", "avsr"] + common, f"{name}/avsr")
              and _run([py, extract, "--stage", "main"] + common, f"{name}/main")
              and _run([py, extract, "--stage", "pack"] + common, f"{name}/pack")
              and _run([py, evaler, "--config", args.config, "--target_name", name,
                        "--gt", gt, "--distortion", dist, "--level", str(lvl),
                        "--ckpt", args.ckpt, "--out", str(result)], f"{name}/eval"))
        print(f"=== {name} {'OK' if ok else 'FAILED'} ===", flush=True)
        return "done" if ok else "failed"

    done = failed = 0
    if args.workers <= 1:
        outcomes = [process_variant(d, lv, c) for d, lv, c in variants]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            outcomes = list(ex.map(lambda v: process_variant(*v), variants))
    done = sum(o in ("done", "skip") for o in outcomes)
    failed = sum(o == "failed" for o in outcomes)

    print(f"\nrobustness sweep: {done} done/skip, {failed} failed, results in {out_dir}")
    print("next: python scripts/plot_robustness.py")


if __name__ == "__main__":
    main()
