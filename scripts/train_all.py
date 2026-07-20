"""
scripts/train_all.py — train the ENTIRE TALVAD-X system end-to-end, one command.

The architecture is intentionally modular: FROZEN encoders -> three separately
trained heads -> parameter-free fusion. (Keeping the encoders frozen is what gives
cross-dataset generalization; an end-to-end fine-tune would re-introduce the
generator-fingerprint overfitting we are fighting.) So "training the whole model"
means running every trainable stage in the right order — which is what this driver
does. Each stage is SKIPPED if its artifact already exists (resumable); pass --force
to retrain.

Stages
  1. Supervised CMIS-2 detector  (models/talvad_x.py)  x seeds
        -> checkpoints/talvad_x_<variant>_s<seed>/best_model.pt
  2. Correspondence prior        (models/correspondence.py, REAL-only SSL)
        -> checkpoints/correspondence/prior.pt
  3. AVSR head                   (logreg on frozen auto_avsr feats; refit at eval time)
        -> reported by train_test_avsr.py  (no persisted artifact by design)
  4. Fusion evaluations
        -> logs/fusion_eval.json        (supervised  + prior)
        -> logs/fusion_avsr_eval.json   (AVSR head   + prior, Route B headline)

Prereqs (FEATURES must already be extracted — this driver TRAINS, it does not extract):
  - LAV-DF 666-d pack          : data/talvad_x/fps{fps}_sl{sl}/packed
  - AVSR feats (train + target): data/talvad_x_crossdataset/{lavdf_train,fakeavceleb_bal}/avsr
  - bal 666-d pack (Route B)   : data/talvad_x_crossdataset/fakeavceleb_bal/packed
A missing prereq downgrades the affected stage to a SKIP with the exact extract command.

Usage
  python scripts/train_all.py --config configs/lavdf_talvad_x.yaml
  python scripts/train_all.py --config configs/lavdf_talvad_x.yaml --variant P1_best --seeds 42 1337 2024
  python scripts/train_all.py --config configs/lavdf_talvad_x.yaml --skip supervised   # prior + fusion only
  python scripts/train_all.py --config configs/lavdf_talvad_x.yaml --force             # retrain everything
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from utils.config import load_config          # noqa: E402
from ablate_talvad_x import VARIANTS           # noqa: E402  (reuse the override ladder)

PY = sys.executable
SCRIPTS = REPO / "scripts"
CKPT = REPO / "checkpoints"
XBASE = REPO / "data" / "talvad_x_crossdataset"


def run(cmd: list[str], title: str) -> int:
    """Run a child stage, inheriting stdout so progress streams live."""
    print("\n" + "=" * 72)
    print(f"  ▶ {title}")
    print("    " + " ".join(cmd))
    print("=" * 72, flush=True)
    t0 = time.perf_counter()
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    dt = time.perf_counter() - t0
    status = "OK" if rc == 0 else f"FAILED (rc={rc})"
    print(f"  ← {title}: {status}  [{dt/60:.1f} min]", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", default="P1_a0_aug",
                    help="supervised CMIS-2 variant from ablate_talvad_x.VARIANTS")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1337, 2024])
    ap.add_argument("--prior_epochs", type=int, default=20)
    ap.add_argument("--sup_epochs", type=int, default=0, help="0 = use the config's epochs")
    ap.add_argument("--train_target", default="lavdf_train")
    ap.add_argument("--test_target", default="fakeavceleb_bal")
    ap.add_argument("--targets", nargs="*", default=["fakeavceleb", "deepfaketimit", "lavdf_selfcheck"])
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["supervised", "prior", "avsr", "fusion"])
    ap.add_argument("--force", action="store_true", help="retrain even if artifacts exist")
    args = ap.parse_args()

    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if args.variant not in VARIANTS:
        sys.exit(f"unknown --variant {args.variant!r}. available: {', '.join(sorted(VARIANTS))}")

    cfg = load_config(args.config)
    fps = int(cfg["preprocessing"]["fps"]); sl = int(cfg["training"]["sequence_length"])
    lavdf_pack = REPO / "data" / "talvad_x" / f"fps{fps}_sl{sl}" / "packed" / "features.npy"
    bal_pack = XBASE / args.test_target / "packed" / "features.npy"
    avsr_tr = XBASE / args.train_target / "avsr" / "features.npz"
    avsr_te = XBASE / args.test_target / "avsr" / "features.npz"

    print("=" * 72)
    print("  TALVAD-X — TRAIN ENTIRE PIPELINE")
    print(f"  variant={args.variant}  seeds={args.seeds}  prior_epochs={args.prior_epochs}")
    print(f"  skip={args.skip or '[]'}  force={args.force}")
    print("=" * 72)

    summary: list[str] = []

    # ── Stage 1: supervised CMIS-2 detector across seeds ──────────────────────
    if "supervised" in args.skip:
        summary.append("supervised: SKIPPED (--skip)")
    elif not lavdf_pack.exists():
        sys.exit(f"\n[FATAL] LAV-DF pack missing: {lavdf_pack}\n"
                 f"  Extract it first with scripts/extract_talvad_x.py (the LAV-DF split mode).")
    else:
        overrides = []
        for kv in VARIANTS[args.variant]:
            overrides += ["--set", kv]
        for s in args.seeds:
            done = (CKPT / f"talvad_x_{args.variant}_s{s}" / "best_model.pt").exists()
            if done and not args.force:
                print(f"  · supervised {args.variant} seed {s}: already trained — skip")
                summary.append(f"supervised {args.variant} s{s}: SKIP (exists)")
                continue
            cmd = [PY, str(SCRIPTS / "train_talvad_x.py"), "--config", args.config,
                   "--seed", str(s), "--tag", f"{args.variant}_s{s}"] + overrides
            if args.sup_epochs > 0:
                cmd += ["--epochs", str(args.sup_epochs)]
            rc = run(cmd, f"Stage 1 · supervised {args.variant} · seed {s}")
            summary.append(f"supervised {args.variant} s{s}: {'OK' if rc == 0 else 'FAIL'}")

    # ── Stage 2: correspondence prior (REAL-only SSL) ─────────────────────────
    prior_pt = CKPT / "correspondence" / "prior.pt"
    if "prior" in args.skip:
        summary.append("prior: SKIPPED (--skip)")
    elif prior_pt.exists() and not args.force:
        print(f"\n  · correspondence prior already trained ({prior_pt}) — skip")
        summary.append("prior: SKIP (exists)")
    else:
        cmd = [PY, str(SCRIPTS / "train_correspondence.py"), "--config", args.config,
               "--epochs", str(args.prior_epochs)]
        rc = run(cmd, "Stage 2 · correspondence prior")
        summary.append(f"prior: {'OK' if rc == 0 else 'FAIL'}")

    # ── Stage 3: AVSR head (logreg; refit at eval time — just report it) ───────
    if "avsr" in args.skip:
        summary.append("avsr: SKIPPED (--skip)")
    elif not (avsr_tr.exists() and avsr_te.exists()):
        miss = avsr_tr if not avsr_tr.exists() else avsr_te
        print(f"\n  · AVSR feats missing ({miss}) — skip. Extract with scripts/extract_avsr.py")
        summary.append("avsr: SKIP (feats missing)")
    else:
        cmd = [PY, str(SCRIPTS / "train_test_avsr.py"),
               "--train_target", args.train_target, "--test_target", args.test_target]
        rc = run(cmd, "Stage 3 · AVSR head (logreg, frozen feats)")
        summary.append(f"avsr: {'OK' if rc == 0 else 'FAIL'}")

    # ── Stage 4: fusion evaluations ───────────────────────────────────────────
    if "fusion" in args.skip:
        summary.append("fusion: SKIPPED (--skip)")
    else:
        if prior_pt.exists():
            cmd = [PY, str(SCRIPTS / "eval_fusion.py"), "--config", args.config,
                   "--supervised", args.variant, "--seeds", *map(str, args.seeds),
                   "--prior_dir", "checkpoints/correspondence", "--targets", *args.targets]
            rc = run(cmd, "Stage 4a · fusion: supervised + prior")
            summary.append(f"fusion(sup+prior): {'OK' if rc == 0 else 'FAIL'}")

            if bal_pack.exists() and avsr_tr.exists() and avsr_te.exists():
                cmd = [PY, str(SCRIPTS / "eval_fusion_avsr.py"), "--config", args.config,
                       "--train_target", args.train_target, "--test_target", args.test_target,
                       "--prior_dir", "checkpoints/correspondence"]
                rc = run(cmd, "Stage 4b · fusion: AVSR head + prior (Route B headline)")
                summary.append(f"fusion(avsr+prior): {'OK' if rc == 0 else 'FAIL'}")
            else:
                print(f"\n  · Route B fusion needs the bal 666-d pack + AVSR feats. Missing one — skip.\n"
                      f"    Extract bal pack:\n"
                      f"    python scripts/extract_talvad_x.py --config {args.config} "
                      f"--target_csv data/fakeavceleb_balanced_metadata.csv "
                      f"--target_name {args.test_target} --root_dir datasets/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2")
                summary.append("fusion(avsr+prior): SKIP (bal pack/AVSR missing)")
        else:
            print("\n  · no prior checkpoint — skipping fusion (train the prior first).")
            summary.append("fusion: SKIP (no prior)")

    print("\n" + "=" * 72)
    print("  PIPELINE SUMMARY")
    for line in summary:
        print(f"   - {line}")
    print("=" * 72)


if __name__ == "__main__":
    main()
