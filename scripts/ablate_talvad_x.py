"""
scripts/ablate_talvad_x.py — the CMIS-2 ablation ladder for TALVAD-X.

Runs each variant across seeds in isolated subprocesses (clean CUDA state),
parses the dev AUC / balanced-acc / F1 from train_talvad_x.py's [RESULT] line,
and aggregates mean +/- std per variant into logs/talvad_x_ablation.{json,md}.

The ladder isolates each CMIS-2 contribution (results, not narrative):
  A0 baseline      : no CMIS branch
  A1 spectral      : enriched-spectral attention, NO consistency coupling
  A2 consistency   : inconsistency feature + InfoNCE, NO spectral attention
  A3 coupling      : consistency-GATED spectral attention (the core claim)
  A4 full          : coupling + consistency feature + InfoNCE  (warm init)
  A5 cold-init     : A4 but gates start ~0 (proves warm-start matters)

Required reading: A3 > max(A1, A2)  and  A4 >= A3 ; A4 > A5 ; all > A0.

Usage:
  python scripts/ablate_talvad_x.py --config configs/lavdf_talvad_x.yaml --seeds 42 1337 2024
  python scripts/ablate_talvad_x.py --config configs/lavdf_talvad_x.yaml --variants A0_baseline A4_full --seeds 42
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable

VARIANTS: dict[str, list[str]] = {
    "A0_baseline": ["model.cmis.enabled=false"],
    "A1_spectral": [
        "model.cmis.enabled=true", "model.cmis.spectral_branch=true",
        "model.cmis.consistency_branch=false", "model.cmis.coupling=false",
        "model.cmis.contrastive=false",
        "training.cmis_incon_loss_weight=0.0", "training.contrastive_loss_weight=0.0",
    ],
    "A2_consistency": [
        "model.cmis.enabled=true", "model.cmis.spectral_branch=false",
        "model.cmis.consistency_branch=true", "model.cmis.coupling=false",
        "model.cmis.contrastive=true", "training.contrastive_loss_weight=0.1",
    ],
    "A3_coupling": [
        "model.cmis.enabled=true", "model.cmis.spectral_branch=true",
        "model.cmis.consistency_branch=false", "model.cmis.coupling=true",
        "model.cmis.contrastive=false", "training.contrastive_loss_weight=0.0",
    ],
    "A4_full": [
        "model.cmis.enabled=true", "model.cmis.spectral_branch=true",
        "model.cmis.consistency_branch=true", "model.cmis.coupling=true",
        "model.cmis.contrastive=true", "model.cmis.warm_init=true",
    ],
    "A5_coldinit": [
        "model.cmis.enabled=true", "model.cmis.spectral_branch=true",
        "model.cmis.consistency_branch=true", "model.cmis.coupling=true",
        "model.cmis.contrastive=true", "model.cmis.warm_init=false",
    ],
    # ── Phase-1 (v2) cross-dataset ladder — all build on A4_full and isolate each
    # generalization lever; baseline for the lift is the gate's A4_full. ──
    "P1_motion": [  # A4 + motion-aware CMIS (articulatory proxy via appearance frame-delta)
        "model.cmis.enabled=true", "model.cmis.spectral_branch=true",
        "model.cmis.consistency_branch=true", "model.cmis.coupling=true",
        "model.cmis.contrastive=true", "model.cmis.warm_init=true",
        "model.cmis.motion=true",
    ],
    "P1_aug": [  # A4 + DG augmentation (spectral-aug + instance-norm), validated +0.03 in LODO
        "model.cmis.enabled=true", "model.cmis.spectral_branch=true",
        "model.cmis.consistency_branch=true", "model.cmis.coupling=true",
        "model.cmis.contrastive=true", "model.cmis.warm_init=true",
        "training.spectral_aug_p=0.3", "training.instance_norm_p=0.3",
    ],
    "P1_swa": [  # A4 + SWA weight averaging
        "model.cmis.enabled=true", "model.cmis.spectral_branch=true",
        "model.cmis.consistency_branch=true", "model.cmis.coupling=true",
        "model.cmis.contrastive=true", "model.cmis.warm_init=true",
        "training.swa_enabled=true",
    ],
    "P1_full": [  # the full Phase-1 free-tier recipe = A4 + motion + DG-aug + SWA
        "model.cmis.enabled=true", "model.cmis.spectral_branch=true",
        "model.cmis.consistency_branch=true", "model.cmis.coupling=true",
        "model.cmis.contrastive=true", "model.cmis.warm_init=true",
        "model.cmis.motion=true",
        "training.spectral_aug_p=0.3", "training.instance_norm_p=0.3",
        "training.swa_enabled=true",
    ],
    # ── Phase-1 v2 (post-gate): the gate showed A0 (plain) transfers best and the
    # spectral/coupling machinery HURTS. So put the DG regime on the A0 base, and
    # test articulatory consistency WITHOUT the harmful frequency fingerprint. ──
    "P1_a0_aug": [  # plain model + DG augmentation (breaks the spectral-input shortcut)
        "model.cmis.enabled=false",
        "training.spectral_aug_p=0.3", "training.instance_norm_p=0.3",
    ],
    "P1_a0_swa": [  # plain model + SWA (flat-minimum weight averaging)
        "model.cmis.enabled=false", "training.swa_enabled=true",
    ],
    "P1_a0_full": [  # plain model + full DG regime (aug + instance-norm + SWA) — main v2 contender
        "model.cmis.enabled=false",
        "training.spectral_aug_p=0.3", "training.instance_norm_p=0.3",
        "training.swa_enabled=true",
    ],
    "P1_artic": [  # articulatory consistency only: motion+consistency, NO spectral, NO coupling
        "model.cmis.enabled=true", "model.cmis.spectral_branch=false",
        "model.cmis.consistency_branch=true", "model.cmis.coupling=false",
        "model.cmis.contrastive=true", "model.cmis.motion=true",
        "training.contrastive_loss_weight=0.1",
        "training.spectral_aug_p=0.3", "training.instance_norm_p=0.3",
        "training.swa_enabled=true",
    ],
    "P1_a0_nobal": [  # diagnostic: plain model WITHOUT balanced_sampler (is the sampler hurting transfer?)
        "model.cmis.enabled=false", "training.balanced_sampler=false",
    ],
    # ── Phase-1 confirmation (post-results): the winner was spectral-aug+instance-norm
    # (NO SWA — SWA cost ~0.11 via full-schedule overfitting), sampler off. ──
    "P1_best": [  # locked recipe: plain model + aug + instance-norm + sampler OFF, no SWA
        "model.cmis.enabled=false",
        "training.spectral_aug_p=0.3", "training.instance_norm_p=0.3",
        "training.balanced_sampler=false",
    ],
    "P1_artic_clean": [  # FAIR articulatory test: motion+consistency, no spectral/coupling, aug on, NO SWA, sampler off
        "model.cmis.enabled=true", "model.cmis.spectral_branch=false",
        "model.cmis.consistency_branch=true", "model.cmis.coupling=false",
        "model.cmis.contrastive=true", "model.cmis.motion=true",
        "training.contrastive_loss_weight=0.1",
        "training.spectral_aug_p=0.3", "training.instance_norm_p=0.3",
        "training.balanced_sampler=false",
    ],
}

_RESULT = re.compile(r"\[RESULT\].*?AUC=([0-9.]+)\s+bal_acc=([0-9.]+)\s+f1=([0-9.]+)")


def run_one(config: str, variant: str, seed: int, epochs: int,
            idx: int, total: int, log_dir: Path, stream: bool = True) -> dict | None:
    overrides = []
    for kv in VARIANTS[variant]:
        overrides += ["--set", kv]
    cmd = [PY, str(REPO / "scripts" / "train_talvad_x.py"), "--config", config,
           "--seed", str(seed), "--tag", f"{variant}_s{seed}"] + overrides
    if epochs > 0:
        cmd += ["--epochs", str(epochs)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               TALVAD_NO_TQDM="1")  # disable child progress-bar flood; print checkpoints instead

    log_path = log_dir / f"{variant}_s{seed}.log"
    started = time.perf_counter()
    print("\n" + "═" * 64, flush=True)
    print(f"  ▶ RUN {idx}/{total}  ·  {variant}  ·  seed {seed}  ·  "
          f"{datetime.now():%H:%M:%S}", flush=True)
    print(f"    overrides: {' '.join(VARIANTS[variant])}", flush=True)
    print(f"    live log : {log_path}", flush=True)
    print("═" * 64, flush=True)

    # Stream child stdout line-by-line so a backgrounded run shows live progress,
    # and tee everything to a per-run log file for later inspection.
    lines: list[str] = []
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
        )
        for line in proc.stdout:  # type: ignore[union-attr]
            lines.append(line)
            lf.write(line)
            if stream:
                # Indent child output so it's visually nested under the run header.
                sys.stdout.write("    │ " + line)
                sys.stdout.flush()
        proc.wait()

    out = "".join(lines)
    elapsed = time.perf_counter() - started
    m = _RESULT.search(out)
    if not m:
        print(f"  ✗ [FAIL] {variant} s{seed} (rc={proc.returncode}, {elapsed/60:.1f}m) "
              f"— no RESULT line. See {log_path}", flush=True)
        tail = "\n".join(out.strip().splitlines()[-6:])
        print("   " + tail.replace("\n", "\n   "), flush=True)
        return None
    auc, bal, f1 = float(m.group(1)), float(m.group(2)), float(m.group(3))
    print(f"  ✓ [ok] {variant:16s} s{seed}: AUC={auc:.4f} bal_acc={bal:.4f} f1={f1:.4f} "
          f"  ({elapsed/60:.1f}m)", flush=True)
    return {"variant": variant, "seed": seed, "auc": auc, "bal_acc": bal, "f1": f1,
            "elapsed_min": round(elapsed / 60, 2)}


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS.keys()))
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1337, 2024])
    ap.add_argument("--epochs", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    run_log_dir = REPO / "logs" / "ablation_runs"
    run_log_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.variants) * len(args.seeds)
    print("╔" + "═" * 62 + "╗")
    print(f"║  TALVAD-X CMIS-2 ABLATION  ·  {len(args.variants)} variants × "
          f"{len(args.seeds)} seeds = {total} runs".ljust(63) + "║")
    print(f"║  variants: {', '.join(args.variants)}".ljust(63) + "║")
    print(f"║  seeds   : {args.seeds}".ljust(63) + "║")
    print(f"║  per-run logs → {run_log_dir}".ljust(63) + "║")
    print("╚" + "═" * 62 + "╝", flush=True)

    rows = []
    grand_start = time.perf_counter()
    idx = 0
    for v in args.variants:
        for s in args.seeds:
            idx += 1
            r = run_one(args.config, v, s, args.epochs, idx, total, run_log_dir)
            if r:
                rows.append(r)
            # Running progress + ETA across the whole grid.
            done, elapsed = idx, time.perf_counter() - grand_start
            eta = (elapsed / done) * (total - done)
            print(f"  ── progress {done}/{total} runs  ·  elapsed {elapsed/60:.1f}m  "
                  f"·  grid ETA {eta/60:.1f}m  ·  ok={len(rows)} fail={done-len(rows)}",
                  flush=True)

    # aggregate
    agg = {}
    for v in args.variants:
        vr = [r for r in rows if r["variant"] == v]
        if vr:
            aucs = np.array([r["auc"] for r in vr])
            agg[v] = {"n": len(vr), "auc_mean": float(aucs.mean()), "auc_std": float(aucs.std()),
                      "bal_mean": float(np.mean([r["bal_acc"] for r in vr])),
                      "f1_mean": float(np.mean([r["f1"] for r in vr]))}

    out_dir = REPO / "logs"
    (out_dir / "talvad_x_ablation.json").write_text(
        json.dumps({"runs": rows, "aggregate": agg}, indent=2), encoding="utf-8")
    lines = ["# TALVAD-X CMIS-2 ablation", "",
             "| variant | seeds | dev AUC (mean±std) | bal_acc | f1 |",
             "|---|---|---|---|---|"]
    for v in args.variants:
        if v in agg:
            a = agg[v]
            lines.append(f"| {v} | {a['n']} | {a['auc_mean']:.4f} ± {a['auc_std']:.4f} | "
                         f"{a['bal_mean']:.4f} | {a['f1_mean']:.4f} |")
    if "A3_coupling" in agg and "A1_spectral" in agg and "A2_consistency" in agg:
        d = agg["A3_coupling"]["auc_mean"] - max(agg["A1_spectral"]["auc_mean"], agg["A2_consistency"]["auc_mean"])
        lines += ["", f"**Coupling effect** A3 − max(A1,A2) = {d:+.4f} AUC"]
    if "A4_full" in agg and "A0_baseline" in agg:
        lines.append(f"**CMIS-2 total** A4 − A0 = {agg['A4_full']['auc_mean'] - agg['A0_baseline']['auc_mean']:+.4f} AUC")
    (out_dir / "talvad_x_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n  -> {out_dir / 'talvad_x_ablation.md'}")


if __name__ == "__main__":
    main()
