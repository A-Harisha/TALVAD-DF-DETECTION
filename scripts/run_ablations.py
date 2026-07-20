"""
scripts/run_ablations.py

Reproducible ablation + multi-seed driver for the CMIS contribution.

For each (variant, seed) it deep-copies the base config, applies a small set of
overrides, rewrites experiment_name / checkpoint_dir / logs_dir to a unique slot,
dumps the resulting YAML to configs/_ablation_generated/ (kept as a provenance
record for the paper appendix), then launches training in a *subprocess*.

Subprocess isolation matters: running 10+ trainings in one process leaks CUDA
memory and cuDNN autotune state across runs; a fresh process per run guarantees
each variant starts from an identical, clean state.

Ablation grid (CMIS story — isolates the novel coupling):
  b0_baseline           CMIS off                        (≈ v7)
  b1_spectral_nocouple  spectral attention, no coupling (B1)
  b2_cmis_couple        inconsistency coupling, no aux  (B2)
  b3_cmis_full          coupling + supervised incon map (B3, == v8 default)

Component ablations (supporting table):
  no_tdfs / no_tflh / no_av / no_maa

Usage:
  python scripts/run_ablations.py --base configs/lavdf_talvad_v8.yaml
  python scripts/run_ablations.py --variants b0_baseline b2_cmis_couple b3_cmis_full
  python scripts/run_ablations.py --seeds 42 1337 2024 --variants b3_cmis_full b0_baseline
  python scripts/run_ablations.py --dry-run        # generate configs, run nothing
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.config import load_config


# Each variant is a flat list of (dotted_key, value) overrides applied on top of
# the base config. Keys use dots to walk nested dicts (created if absent).
VARIANTS: dict[str, list[tuple[str, object]]] = {
    # ── CMIS ablation (the headline) ─────────────────────────────────────────
    "b0_baseline": [
        ("model.cmis.enabled", False),
        ("training.cmis_incon_loss_weight", 0.0),
    ],
    "b1_spectral_nocouple": [
        ("model.cmis.enabled", True),
        ("model.cmis.inconsistency_coupling", False),
        ("training.cmis_incon_loss_weight", 0.0),
    ],
    "b2_cmis_couple": [
        ("model.cmis.enabled", True),
        ("model.cmis.inconsistency_coupling", True),
        ("training.cmis_incon_loss_weight", 0.0),
    ],
    "b3_cmis_full": [
        ("model.cmis.enabled", True),
        ("model.cmis.inconsistency_coupling", True),
        ("training.cmis_incon_loss_weight", 0.1),
    ],
    # ── Supporting component ablations ───────────────────────────────────────
    "no_tdfs": [("model.temporal_feature_selection", False)],
    "no_tflh": [
        ("model.frame_localization", False),
        ("training.localization_loss_weight", 0.0),
    ],
    "no_av": [("training.av_agreement_weight", 0.0)],
    "no_maa": [
        ("model.modality_attribution", False),
        ("training.modality_attribution_loss_weight", 0.0),
    ],
}

CMIS_GRID = ["b0_baseline", "b1_spectral_nocouple", "b2_cmis_couple", "b3_cmis_full"]


def _set_dotted(cfg: dict, dotted: str, value: object) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _slug(base_name: str, variant: str, seed: int) -> str:
    return f"{base_name}__{variant}__s{seed}"


def build_variant_config(base_cfg: dict, base_name: str, variant: str, seed: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    for dotted, value in VARIANTS[variant]:
        _set_dotted(cfg, dotted, value)

    slug = _slug(base_name, variant, seed)
    cfg["experiment_name"] = slug
    # Keep each run's checkpoints/logs isolated so nothing is overwritten.
    cfg["training"]["checkpoint_dir"] = f"./checkpoints/{slug}"
    cfg["training"]["logs_dir"] = "./logs"
    if "evaluation" in cfg:
        cfg["evaluation"]["output_dir"] = f"./visualizations/{slug}"
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="CMIS ablation + multi-seed runner")
    parser.add_argument("--base", default="configs/lavdf_talvad_v8.yaml",
                        help="Base config the overrides are applied on top of")
    parser.add_argument("--variants", nargs="+", default=CMIS_GRID,
                        help=f"Variants to run. Available: {list(VARIANTS)} (default: CMIS grid)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Seeds to sweep (multi-seed → significance)")
    parser.add_argument("--gen_dir", default="configs/_ablation_generated",
                        help="Where generated per-run configs are written")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate configs and print the plan, but launch nothing")
    parser.add_argument("--pre_extract", action="store_true",
                        help="Pass --pre_extract through to train.py")
    args = parser.parse_args()

    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        parser.error(f"Unknown variant(s): {unknown}. Available: {list(VARIANTS)}")

    base_cfg = load_config(args.base)
    base_name = Path(args.base).stem
    gen_dir = REPO_ROOT / args.gen_dir
    gen_dir.mkdir(parents=True, exist_ok=True)

    plan = [(v, s) for v in args.variants for s in args.seeds]
    print("=" * 64)
    print(f"  Ablation plan — {len(plan)} run(s)  (base: {args.base})")
    for v, s in plan:
        print(f"    - {v:24s} seed={s}")
    print("=" * 64)

    for variant, seed in plan:
        cfg = build_variant_config(base_cfg, base_name, variant, seed)
        slug = cfg["experiment_name"]
        cfg_path = gen_dir / f"{slug}.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"\n[gen] {cfg_path}")

        if args.dry_run:
            continue

        cmd = [sys.executable, str(REPO_ROOT / "training" / "train.py"),
               "--config", str(cfg_path)]
        if args.pre_extract:
            cmd.append("--pre_extract")
        print(f"[run] {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            print(f"[warn] run '{slug}' exited with code {result.returncode}; continuing.")

    print("\nDone. Aggregate with: python scripts/aggregate_results.py")


if __name__ == "__main__":
    main()
