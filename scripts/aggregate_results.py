"""
scripts/aggregate_results.py

Collect metrics produced by run_ablations.py runs into a single paper-ready
table with mean +/- std across seeds.

Sources per run (logs/<experiment_name>/):
  - validation_threshold.json   (validation set, written every best epoch)
  - eval_temporal_results.json  (test set, written by scripts/eval_temporal.py)

Validation metrics are always available after training; test metrics appear
once eval_temporal.py has been run for that experiment. Both are reported when
present so the table fills in as evaluations complete.

Usage:
  python scripts/aggregate_results.py
  python scripts/aggregate_results.py --logs_dir ./logs --base lavdf_talvad_v8
  python scripts/aggregate_results.py --out logs/ablation_summary
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _parse_slug(name: str) -> tuple[str, str, int] | None:
    """experiment_name == '<base>__<variant>__s<seed>'."""
    parts = name.split("__")
    if len(parts) < 3 or not parts[-1].startswith("s"):
        return None
    try:
        seed = int(parts[-1][1:])
    except ValueError:
        return None
    return "__".join(parts[:-2]), parts[-2], seed


def _mean_std(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def _fmt(m: float, s: float) -> str:
    if math.isnan(m):
        return "—"
    return f"{m:.4f} ± {s:.4f}" if s > 0 else f"{m:.4f}"


# (column header, source file, dotted path into the json)
METRICS = [
    ("val_AUC",     "validation_threshold.json", "auc"),
    ("val_BalAcc",  "validation_threshold.json", "balanced_accuracy"),
    ("val_F1",      "validation_threshold.json", "f1"),
    ("test_AUC",    "eval_temporal_results.json", "video_metrics_tuned.auc"),
    ("test_F1",     "eval_temporal_results.json", "video_metrics_tuned.f1"),
    ("frame_AUC",   "eval_temporal_results.json", "frame_auc"),
    ("mAP",         "eval_temporal_results.json", "mean_ap"),
    ("AP@0.5",      "eval_temporal_results.json", "segment_ap.ap_05"),
]

# Preferred display order; anything else is appended alphabetically.
VARIANT_ORDER = [
    "b0_baseline", "b1_spectral_nocouple", "b2_cmis_couple", "b3_cmis_full",
    "no_tdfs", "no_tflh", "no_av", "no_maa",
]


def _dig(d: dict | None, dotted: str):
    if d is None:
        return None
    node = d
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ablation results")
    parser.add_argument("--logs_dir", default="./logs")
    parser.add_argument("--base", default=None,
                        help="Only include runs whose base config stem matches this")
    parser.add_argument("--out", default="logs/ablation_summary",
                        help="Output path prefix (.md and .csv are written)")
    args = parser.parse_args()

    logs_dir = (REPO_ROOT / args.logs_dir).resolve()
    if not logs_dir.exists():
        parser.error(f"logs_dir not found: {logs_dir}")

    # variant -> metric_col -> list of values across seeds
    collected: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seeds_seen: dict[str, set[int]] = defaultdict(set)
    n_runs = 0

    for exp_dir in sorted(p for p in logs_dir.iterdir() if p.is_dir()):
        parsed = _parse_slug(exp_dir.name)
        if parsed is None:
            continue
        base, variant, seed = parsed
        if args.base and base != args.base:
            continue
        n_runs += 1
        seeds_seen[variant].add(seed)
        cache: dict[str, dict | None] = {}
        for col, src, path in METRICS:
            if src not in cache:
                cache[src] = _load_json(exp_dir / src)
            val = _dig(cache[src], path)
            if isinstance(val, (int, float)):
                collected[variant][col].append(float(val))

    if not collected:
        print(f"No ablation runs found under {logs_dir} "
              f"(expected dirs named '<base>__<variant>__s<seed>').")
        return

    variants = [v for v in VARIANT_ORDER if v in collected]
    variants += sorted(v for v in collected if v not in VARIANT_ORDER)

    headers = ["variant", "seeds"] + [c for c, _, _ in METRICS]
    rows = []
    for v in variants:
        row = [v, str(len(seeds_seen[v]))]
        for col, _, _ in METRICS:
            row.append(_fmt(*_mean_std(collected[v].get(col, []))))
        rows.append(row)

    # Markdown
    md = ["| " + " | ".join(headers) + " |",
          "|" + "|".join(["---"] * len(headers)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md_text = "\n".join(md)

    print(f"\nAggregated {n_runs} run(s) across {len(variants)} variant(s):\n")
    print(md_text)

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(md_text + "\n", encoding="utf-8")
    csv_lines = [",".join(headers)] + [",".join(c.replace(",", ";") for c in r) for r in rows]
    out.with_suffix(".csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"\nWritten:\n  {out.with_suffix('.md')}\n  {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
