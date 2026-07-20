"""
scripts/gate_crossdataset.py — the make-or-break CMIS-2 cross-dataset GATE.

For each ablation variant x seed it loads the preserved checkpoint
(checkpoints/talvad_x_<variant>_s<seed>/), zero-shot-scores it on each target
pack (data/talvad_x_crossdataset/<target>/packed/), and aggregates:

  * per-variant per-target AUC  (mean +/- std across seeds)
  * the DECISION CONTRAST  delta = AUC(best of {A3,A4}) - AUC(max of {A0,A1})
    with a paired clip-level bootstrap CI (variants share the same target clips,
    so we resample clip indices and recompute both AUCs on each resample).

Pre-registered rule (PROJECT_REVIEW sec 8): CMIS-2 coupling "earns its place" iff the
coupling variants beat the strongest non-coupling baseline cross-dataset by a margin
whose bootstrap CI excludes 0, on at least one real target (FakeAVCeleb / DeepfakeTIMIT).

Zero-shot is honest: each checkpoint uses ITS OWN saved LAV-DF scaler, nothing is
fit on the target. Per-clip scores are averaged across seeds (soft-vote) for a stable
paired bootstrap; per-seed AUCs are reported separately for seed variability.

Usage:
  python scripts/gate_crossdataset.py --config configs/lavdf_talvad_x.yaml \
      --targets fakeavceleb deepfaketimit --seeds 42 1337 2024
  # include the in-domain parity check + the instance-norm regime:
  python scripts/gate_crossdataset.py --config configs/lavdf_talvad_x.yaml \
      --targets fakeavceleb deepfaketimit lavdf_selfcheck --instance_norm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from models.talvad_x import TALVADXDetector  # noqa: E402
from utils.config import load_config  # noqa: E402
from ablate_talvad_x import VARIANTS  # noqa: E402  (the A0..A5 override table)

BASELINE_VARIANTS = ["A0_baseline", "A1_spectral"]   # non-coupling reference
COUPLING_VARIANTS = ["A3_coupling", "A4_full"]        # the CMIS-2 claim


def _apply_overrides(cfg: dict, sets: list[str]) -> None:
    """Apply dotted key=value overrides (same semantics as the train/eval scripts).

    Only model.* overrides change the architecture; applying the rest is harmless
    at inference time, and matters for picking the right cmis sub-modules so the
    checkpoint state_dict loads cleanly per variant."""
    for kv in sets or []:
        key, val = kv.split("=", 1)
        v: object = val
        if val.lower() in ("true", "false"):
            v = val.lower() == "true"
        else:
            try:
                v = int(val) if val.isdigit() else float(val)
            except ValueError:
                v = val
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v


def _load_target(target: str) -> tuple[np.ndarray, np.ndarray, int]:
    pack = REPO / "data" / "talvad_x_crossdataset" / target / "packed"
    if not (pack / "features.npy").exists():
        raise FileNotFoundError(f"no target pack at {pack}")
    manifest = json.loads((pack / "manifest.json").read_text())
    feats = np.load(pack / "features.npy").astype(np.float32)
    labels = np.load(pack / "labels.npy").astype(np.int64)
    return feats, labels, int(manifest["audio_dim"])


def _score(cfg_base: dict, variant: str, seed: int, feats_raw: np.ndarray,
           audio_dim: int, device: str, instance_norm: bool) -> np.ndarray | None:
    """Zero-shot per-clip fake-probability for one checkpoint, or None if missing."""
    ckpt_dir = REPO / "checkpoints" / f"talvad_x_{variant}_s{seed}"
    ckpt_path = ckpt_dir / "best_model.pt"
    scaler_path = ckpt_dir / "scaler.joblib"
    if not ckpt_path.exists():
        return None

    cfg = json.loads(json.dumps(cfg_base))          # deep copy
    _apply_overrides(cfg, VARIANTS[variant])         # variant-correct architecture

    feats = feats_raw
    if scaler_path.exists():
        from joblib import load as jload
        scaler = jload(scaler_path)
        feats = scaler.transform(feats_raw.reshape(-1, feats_raw.shape[-1])).reshape(feats_raw.shape).astype(np.float32)
    if instance_norm:
        from models.domain_adv import InstanceFeatureNorm
        feats = InstanceFeatureNorm()(torch.from_numpy(feats)).numpy().astype(np.float32)

    model = TALVADXDetector(cfg["model"]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt)
    model.eval()

    bs = int(cfg.get("inference", {}).get("batch_size", 64))
    out = []
    with torch.no_grad():
        for i in range(0, len(feats), bs):
            x = torch.from_numpy(feats[i:i + bs]).float().to(device)
            logits, _, _ = model(x[..., :audio_dim], x[..., audio_dim:])
            out.extend(torch.sigmoid(logits).squeeze(-1).cpu().numpy().tolist())
    return np.asarray(out, dtype=np.float32)


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def _boot_ci(y: np.ndarray, s_cpl: np.ndarray, s_base: np.ndarray,
             n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Paired clip-level bootstrap of delta = AUC(coupling) - AUC(baseline)."""
    from sklearn.metrics import roc_auc_score
    n = len(y)
    deltas = np.empty(n_boot, dtype=np.float64)
    got = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():            # need both classes in the resample
            continue
        deltas[got] = roc_auc_score(yb, s_cpl[idx]) - roc_auc_score(yb, s_base[idx])
        got += 1
    deltas = deltas[:got]
    return (float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)),
            float((deltas > 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--targets", nargs="*", default=["fakeavceleb", "deepfaketimit"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1337, 2024])
    ap.add_argument("--variants", nargs="*", default=["A0_baseline", "A1_spectral", "A3_coupling", "A4_full"])
    ap.add_argument("--instance_norm", action="store_true", help="apply per-sample InstanceFeatureNorm")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    cfg_base = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    real_targets = [t for t in args.targets if t not in ("lavdf_selfcheck",)]

    report: dict = {"instance_norm": args.instance_norm, "seeds": args.seeds,
                    "variants": args.variants, "targets": {}, "verdict": {}}
    md = ["# TALVAD-X cross-dataset GATE", "",
          f"instance_norm={args.instance_norm} · seeds={args.seeds} · n_boot={args.n_boot}", ""]

    for target in args.targets:
        try:
            feats_raw, labels, audio_dim = _load_target(target)
        except FileNotFoundError as e:
            print(f"  ! skip {target}: {e}"); continue
        y = labels.astype(int)
        print(f"\n{'='*64}\n  TARGET {target}  (n={len(y)}, real={int((y==0).sum())}, fake={int((y==1).sum())})\n{'='*64}")

        per_variant_auc_mean: dict[str, float] = {}
        mean_scores: dict[str, np.ndarray] = {}
        tgt_rows = {}
        for v in args.variants:
            seed_aucs, seed_scores = [], []
            for s in args.seeds:
                sc = _score(cfg_base, v, s, feats_raw, audio_dim, device, args.instance_norm)
                if sc is None:
                    print(f"    {v} s{s}: [missing checkpoint]"); continue
                a = _auc(y, sc)
                seed_aucs.append(a); seed_scores.append(sc)
                print(f"    {v:14s} s{s}: AUC={a:.4f}")
            if not seed_aucs:
                continue
            aucs = np.array(seed_aucs)
            mean_scores[v] = np.mean(seed_scores, axis=0)          # soft-vote across seeds
            per_variant_auc_mean[v] = float(np.mean([_auc(y, mean_scores[v])]))
            tgt_rows[v] = {"n_seeds": len(seed_aucs),
                           "auc_seed_mean": float(aucs.mean()), "auc_seed_std": float(aucs.std()),
                           "auc_ensemble": per_variant_auc_mean[v]}
            print(f"    -> {v:14s} seed-mean AUC={aucs.mean():.4f} ± {aucs.std():.4f} "
                  f"| ensemble AUC={per_variant_auc_mean[v]:.4f}")

        # ── decision contrast on this target ──────────────────────────────────
        avail_base = [v for v in BASELINE_VARIANTS if v in mean_scores]
        avail_cpl = [v for v in COUPLING_VARIANTS if v in mean_scores]
        contrast = None
        if avail_base and avail_cpl:
            base_v = max(avail_base, key=lambda v: per_variant_auc_mean[v])
            cpl_v = max(avail_cpl, key=lambda v: per_variant_auc_mean[v])
            delta = per_variant_auc_mean[cpl_v] - per_variant_auc_mean[base_v]
            lo, hi, p_pos = _boot_ci(y, mean_scores[cpl_v], mean_scores[base_v], args.n_boot, rng)
            sig = lo > 0
            contrast = {"baseline_best": base_v, "coupling_best": cpl_v, "delta_auc": delta,
                        "ci95": [lo, hi], "p_delta_gt_0": p_pos, "significant_positive": sig}
            print(f"\n  CONTRAST  {cpl_v} − {base_v}: Δ={delta:+.4f}  "
                  f"CI95=[{lo:+.4f}, {hi:+.4f}]  P(Δ>0)={p_pos:.3f}  "
                  f"{'✅ significant' if sig else '✗ not significant'}")

        # ── Phase-1 lift: best P1_* variant vs the plain A0 baseline (the strongest
        # cross-dataset base per the gate). Answers "do the v2 DG/articulatory levers
        # beat doing nothing?" with the same paired bootstrap CI. ──
        p1_vars = [v for v in mean_scores if v.startswith("P1")]
        p1_contrast = None
        if p1_vars and "A0_baseline" in mean_scores:
            p1_best = max(p1_vars, key=lambda v: per_variant_auc_mean[v])
            lo, hi, p_pos = _boot_ci(y, mean_scores[p1_best], mean_scores["A0_baseline"], args.n_boot, rng)
            p1_contrast = {"p1_best": p1_best, "baseline": "A0_baseline",
                           "delta_auc": per_variant_auc_mean[p1_best] - per_variant_auc_mean["A0_baseline"],
                           "ci95": [lo, hi], "p_delta_gt_0": p_pos, "significant_positive": lo > 0}
            print(f"  PHASE-1 LIFT  {p1_best} − A0_baseline: Δ={p1_contrast['delta_auc']:+.4f} "
                  f"CI95=[{lo:+.4f}, {hi:+.4f}]  P(Δ>0)={p_pos:.3f}  "
                  f"{'✅ significant' if lo > 0 else '✗ not significant'}")

        report["targets"][target] = {"n": len(y), "n_real": int((y == 0).sum()),
                                     "n_fake": int((y == 1).sum()),
                                     "variants": tgt_rows, "contrast": contrast,
                                     "p1_contrast": p1_contrast}
        md.append(f"## {target}  (n={len(y)}, real={int((y==0).sum())}, fake={int((y==1).sum())})\n")
        md += ["| variant | seeds | AUC (seed mean±std) | AUC (ensemble) |", "|---|---|---|---|"]
        for v in args.variants:
            if v in tgt_rows:
                r = tgt_rows[v]
                md.append(f"| {v} | {r['n_seeds']} | {r['auc_seed_mean']:.4f} ± {r['auc_seed_std']:.4f} | {r['auc_ensemble']:.4f} |")
        if contrast:
            c = contrast
            md += ["", f"**Contrast** {c['coupling_best']} − {c['baseline_best']}: "
                       f"Δ = {c['delta_auc']:+.4f}, CI95 [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}], "
                       f"P(Δ>0)={c['p_delta_gt_0']:.3f} → "
                       f"{'**significant positive**' if c['significant_positive'] else 'not significant'}", ""]
        if p1_contrast:
            pc = p1_contrast
            md += ["", f"**Phase-1 lift** {pc['p1_best']} − A0_baseline: "
                       f"Δ = {pc['delta_auc']:+.4f}, CI95 [{pc['ci95'][0]:+.4f}, {pc['ci95'][1]:+.4f}], "
                       f"P(Δ>0)={pc['p_delta_gt_0']:.3f} → "
                       f"{'**significant positive**' if pc['significant_positive'] else 'not significant'}", ""]

    # ── overall verdict ──────────────────────────────────────────────────────
    # FakeAVCeleb is the CLEAN primary target. DeepfakeTIMIT has a known real-vs-fake
    # re-encoding confound (VidTIMIT JPEG→h264 reals vs original .avi fakes; see the
    # talvad-x-lodo-results note), so its contrast corroborates but does NOT by itself
    # trigger GO — a DeepfakeTIMIT-only "win" is INCONCLUSIVE until that confound is
    # resolved (re-encode fakes identically, or score-vs-encoding control).
    PRIMARY = "fakeavceleb"
    CONFOUNDED = {"deepfaketimit"}
    sig_targets = [t for t in real_targets
                   if report["targets"].get(t, {}).get("contrast")
                   and report["targets"][t]["contrast"]["significant_positive"]]
    any_contrast = any(report["targets"].get(t, {}).get("contrast") for t in real_targets)
    primary_sig = PRIMARY in sig_targets
    confounded_only = bool(sig_targets) and not primary_sig and all(t in CONFOUNDED for t in sig_targets)
    go = primary_sig

    if not any_contrast:
        # e.g. a Phase-1-only run (P1_* variants) without both an A0/A1 baseline and an
        # A3/A4 coupling variant — report per-variant AUC tables, no GO/NO-GO claim.
        report["verdict"] = {"go": None, "note": "contrast N/A — needs both a non-coupling "
                             "baseline (A0/A1) and a coupling (A3/A4) variant in --variants"}
        print(f"\n{'='*64}\n  GATE VERDICT: N/A (no A0/A1-vs-A3/A4 contrast in this run)\n"
              f"  per-variant cross-dataset AUC reported above.\n{'='*64}")
        md += ["## Verdict", "",
               "**N/A** — this run lacks both a non-coupling baseline (A0/A1) and a coupling "
               "(A3/A4) variant, so no GO/NO-GO contrast was computed. Per-variant AUC tables above.", ""]
        out = REPO / "logs"
        tag = "_instnorm" if args.instance_norm else ""
        (out / f"talvad_x_gate{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out / f"talvad_x_gate{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"\n  -> {out / f'talvad_x_gate{tag}.md'}")
        return
    report["verdict"] = {
        "go": go, "primary_target": PRIMARY, "primary_significant": primary_sig,
        "significant_targets": sig_targets, "confounded_only_significance": confounded_only,
        "rule": ("GO iff CMIS-2 coupling CI95 excludes 0 above the strongest non-coupling "
                 "baseline on the PRIMARY clean target (FakeAVCeleb). DeepfakeTIMIT is "
                 "corroborating-only due to a real-vs-fake re-encoding confound."),
    }
    verdict_txt = "GO ✅" if go else ("INCONCLUSIVE ⚠ (only confounded DeepfakeTIMIT significant)"
                                      if confounded_only else "NO-GO ✗")
    print(f"\n{'='*64}\n  GATE VERDICT: {verdict_txt}\n"
          f"  primary (FakeAVCeleb) significant: {primary_sig} | all significant: {sig_targets or 'none'}\n{'='*64}")
    md += ["## Verdict", "",
           f"**{verdict_txt}** — on the clean primary target FakeAVCeleb, CMIS-2 coupling "
           f"{'significantly beats' if primary_sig else 'does NOT significantly beat'} the strongest "
           f"non-coupling baseline. (DeepfakeTIMIT corroborating-only; significant targets: {sig_targets or 'none'}.)", ""]

    out = REPO / "logs"
    tag = "_instnorm" if args.instance_norm else ""
    (out / f"talvad_x_gate{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / f"talvad_x_gate{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n  -> {out / f'talvad_x_gate{tag}.md'}")


if __name__ == "__main__":
    main()
