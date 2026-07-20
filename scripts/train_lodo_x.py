"""
scripts/train_lodo_x.py — combined FakeAVCeleb feature-extraction check +
Leave-One-Dataset-Out (LODO) training for TALVAD-X.

What it does
------------
1. ENSURE EXTRACTION (idempotent): for every requested dataset, make sure a
   TALVAD-X packed/ feature cache exists. FakeAVCeleb is normally already
   extracted (data/talvad_x_crossdataset/fakeavceleb/packed); if a pack is
   missing it shells out to scripts/extract_talvad_x.py in target mode using the
   SAVED LAV-DF PCA bases (so every domain lives in the same feature space).
2. LODO TRAINING: for the chosen hold-out domain (or every fold with
   --holdout all), train the detector on the *other* domains and evaluate on the
   held-out one. AUC / EER / balanced-accuracy are reported per fold.

Honest protocol (no test leakage)
---------------------------------
The held-out domain is used ONLY for the final evaluation. Early stopping and
threshold tuning use a validation split carved from the SOURCE domains
(stratified by domain×label). This is the methodological point that keeps the
LODO number an honest cross-dataset generalization estimate.

Packs are memory-mapped and sub-sampled per domain (--train_cap / --test_cap) so
the giant LAV-DF pack (136k clips) does not have to be fully materialized in RAM.

Examples
--------
  # 2-domain LODO over LAV-DF + FakeAVCeleb, both folds, quick caps:
  .venv/Scripts/python.exe scripts/train_lodo_x.py \
      --config configs/lavdf_talvad_x.yaml \
      --datasets lavdf,fakeavceleb --holdout all \
      --train_cap 8000 --epochs 25 --tag erm

  # smoke (tiny, 2 epochs):
  .venv/Scripts/python.exe scripts/train_lodo_x.py \
      --config configs/lavdf_talvad_x.yaml --datasets lavdf,fakeavceleb \
      --holdout fakeavceleb --train_cap 400 --test_cap 400 --epochs 2 --tag smoke

  # turn on domain-adversarial training (DANN) instead of plain ERM:
  ... --set training.dann_weight=0.5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from datasets.multisource import DOMAIN_IDS, MultiSourceDataset  # noqa: E402
from evaluation.metrics import compute_binary_metrics, find_best_threshold  # noqa: E402
from models.talvad_x import TALVADXDetector  # noqa: E402
from training.trainer import Trainer  # noqa: E402
from utils.config import load_config  # noqa: E402

# our short keys -> (pack dir, MultiSourceDataset domain name, default extract CSV)
DATASETS = {
    "lavdf": {
        "pack": REPO / "data" / "talvad_x" / "fps12_sl48" / "packed",
        "domain": "LAV-DF",
        "csv": None,  # produced by the main LAV-DF extraction, never auto-extracted here
    },
    "fakeavceleb": {
        "pack": REPO / "data" / "talvad_x_crossdataset" / "fakeavceleb" / "packed",
        "domain": "FakeAVCeleb",
        "csv": REPO / "data" / "fakeavceleb_metadata.csv",
    },
    "deepfaketimit": {
        "pack": REPO / "data" / "talvad_x_crossdataset" / "deepfaketimit" / "packed",
        "domain": "DeepfakeTIMIT",
        "csv": REPO / "data" / "deepfaketimit_metadata.csv",
    },
}


def _force_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass


def _eer(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """Equal Error Rate from the ROC curve (needs both classes present)."""
    from sklearn.metrics import roc_curve
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def _apply_overrides(cfg: dict, sets: list[str]) -> None:
    for kv in sets or []:
        key, val = kv.split("=", 1)
        v: object = val
        if val.lower() in ("true", "false"):
            v = val.lower() == "true"
        else:
            try:
                v = int(val) if val.lstrip("-").isdigit() else float(val)
            except ValueError:
                v = val
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v


def ensure_extracted(key: str, config_path: str) -> None:
    """Make sure DATASETS[key] has a packed feature cache; extract if missing."""
    info = DATASETS[key]
    if (info["pack"] / "features.npy").exists():
        man = json.loads((info["pack"] / "manifest.json").read_text(encoding="utf-8"))
        print(f"  [{key}] pack present: n={man.get('n')} dim={man.get('feature_dim')} — skip extraction")
        return
    if key == "lavdf":
        raise FileNotFoundError(
            "LAV-DF pack missing — run the main extraction "
            "(scripts/extract_talvad_x.py without --target_csv) first.")
    csv = info["csv"]
    if csv is None or not Path(csv).exists():
        raise FileNotFoundError(f"[{key}] no pack and no CSV at {csv} to extract from.")
    print(f"  [{key}] pack missing — extracting from {csv} (saved LAV-DF PCA bases)…")
    cmd = [sys.executable, str(REPO / "scripts" / "extract_talvad_x.py"),
           "--config", config_path, "--target_csv", str(csv), "--target_name", key]
    rc = subprocess.run(cmd)
    if rc.returncode != 0 or not (info["pack"] / "features.npy").exists():
        raise RuntimeError(f"[{key}] extraction failed (exit {rc.returncode}).")


def load_pack(key: str, cap: int, seed: int):
    """Memory-map a pack and return a sub-sampled (features fp32, labels) tuple."""
    info = DATASETS[key]
    feats = np.load(info["pack"] / "features.npy", mmap_mode="r")   # [N,T,D] fp16
    labels = np.load(info["pack"] / "labels.npy")
    n = len(labels)
    idx = np.arange(n)
    if cap and cap > 0 and n > cap:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.permutation(n)[:cap])
    x = np.asarray(feats[idx], dtype=np.float32)   # materialize only the subset
    y = np.asarray(labels[idx], dtype=np.int64)
    return x, y


def train_and_eval(sources: list[str], tests: list[str], cfg: dict, args, device: str) -> list[dict]:
    """Train once on `sources`, then evaluate the SAME model on EACH domain in
    `tests` (held-out targets). Returns one result dict per test target."""
    seq_len = int(cfg["training"]["sequence_length"])
    seed = int(args.seed)
    print("\n" + "═" * 64)
    print(f"  TRAIN on: {', '.join(sources)}   TEST on: {', '.join(tests)}")
    print("═" * 64)

    # ── load source domains (train pool) ───────────────────────────────────
    src_feats, src_labels, src_domains = [], [], []
    for k in sources:
        x, y = load_pack(k, args.train_cap, seed)
        src_feats.append(x); src_labels.append(y)
        src_domains.append(np.full(len(y), DOMAIN_IDS[DATASETS[k]["domain"]], dtype=np.int64))
        print(f"   source {k:13s}: n={len(y):6d}  real={(y==0).sum():6d} fake={(y==1).sum():6d}")
    Xs = np.concatenate(src_feats, 0)
    Ys = np.concatenate(src_labels, 0)
    Ds = np.concatenate(src_domains, 0)

    # ── carve an honest val split from the SOURCE domains (strat by dom×lbl) ─
    rng = np.random.default_rng(seed + 7)
    val_frac = float(args.val_frac)
    is_val = np.zeros(len(Ys), dtype=bool)
    for d in np.unique(Ds):
        for c in (0, 1):
            grp = np.where((Ds == d) & (Ys == c))[0]
            if len(grp) == 0:
                continue
            k_val = max(1, int(round(len(grp) * val_frac)))
            is_val[rng.permutation(grp)[:k_val]] = True
    tr_mask = ~is_val

    # ── standardize: fit on SOURCE-TRAIN only, apply everywhere ────────────
    from sklearn.preprocessing import StandardScaler
    D = Xs.shape[-1]
    scaler = StandardScaler().fit(Xs[tr_mask].reshape(-1, D))

    def apply(x):
        return scaler.transform(x.reshape(-1, D)).reshape(x.shape).astype(np.float32)
    Xs = apply(Xs)

    # ── checkpoint dir (named by the training source set) ──────────────────
    tag = f"_{args.tag}" if args.tag else ""
    name = f"src-{'+'.join(sources)}{tag}"
    ckpt = REPO / "checkpoints" / name
    ckpt.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(cfg))  # deep copy
    cfg["experiment_name"] = name
    cfg["training"]["checkpoint_dir"] = str(ckpt)
    from joblib import dump
    dump(scaler, ckpt / "scaler.joblib")

    # ── train datasets / loaders ───────────────────────────────────────────
    train_srcs = []
    for k in sources:
        dom = DATASETS[k]["domain"]
        m = tr_mask & (Ds == DOMAIN_IDS[dom])
        train_srcs.append({"name": dom, "features": Xs[m], "labels": Ys[m]})
    train_ds = MultiSourceDataset(train_srcs, sequence_length=seq_len)
    val_ds = TensorDataset(torch.from_numpy(Xs[is_val]).float(),
                           torch.from_numpy(Ys[is_val]).long())

    bs = int(cfg["training"]["batch_size"])
    sampler = train_ds.make_balanced_sampler()
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    print(f"   train={len(train_ds)}  val(source)={len(val_ds)}  sampler=per-domain-balanced")

    # ── model + trainer ────────────────────────────────────────────────────
    man = json.loads((DATASETS[sources[0]]["pack"] / "manifest.json").read_text(encoding="utf-8"))
    audio_dim = int(man["audio_dim"])
    video_dim = int(man["appearance_dim"] + man["spectral_dim"])
    if float(cfg["training"].get("dann_weight", 0.0)) > 0.0:
        cfg["training"]["n_domains"] = max(DOMAIN_IDS.values()) + 1

    model = TALVADXDetector(cfg["model"])
    if not model.use_cmis or not cfg["model"]["cmis"].get("contrastive", True):
        cfg["training"]["contrastive_loss_weight"] = 0.0
    if not model.use_cmis:
        cfg["training"]["cmis_incon_loss_weight"] = 0.0

    trainer = Trainer(model, cfg, audio_dim=audio_dim, video_dim=video_dim, device=device)
    state = trainer.fit(train_loader, val_loader)

    # ── evaluate the trained model on EACH held-out test target ────────────
    results = []
    for tkey in tests:
        Xt, Yt = load_pack(tkey, args.test_cap, seed + 1)
        Xt = apply(Xt)
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(Xt).float(), torch.from_numpy(Yt).long()),
            batch_size=bs, shuffle=False)
        yt, pt = trainer.predict_probabilities(test_loader)
        m = compute_binary_metrics(yt, pt, threshold=state.best_threshold)
        auc = float(m.get("auc", 0.0)); bal = float(m.get("balanced_accuracy", 0.0))
        eer = _eer(yt, pt)
        print("─" * 64)
        print(f"  [train {'+'.join(sources)} → test {tkey}]  AUC={auc:.4f}  bal_acc={bal:.4f}  "
              f"EER={'%.4f' % eer if eer is not None else 'n/a'}  "
              f"(n={len(Yt)} real={(Yt==0).sum()} fake={(Yt==1).sum()}; "
              f"src-val thr={state.best_threshold:.3f}, best_ep={state.best_epoch+1})")
        print("─" * 64)
        results.append({"train_on": sources, "test_on": tkey, "auc": auc, "eer": eer,
                        "balanced_accuracy": bal, "threshold": float(state.best_threshold),
                        "best_epoch": int(state.best_epoch + 1),
                        "n_train": len(train_ds), "n_test": int(len(Yt))})
    return results


def main() -> None:
    _force_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--datasets", default="lavdf,fakeavceleb",
                    help="LODO mode: comma list of domain keys (lavdf,fakeavceleb,deepfaketimit)")
    ap.add_argument("--holdout", default="all",
                    help="LODO mode: 'all' (every fold) or a single domain key to hold out")
    ap.add_argument("--train_on", default=None,
                    help="explicit mode: comma list of SOURCE domain keys to train on")
    ap.add_argument("--test_on", default=None,
                    help="explicit mode: comma list of TARGET domain keys to test on (1+)")
    ap.add_argument("--train_cap", type=int, default=8000, help="max clips per SOURCE domain (0=all)")
    ap.add_argument("--test_cap", type=int, default=0, help="max clips for the HELD-OUT domain (0=all)")
    ap.add_argument("--val_frac", type=float, default=0.12, help="source val carve fraction")
    ap.add_argument("--epochs", type=int, default=0, help="override training.epochs (0=use config)")
    ap.add_argument("--tag", default="lodo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--set", action="append", default=[], help="dotted override key=value (repeatable)")
    ap.add_argument("--skip_extract", action="store_true", help="don't auto-extract missing packs")
    args = ap.parse_args()

    def _parse(s):
        return [k.strip() for k in s.split(",") if k.strip()]

    # Build the list of (sources, tests) runs for either interface.
    # Explicit mode: --train_on A,B --test_on C,D  -> one training run, eval on C and D.
    # LODO mode:     --datasets ... --holdout all  -> one run per held-out domain.
    if args.train_on or args.test_on:
        if not (args.train_on and args.test_on):
            sys.exit("explicit mode needs BOTH --train_on and --test_on.")
        sources, tests = _parse(args.train_on), _parse(args.test_on)
        if set(sources) & set(tests):
            sys.exit("a domain cannot be in both --train_on and --test_on (would leak).")
        plan = [(sources, tests)]
        all_keys = sorted(set(sources) | set(tests))
    else:
        keys = _parse(args.datasets)
        if len(keys) < 2:
            sys.exit("LODO mode needs >=2 --datasets.")
        folds = keys if args.holdout == "all" else _parse(args.holdout)
        for f in folds:
            if f not in keys:
                sys.exit(f"holdout '{f}' must be one of --datasets {keys}")
        plan = [([k for k in keys if k != f], [f]) for f in folds]
        all_keys = keys

    for k in all_keys:
        if k not in DATASETS:
            sys.exit(f"unknown dataset key '{k}' (valid: {list(DATASETS)})")

    cfg = load_config(args.config)
    _apply_overrides(cfg, args.set)
    if args.epochs > 0:
        cfg["training"]["epochs"] = args.epochs
    # LODO-sane defaults: cross-dataset packs carry no real frame labels, so the
    # frame-localization / CMIS-inconsistency losses (which need them) are off.
    cfg["training"].setdefault("localization_loss_weight", 0.0)
    cfg["training"]["localization_loss_weight"] = 0.0
    cfg["training"]["cmis_incon_loss_weight"] = 0.0
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("═" * 64)
    print("  TALVAD-X — cross-dataset / LODO training")
    print(f"  runs: {[(s, t) for s, t in plan]}   device: {device}")
    print("═" * 64)

    # ── phase 1: ensure every involved domain is extracted ─────────────────
    if not args.skip_extract:
        print("── ensuring feature packs ──")
        for k in all_keys:
            ensure_extracted(k, args.config)

    # ── phase 2: run each (sources -> tests) training+eval ─────────────────
    results = []
    for sources, tests in plan:
        results.extend(train_and_eval(sources, tests, cfg, args, device))

    # ── summary + persist ──────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print("  SUMMARY")
    print("═" * 64)
    for r in results:
        eer = f"{r['eer']:.4f}" if r["eer"] is not None else "n/a"
        print(f"   train {'+'.join(r['train_on']):24s} → test {r['test_on']:13s}  "
              f"AUC={r['auc']:.4f}  bal_acc={r['balanced_accuracy']:.4f}  EER={eer}  "
              f"(train n={r['n_train']}, test n={r['n_test']})")
    if len(results) > 1:
        print(f"   mean AUC over {len(results)} evals = "
              f"{np.mean([r['auc'] for r in results]):.4f}")
    out = REPO / "logs" / f"talvad_x_lodo_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": args.config, "plan": plan,
                               "train_cap": args.train_cap, "results": results},
                              indent=2), encoding="utf-8")
    print(f"\n  results -> {out}")


if __name__ == "__main__":
    main()
