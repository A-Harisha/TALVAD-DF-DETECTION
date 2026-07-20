"""
scripts/eval_crossdataset.py

Zero-shot cross-dataset evaluation. Train on LAV-DF, test on a *different*
audio-visual dataset (e.g. FakeAVCeleb) with NO refitting on the target.

Discipline (this is what makes the result honest):
  - Features for the target are extracted with the SAME pipeline as training
    (MFCC+Δ+ΔΔ, frozen ResNet, GFC), at the same fps / sequence length.
  - The saved LAV-DF global scaler + feature mask + decision threshold are
    applied AS-IS. Nothing is fit or tuned on the target.

Reports: AUC, EER, and F1 / balanced-acc / FPR at both the 0.5 threshold and the
LAV-DF-tuned threshold (from the checkpoint). For real-only sets (VoxCeleb) it
reports the false-positive rate only.

CSV contract (--data_csv): one row per clip, columns:
    video_path   (absolute, or relative to --root_dir)
    label        (0/1 or real/fake)   — omit/ignore for --real_only sets
  optional: modify_audio, modify_video  (1/0; unused unless present)

Usage:
  # sanity check — run on LAV-DF itself, should reproduce ~0.956 AUC
  python scripts/eval_crossdataset.py --config configs/lavdf_talvad_v8gen.yaml \
      --data_csv data/lavdf_compact_metadata.csv --root_dir ./datasets/LAV-DF-REDUCED-COMPACT/LAV-DF \
      --name lavdf_selfcheck --limit 2000

  # real cross-dataset (FakeAVCeleb)
  python scripts/eval_crossdataset.py --config configs/lavdf_talvad_v8gen.yaml \
      --data_csv data/fakeavceleb_metadata.csv --root_dir /path/to/FakeAVCeleb --name fakeavceleb

  # real-only false-positive probe (VoxCeleb)
  python scripts/eval_crossdataset.py --config configs/lavdf_talvad_v8gen.yaml \
      --data_csv data/voxceleb_metadata.csv --root_dir /path/to/VoxCeleb --name voxceleb --real_only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from features.audio_features import audio_feature_dim
from features.fusion import FeatureFusionNormalizer
from features.video_cnn import _load_resnet18_feature_extractor
from models.bigru_detector import BiGRUSoftplusDetector
# Reuse the EXACT training-time extraction routine — critical for feature parity.
# The generic prepare_sequences fallback applies median-blur/greyscale that
# corrupts GFC (a high-frequency signal) and shifts ResNet features, silently
# tanking zero-shot scores (the LAV-DF self-check caught this: 0.86 vs 0.956).
from scripts.extract_all import _extract_one
from utils.config import load_config


def _load_target_df(data_csv: str, root_dir: str | None) -> pd.DataFrame:
    df = pd.read_csv(data_csv)
    # normalise column names
    if "video_path" not in df.columns:
        for alt in ["path", "filepath", "file", "video"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "video_path"}); break
    if "video_path" not in df.columns:
        raise ValueError(f"{data_csv} must contain a 'video_path' column")
    if "label" not in df.columns:
        for alt in ["target", "is_fake", "class"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "label"}); break
    if "label" not in df.columns:
        df["label"] = 0  # real-only sets may omit it
    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().map({"real": 0, "fake": 1}).fillna(df["label"]).astype(int)

    df["video_path"] = df["video_path"].astype(str)
    if root_dir:
        df["video_path"] = df["video_path"].apply(
            lambda p: str(Path(root_dir) / p) if not Path(p).is_absolute() else p
        )
    exists = df["video_path"].map(lambda p: Path(p).exists())
    missing = int((~exists).sum())
    if missing:
        print(f"  WARNING: {missing} rows reference missing files and are dropped.")
    df = df.loc[exists].reset_index(drop=True)
    # Unique per-clip id. Datasets like FakeAVCeleb reuse filenames across
    # identity folders (e.g. 00028.mp4 appears 31×), so the bare stem collides:
    # _extract_one would skip later clips as "cached" and many rows would load
    # ONE shared feature file. Hash the full path to guarantee uniqueness while
    # keeping a readable stem prefix.
    def _uid(p: str) -> str:
        h = hashlib.md5(str(p).encode("utf-8")).hexdigest()[:12]
        return f"{Path(p).stem}_{h}"
    df["id"] = df["video_path"].map(_uid)
    assert df["id"].nunique() == len(df), "clip ids are not unique after hashing"
    return df


def _eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def run(config_path: str, data_csv: str, root_dir: str | None, name: str,
        real_only: bool, out_dir: str | None, limit: int) -> dict:
    cfg = load_config(config_path)
    _g = cfg.get("features", {}).get("gfc", {})
    cfg.setdefault("model", {}).setdefault(
        "gfc_bands", int(_g.get("n_bands", 0)) if _g.get("enabled", False) else 0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = cfg["training"].get("sequence_length", 24)
    if root_dir:
        cfg["dataset"]["root_dir"] = root_dir

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    print(f"\n{'='*64}\n  Cross-dataset eval — model='{cfg['experiment_name']}'  target='{name}'")
    print(f"  Device: {device}  |  zero-shot (no target refit)\n{'='*64}")

    df = _load_target_df(data_csv, root_dir)
    if limit and limit > 0 and limit < len(df):
        df = df.sample(n=limit, random_state=cfg.get("seed", 42)).reset_index(drop=True)
    n_pos = int((df["label"] == 1).sum()); n_neg = int((df["label"] == 0).sum())
    print(f"  Target clips: {len(df):,}  (real={n_neg:,}  fake={n_pos:,})")

    # ── Extract features with the EXACT training routine (extract_all._extract_one) ──
    import os as _os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    feat_dir = REPO_ROOT / "data" / "crossdataset" / name / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting features → {feat_dir}  (reused on re-run)")
    # Pre-warm the ResNet once (worker threads share the cached GPU model).
    _load_resnet18_feature_extractor(cfg["features"].get("cnn", {}).get("pretrained", True)).to(device)

    rows = [(str(r["id"]), str(r["video_path"]), int(r["label"])) for _, r in df.iterrows()]
    n_workers = min(8, _os.cpu_count() or 4)
    n_err = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_extract_one, sid, vp, feat_dir, cfg, seq_len, device) for sid, vp, _ in rows]
        with tqdm(total=len(rows), desc=f"  Extract {name}", unit="clip") as pbar:
            for fut in as_completed(futs):
                if not fut.result()[1]:
                    n_err += 1
                pbar.update(1)
    if n_err:
        print(f"  WARNING: {n_err} clip(s) failed extraction — skipped.")

    seqs_list, labels_list, ids_list = [], [], []
    for sid, _vp, lbl in rows:
        p = feat_dir / f"{sid}_sequence.npy"
        if p.exists():
            seqs_list.append(np.load(p))
            labels_list.append(lbl)
            ids_list.append(sid)
    if not seqs_list:
        raise RuntimeError("No features extracted — check video paths / codecs.")
    seqs = np.stack(seqs_list, axis=0).astype(np.float32)
    labels = np.asarray(labels_list, dtype=np.int64)

    # ── Apply SAVED mask + scaler (no refit) ──────────────────────────────────
    mask_path = ckpt_dir / "feature_mask.npy"
    if mask_path.exists():
        seqs = seqs * np.load(mask_path).astype(np.float32)
    scaler_path = ckpt_dir / "global_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"global scaler not found: {scaler_path}")
    scaler = FeatureFusionNormalizer(); scaler.load(scaler_path)
    seqs = scaler.transform(seqs)

    # ── Load model ─────────────────────────────────────────────────────────────
    audio_dim = audio_feature_dim(cfg["features"]["audio"]["n_mfcc"])
    video_dim = seqs.shape[-1] - audio_dim
    model = BiGRUSoftplusDetector(audio_dim=audio_dim, video_dim=video_dim, cfg=cfg["model"]).to(device)
    state = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    tuned_thr = float(state.get("threshold", 0.5)) if isinstance(state, dict) else 0.5

    # ── Optional instance normalisation (kills domain-shift from global scaler) ──
    # Apply before inference; label-free, no target-set leakage. Standardises each
    # clip independently so the OOD scaler shift doesn't invert rankings.
    inst_norm = cfg.get("evaluation", {}).get("instance_norm", False)
    if inst_norm:
        from models.domain_adv import InstanceFeatureNorm
        ifn = InstanceFeatureNorm()
        seqs = ifn(torch.from_numpy(seqs)).numpy().astype(np.float32)
        print("  [instance_norm=true] applied per-sample normalisation")

    # ── Inference ──────────────────────────────────────────────────────────────
    probs = []
    bs = cfg.get("inference", {}).get("batch_size", 32)
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            x = torch.from_numpy(seqs[i:i + bs]).float().to(device)
            logits, _, _ = model(x[..., :audio_dim], x[..., audio_dim:])
            probs.extend(torch.sigmoid(logits).squeeze(-1).cpu().numpy().tolist())
    y_score = np.asarray(probs, dtype=np.float32)
    y_true = labels.astype(int)

    # ── Metrics ────────────────────────────────────────────────────────────────
    from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, accuracy_score

    def at(thr: float) -> dict:
        pred = (y_score >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum()); fp = int(((pred == 1) & (y_true == 0)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum()); fn = int(((pred == 0) & (y_true == 1)).sum())
        fpr = fp / max(fp + tn, 1)
        # confusion matrix: rows = actual (real, fake), cols = predicted (real, fake)
        d = {"threshold": round(thr, 4), "fpr": round(fpr, 4),
             "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}
        if not real_only and len(np.unique(y_true)) > 1:
            d.update(f1=round(f1_score(y_true, pred, zero_division=0), 4),
                     balanced_accuracy=round(balanced_accuracy_score(y_true, pred), 4),
                     accuracy=round(accuracy_score(y_true, pred), 4))
        return d

    def _print_cm(cm: dict, thr: float) -> None:
        tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
        print(f"\n  Confusion matrix @ thr={thr:.2f}   (rows=actual, cols=predicted)")
        print(f"                 pred:REAL   pred:FAKE")
        print(f"    actual:REAL   {tn:>8}   {fp:>9}")
        print(f"    actual:FAKE   {fn:>8}   {tp:>9}")

    report = {"model": cfg["experiment_name"], "target": name, "n": int(len(y_true)),
              "n_real": n_neg, "n_fake": n_pos, "zero_shot": True,
              "at_0.5": at(0.5), "at_tuned": at(tuned_thr), "tuned_threshold": round(tuned_thr, 4)}
    if not real_only and len(np.unique(y_true)) > 1:
        report["AUC"] = round(float(roc_auc_score(y_true, y_score)), 4)
        report["EER"] = round(_eer(y_true, y_score), 4)
    else:
        report["note"] = "real-only set: AUC/EER undefined; report FPR (lower is better)."

    print(f"\n  AUC : {report.get('AUC', 'n/a (real-only)')}   EER : {report.get('EER', 'n/a')}")
    print(f"  @0.50  : f1={report['at_0.5'].get('f1')} bal_acc={report['at_0.5'].get('balanced_accuracy')} fpr={report['at_0.5']['fpr']}")
    print(f"  @tuned : f1={report['at_tuned'].get('f1')} bal_acc={report['at_tuned'].get('balanced_accuracy')} fpr={report['at_tuned']['fpr']}  (LAV-DF threshold)")
    if not real_only and len(np.unique(y_true)) > 1:
        _print_cm(report["at_0.5"]["confusion_matrix"], 0.5)
        _print_cm(report["at_tuned"]["confusion_matrix"], tuned_thr)

    out = Path(out_dir) if out_dir else (REPO_ROOT / "logs" / f"crossdataset_{name}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "crossdataset_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ── Per-clip predictions (for re-plotting / per-subtype CM later) ───────────
    pred_df = pd.DataFrame({"id": ids_list, "label": y_true,
                            "score": np.round(y_score, 6),
                            "pred@0.5": (y_score >= 0.5).astype(int),
                            "pred@tuned": (y_score >= tuned_thr).astype(int)})
    pred_df.to_csv(out / "predictions.csv", index=False)

    # ── Confusion-matrix figure (best-effort; skipped if matplotlib absent) ─────
    if not real_only and len(np.unique(y_true)) > 1:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            cm = report["at_0.5"]["confusion_matrix"]
            mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
            fig, ax = plt.subplots(figsize=(4.2, 3.8))
            im = ax.imshow(mat, cmap="Blues")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["REAL", "FAKE"]); ax.set_yticklabels(["REAL", "FAKE"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"{name}  (thr=0.5)\nAUC={report.get('AUC')}  EER={report.get('EER')}")
            thresh = mat.max() / 2.0
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                            color="white" if mat[i, j] > thresh else "black", fontsize=14)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(out / "confusion_matrix.png", dpi=150)
            plt.close(fig)
            print(f"  CM figure → {out / 'confusion_matrix.png'}")
        except Exception as exc:  # noqa: BLE001
            print(f"  (CM figure skipped: {exc})")

    print(f"\n  Report → {out / 'crossdataset_results.json'}")
    print(f"  Predictions → {out / 'predictions.csv'}\n{'='*64}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Zero-shot cross-dataset evaluation")
    ap.add_argument("--config", required=True, help="Trained model's config (e.g. configs/lavdf_talvad_v8gen.yaml)")
    ap.add_argument("--data_csv", required=True, help="Target metadata CSV (video_path,label[,modify_audio,modify_video])")
    ap.add_argument("--root_dir", default=None, help="Prefix for relative video_path entries")
    ap.add_argument("--name", required=True, help="Target dataset name (used for cache + report)")
    ap.add_argument("--real_only", action="store_true", help="Target has no fakes → report FPR only")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--limit", type=int, default=0, help="Subsample N clips (0=all) — handy for a quick check")
    args = ap.parse_args()
    run(args.config, args.data_csv, args.root_dir, args.name, args.real_only, args.out_dir, args.limit)
