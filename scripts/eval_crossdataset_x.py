"""
scripts/eval_crossdataset_x.py — zero-shot cross-dataset evaluation for TALVAD-X.

The TALVAD-X counterpart of eval_crossdataset.py (which only supports the legacy
BiGRUSoftplusDetector + ResNet/GFC pipeline). Here we:

  1. Load a target pack produced by `extract_talvad_x.py --target_csv ...`
     (data/talvad_x_crossdataset/<name>/packed/), already projected with the SAVED
     PCA bases for feature parity.
  2. Apply the SAVED LAV-DF scaler (checkpoints/.../scaler.joblib) AS-IS — nothing
     is fit on the target (this is what makes the number honest / zero-shot).
  3. Optionally apply per-sample InstanceFeatureNorm (label-free; the de-confounding
     lever for the cross-dataset inversion) via `--set evaluation.instance_norm=true`.
  4. Run TALVADXDetector and report AUC / EER / FPR + confusion at the 0.5 and the
     LAV-DF-tuned thresholds.

Datasets without fake_periods (e.g. FakeAVCeleb) get video-level metrics only;
localization is N/A here (use eval_talvad_x.py on LAV-DF for that).

Usage:
  python scripts/eval_crossdataset_x.py --config configs/lavdf_talvad_x.yaml \
      --target_name fakeavceleb --ckpt_dir checkpoints/talvad_x_x1
  # with the de-confounding instance-norm:
  python scripts/eval_crossdataset_x.py --config configs/lavdf_talvad_x.yaml \
      --target_name fakeavceleb --ckpt_dir checkpoints/talvad_x_x1 --set evaluation.instance_norm=true
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

from models.talvad_x import TALVADXDetector  # noqa: E402
from utils.config import load_config  # noqa: E402


def _apply_overrides(cfg: dict, sets: list) -> None:
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


def _eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def tent_adapt(model, feats: np.ndarray, audio_dim: int, device: str,
               steps: int = 3, lr: float = 1e-3, bs: int = 64) -> None:
    """Test-time adaptation (Tent): minimize the model's prediction entropy on the
    UNLABELED target by updating only the LayerNorm affine params (the model is
    LayerNorm-only, so this is the BN-free analogue of Tent). Label-free → honest
    zero-shot. In place; dropout stays disabled (eval mode) for stability."""
    import torch.nn as nn
    ln_params = []
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.LayerNorm) and m.elementwise_affine:
            m.weight.requires_grad_(True); m.bias.requires_grad_(True)
            ln_params += [m.weight, m.bias]
    if not ln_params:
        print("  [tta] no LayerNorm affine params found — skipping"); return
    opt = torch.optim.SGD(ln_params, lr=lr, momentum=0.9)
    x_all = torch.from_numpy(feats).float()
    model.eval()  # LayerNorm has no running stats; eval() only disables dropout
    # cuDNN's fused RNN backward refuses to run in eval mode; disable it for the few
    # TTA steps so the BiGRU's eval-mode backward uses the generic (dropout-free) path.
    cudnn_was = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        for _ in range(int(steps)):
            perm = torch.randperm(len(x_all))
            for i in range(0, len(x_all), bs):
                idx = perm[i:i + bs]
                x = x_all[idx].to(device)
                with torch.enable_grad():
                    logits, _, _ = model(x[..., :audio_dim], x[..., audio_dim:])
                    p = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
                    ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).mean()
                    opt.zero_grad(); ent.backward(); opt.step()
    finally:
        torch.backends.cudnn.enabled = cudnn_was
    print(f"  [tta] Tent: {steps} steps, lr={lr}, adapted {len(ln_params)//2} LayerNorm layers")


def run(config_path: str, target_name: str, ckpt_dir_override: str | None,
        out_dir: str | None, real_only: bool, sets: list | None) -> dict:
    cfg = load_config(config_path)
    _apply_overrides(cfg, sets or [])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pack = REPO / "data" / "talvad_x_crossdataset" / target_name / "packed"
    if not (pack / "features.npy").exists():
        raise FileNotFoundError(
            f"No target pack at {pack}. Run:\n"
            f"  python scripts/extract_talvad_x.py --config {config_path} "
            f"--target_csv <csv> --target_name {target_name}")
    manifest = json.loads((pack / "manifest.json").read_text())
    feats = np.load(pack / "features.npy").astype(np.float32)
    labels = np.load(pack / "labels.npy").astype(np.int64)
    ids = json.loads((pack / "ids.json").read_text())
    audio_dim = int(manifest["audio_dim"])
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())

    ckpt_dir = Path(ckpt_dir_override or cfg["training"]["checkpoint_dir"])
    print(f"\n{'='*64}\n  TALVAD-X cross-dataset — model='{ckpt_dir.name}'  target='{target_name}'")
    print(f"  Device: {device}  |  zero-shot (no target refit)")
    print(f"  Target clips: {len(labels):,}  (real={n_neg:,}  fake={n_pos:,})\n{'='*64}")

    # ── Apply SAVED scaler (no refit) ───────────────────────────────────────
    from joblib import load as jload
    scaler_path = ckpt_dir / "scaler.joblib"
    if scaler_path.exists():
        scaler = jload(scaler_path)
        feats = scaler.transform(feats.reshape(-1, feats.shape[-1])).reshape(feats.shape).astype(np.float32)
        print(f"  Applied saved scaler: {scaler_path}")
    else:
        print(f"  WARNING: no scaler at {scaler_path} — features NOT normalised!")

    # ── Optional per-sample instance norm (de-confounding lever) ────────────
    if cfg.get("evaluation", {}).get("instance_norm", False):
        from models.domain_adv import InstanceFeatureNorm
        feats = InstanceFeatureNorm()(torch.from_numpy(feats)).numpy().astype(np.float32)
        print("  [instance_norm=true] applied per-sample standardization")

    # ── Load model ──────────────────────────────────────────────────────────
    model = TALVADXDetector(cfg["model"]).to(device)
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt)
    model.eval()
    tuned_thr = float(ckpt.get("threshold", 0.5)) if isinstance(ckpt, dict) else 0.5

    # ── Optional test-time adaptation (Tent entropy-min on LayerNorm affine) ──
    if cfg.get("evaluation", {}).get("tta", False):
        tent_adapt(model, feats, audio_dim, device,
                   steps=int(cfg["evaluation"].get("tta_steps", 3)),
                   lr=float(cfg["evaluation"].get("tta_lr", 1e-3)),
                   bs=int(cfg.get("inference", {}).get("batch_size", 64)))

    # ── Inference ────────────────────────────────────────────────────────────
    probs = []
    bs = int(cfg.get("inference", {}).get("batch_size", 64))
    with torch.no_grad():
        for i in range(0, len(feats), bs):
            x = torch.from_numpy(feats[i:i + bs]).float().to(device)
            logits, _, _ = model(x[..., :audio_dim], x[..., audio_dim:])
            probs.extend(torch.sigmoid(logits).squeeze(-1).cpu().numpy().tolist())
    y_score = np.asarray(probs, dtype=np.float32)
    y_true = labels.astype(int)

    # ── Metrics ────────────────────────────────────────────────────────────
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

    def at(thr: float) -> dict:
        pred = (y_score >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum()); fp = int(((pred == 1) & (y_true == 0)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum()); fn = int(((pred == 0) & (y_true == 1)).sum())
        d = {"threshold": round(thr, 4), "fpr": round(fp / max(fp + tn, 1), 4),
             "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}
        if not real_only and len(np.unique(y_true)) > 1:
            d.update(f1=round(float(f1_score(y_true, pred, zero_division=0)), 4),
                     balanced_accuracy=round(float(balanced_accuracy_score(y_true, pred)), 4),
                     accuracy=round(float(accuracy_score(y_true, pred)), 4))
        return d

    report = {"model": ckpt_dir.name, "target": target_name, "n": int(len(y_true)),
              "n_real": n_neg, "n_fake": n_pos, "zero_shot": True,
              "instance_norm": bool(cfg.get("evaluation", {}).get("instance_norm", False)),
              "tta": bool(cfg.get("evaluation", {}).get("tta", False)),
              "tuned_threshold": round(tuned_thr, 4), "at_0.5": at(0.5), "at_tuned": at(tuned_thr)}
    if not real_only and len(np.unique(y_true)) > 1:
        report["AUC"] = round(float(roc_auc_score(y_true, y_score)), 4)
        report["EER"] = round(_eer(y_true, y_score), 4)
    else:
        report["note"] = "real-only set: AUC/EER undefined; report FPR (lower is better)."

    print(f"\n  AUC : {report.get('AUC', 'n/a (real-only)')}   EER : {report.get('EER', 'n/a')}")
    print(f"  @0.50  : f1={report['at_0.5'].get('f1')} bal_acc={report['at_0.5'].get('balanced_accuracy')} fpr={report['at_0.5']['fpr']}")
    print(f"  @tuned : f1={report['at_tuned'].get('f1')} bal_acc={report['at_tuned'].get('balanced_accuracy')} fpr={report['at_tuned']['fpr']}  (LAV-DF threshold={tuned_thr:.3f})")
    cm = report["at_0.5"]["confusion_matrix"]
    print(f"\n  Confusion @0.5   (rows=actual, cols=predicted)")
    print(f"                 pred:REAL   pred:FAKE")
    print(f"    actual:REAL   {cm['tn']:>8}   {cm['fp']:>9}")
    print(f"    actual:FAKE   {cm['fn']:>8}   {cm['tp']:>9}")

    out = Path(out_dir) if out_dir else (REPO / "logs" / f"crossdataset_x_{target_name}")
    out.mkdir(parents=True, exist_ok=True)
    tag = ("_instnorm" if report["instance_norm"] else "") + ("_tta" if report["tta"] else "")
    (out / f"results{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    import pandas as pd
    pd.DataFrame({"id": ids, "label": y_true, "score": np.round(y_score, 6),
                  "pred@0.5": (y_score >= 0.5).astype(int),
                  "pred@tuned": (y_score >= tuned_thr).astype(int)}).to_csv(
        out / f"predictions{tag}.csv", index=False)
    print(f"\n  Report -> {out / f'results{tag}.json'}\n{'='*64}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Zero-shot cross-dataset eval for TALVAD-X")
    ap.add_argument("--config", required=True)
    ap.add_argument("--target_name", required=True, help="target pack name (matches extract_talvad_x --target_name)")
    ap.add_argument("--ckpt_dir", default=None, help="override checkpoint dir (e.g. checkpoints/talvad_x_x1)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--real_only", action="store_true", help="target has no fakes -> report FPR only")
    ap.add_argument("--set", action="append", default=[], help="dotted override key=value (e.g. evaluation.instance_norm=true)")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    run(args.config, args.target_name, args.ckpt_dir, args.out_dir, args.real_only, args.set)
