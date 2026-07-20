"""
scripts/diagnose_localization_l.py — why is AR capped / AP@0.90 low?

For the trained TALVAD-L model, decode the split and, for every GT fake segment,
find the best-IoU prediction in its clip. Then break the recovery down by:
  * segment DURATION (seconds)   -> is the miss a resolution / short-segment issue?
  * FAKE_TYPE (audio/video/both) -> is the correspondence blind to audio-only fakes?
  * NMS sensitivity              -> recall before vs after Soft-NMS

This decides which fix to spend effort on (resolution / modality coverage / NMS)
BEFORE adding any capacity.

Usage:
  .venv\\Scripts\\python.exe scripts\\diagnose_localization_l.py --config configs/lavdf_talvad_l.yaml \
      --split test --ckpt checkpoints/talvad_l_nogate/best_model.pt
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from utils.loc_ops import iou_1d                                  # noqa: E402
from datasets.annotation_loader import get_modality_labels        # noqa: E402
from eval_localization import split_streams, decode_sample, _build_loader_from_pack  # noqa: E402


def fake_type(sid, ann):
    a, v = get_modality_labels(sid, ann)
    if a and v:
        return "video+audio"
    if a:
        return "audio_only"
    if v:
        return "video_only"
    return "real?"


def _bucket(stats, recovered50, recovered90, best):
    stats["n"] += 1
    stats["r50"] += int(recovered50)
    stats["r90"] += int(recovered90)
    stats["iou_sum"] += best


def _print_table(title, table, order=None):
    print(f"\n  ── {title} ──")
    print(f"   {'group':<14} {'n':>6} {'recall@.5':>10} {'recall@.9':>10} {'mean IoU':>9}")
    keys = order or sorted(table.keys())
    for k in keys:
        s = table.get(k)
        if not s or s["n"] == 0:
            continue
        print(f"   {str(k):<14} {s['n']:>6} {s['r50']/s['n']:>10.3f} "
              f"{s['r90']/s['n']:>10.3f} {s['iou_sum']/s['n']:>9.3f}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    from utils.config import load_config
    from models.talvad_l import TALVADLLocalizer

    cfg = load_config(args.config)
    ann = cfg["dataset"]["annotation_json"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = int(cfg["training"]["sequence_length"])
    dcfg = cfg["localization"]["decode"]
    scaler_path = Path(args.ckpt).parent / "scaler.joblib"
    loader, manifest = _build_loader_from_pack(cfg, args.split, int(cfg["inference"]["batch_size"]), scaler_path)
    ds = loader.dataset

    mcfg = dict(cfg["model"])
    mcfg["dims"] = {"audio": manifest["audio_dim"], "artic": manifest["artic_dim"],
                    "appearance": manifest["appearance_dim"], "spectral": manifest["spectral_dim"]}
    model = TALVADLLocalizer(mcfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    # decode all clips -> preds per video id (also keep pre-NMS for the NMS-sensitivity test)
    preds_by_vid = defaultdict(list)
    raw_by_vid = defaultdict(list)
    for batch in loader:
        x = batch["x"].to(device)
        idxs = batch["idx"].numpy(); durs = batch["duration"].numpy()
        a, ar, apf, sp = split_streams(x, manifest["offsets"])
        out = model(a, ar, apf, sp)
        L = len(out["level_cls"])
        cls_np = [out["level_cls"][l].squeeze(-1).cpu().numpy() for l in range(L)]
        reg_np = [out["level_reg"][l].cpu().numpy() for l in range(L)]
        ctr_np = [out["level_centers"][l].cpu().numpy() for l in range(L)]
        cbm_np = out["cbm_b"].cpu().numpy(); rec_np = out["recon_err"].cpu().numpy()
        for bi in range(x.shape[0]):
            sid = ds.ids[int(idxs[bi])]; dur = float(durs[bi])
            segs = decode_sample([cls_np[l][bi] for l in range(L)], [reg_np[l][bi] for l in range(L)],
                                 ctr_np, cbm_np[bi], rec_np[bi], dur, seq_len, dcfg)
            preds_by_vid[sid] = segs
            # raw (no NMS): same decode but skip soft_nms -> approximate via score_thr only
            from utils import loc_ops
            scores = []
            for l in range(L):
                s = 1.0 / (1.0 + np.exp(-cls_np[l][bi]))
                scores.append(s)
            raw = loc_ops.decode(scores, [reg_np[l][bi] for l in range(L)], ctr_np, dur, seq_len,
                                 float(dcfg.get("score_thr", 0.05)))
            raw_by_vid[sid] = raw

    by_type = defaultdict(lambda: defaultdict(float))
    by_dur = defaultdict(lambda: defaultdict(float))
    by_type_raw = defaultdict(lambda: defaultdict(float))
    overall = defaultdict(float)
    miss_by_type = defaultdict(int)

    dur_bins = [(0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 2.0), (2.0, 1e9)]

    def dur_label(d):
        for lo, hi in dur_bins:
            if lo <= d < hi:
                return f"{lo:g}-{hi:g}s" if hi < 1e9 else f">{lo:g}s"
        return "?"

    for n in range(len(ds)):
        sid = ds.ids[n]
        ft = fake_type(sid, ann)
        preds = preds_by_vid.get(sid, [])
        raws = raw_by_vid.get(sid, [])
        for (gs, ge) in ds.gt_seconds[n]:
            seglen = ge - gs
            best = max((iou_1d(gs, ge, ps, pe) for ps, pe, _ in preds), default=0.0)
            best_raw = max((iou_1d(gs, ge, ps, pe) for ps, pe, _ in raws), default=0.0)
            r50, r90 = best >= 0.5, best >= 0.9
            _bucket(by_type[ft], r50, r90, best)
            _bucket(by_dur[dur_label(seglen)], r50, r90, best)
            _bucket(by_type_raw[ft], best_raw >= 0.5, best_raw >= 0.9, best_raw)
            _bucket(overall, r50, r90, best)
            if not r50:
                miss_by_type[ft] += 1

    n = int(overall["n"])
    print(f"\n  TALVAD-L localization DIAGNOSTIC [{args.split}]  ({n} GT segments)")
    print(f"   overall recall@.5={overall['r50']/n:.3f}  recall@.9={overall['r90']/n:.3f}  "
          f"mean best IoU={overall['iou_sum']/n:.3f}")
    _print_table("by FAKE_TYPE (post-NMS)", by_type, order=["audio_only", "video_only", "video+audio"])
    _print_table("by FAKE_TYPE (pre-NMS — NMS sensitivity)", by_type_raw,
                 order=["audio_only", "video_only", "video+audio"])
    _print_table("by SEGMENT DURATION (post-NMS)", by_dur,
                 order=[f"{lo:g}-{hi:g}s" if hi < 1e9 else f">{lo:g}s" for lo, hi in dur_bins])
    tot_miss = sum(miss_by_type.values())
    print(f"\n  ── where the recall@.5 misses come from ({tot_miss} missed segments) ──")
    for k in ["audio_only", "video_only", "video+audio"]:
        if miss_by_type.get(k):
            print(f"   {k:<14} {miss_by_type[k]:>5}  ({100*miss_by_type[k]/max(tot_miss,1):.0f}% of misses)")


if __name__ == "__main__":
    main()
