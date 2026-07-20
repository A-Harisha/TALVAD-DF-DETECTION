"""
Multi-experiment results dashboard.
Run from project root:  python scripts/gen_dashboard.py
Output:                 dashboard.html
"""
from __future__ import annotations

import base64
import json
import pathlib
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
LOGS_DIR = pathlib.Path("logs")
OUT_FILE  = pathlib.Path("dashboard.html")

EXPERIMENT_ORDER = [
    "gbwso_av_lavdf_compact_smoke",
    "gbwso_av_lavdf_compact_subset",
    "gbwso_av_lavdf_compact_balanced",
    "gbwso_av_lavdf_full",
    "gbwso_av_lavdf_full_tune",
    "gbwso_av_lavdf_improved_v2",
    "arch_fix_cross_attention",
    "talvad_gbwso_hybrid",
    "talvad_v2",
    "talvad_v3",
    "talvad_v4",
    "talvad_v5",
]

DISPLAY_NAMES: dict[str, str] = {
    "gbwso_av_lavdf_compact_smoke":    "GBWSO Smoke",
    "gbwso_av_lavdf_compact_subset":   "GBWSO Subset",
    "gbwso_av_lavdf_compact_balanced": "GBWSO Balanced",
    "gbwso_av_lavdf_full":             "GBWSO Full",
    "gbwso_av_lavdf_full_tune":        "GBWSO Full+Tune",
    "gbwso_av_lavdf_improved_v2":      "GBWSO Improved v2",
    "arch_fix_cross_attention":        "Cross-Attn Fix",
    "talvad_gbwso_hybrid":             "TALVAD+GBWSO",
    "talvad_v2":                       "TALVAD v2",
    "talvad_v3":                       "TALVAD v3",
    "talvad_v4":                       "TALVAD v4",
    "talvad_v5":                       "TALVAD v5",
}

SMOKE_SET = {"gbwso_av_lavdf_compact_smoke", "gbwso_av_lavdf_compact_subset"}

# Colors assigned in EXPERIMENT_ORDER order
PALETTE = [
    "#555566", "#777788",          # smoke / subset  (muted)
    "#ef9a9a", "#ff8a65",          # GBWSO Balanced, Full
    "#ffb74d", "#ffcc02",          # GBWSO Full+Tune, Improved v2
    "#ce93d8", "#80cbc4",          # Cross-Attn Fix, TALVAD+GBWSO
    "#90caf9", "#4fc3f7",          # v2, v3
    "#81c784", "#a5ffb0",          # v4, v5  (greenest = best)
]


# ── Data loading ──────────────────────────────────────────────────────────────

def _downsample(arr: list, n: int = 200) -> list:
    if len(arr) <= n:
        return [round(float(x), 4) for x in arr]
    idx = np.round(np.linspace(0, len(arr) - 1, n)).astype(int)
    return [round(float(arr[i]), 4) for i in idx]


def load_experiment(name: str) -> dict | None:
    d = LOGS_DIR / name
    sp = d / "training_status.json"
    tp = d / "validation_threshold.json"
    if not sp.exists():
        return None

    status = json.loads(sp.read_text(encoding="utf-8"))
    thresh: dict = {}
    if tp.exists():
        thresh = json.loads(tp.read_text(encoding="utf-8"))

    fpr = _downsample(thresh.get("fpr", [0, 1]))
    tpr = _downsample(thresh.get("tpr", [0, 1]))

    curves_b64 = ""
    cp = d / "training_curves.png"
    if cp.exists():
        curves_b64 = base64.b64encode(cp.read_bytes()).decode("ascii")

    return {
        "name":         name,
        "display_name": DISPLAY_NAMES.get(name, name),
        "is_smoke":     name in SMOKE_SET,
        "status":       status,
        "metrics": {
            "accuracy":          round(thresh.get("accuracy",          0), 4),
            "balanced_accuracy": round(thresh.get("balanced_accuracy", 0), 4),
            "precision":         round(thresh.get("precision",         0), 4),
            "recall":            round(thresh.get("recall",            0), 4),
            "f1":                round(thresh.get("f1",                0), 4),
            "auc":               round(thresh.get("auc",               0), 4),
        },
        "confusion_matrix": thresh.get("confusion_matrix", [[0, 0], [0, 0]]),
        "threshold":        round(thresh.get("threshold", 0.5), 4),
        "fpr":              fpr,
        "tpr":              tpr,
        "curves_b64":       curves_b64,
    }


# ── HTML building ─────────────────────────────────────────────────────────────

def _status_badge(s: str) -> str:
    cls = {"complete": "badge-ok", "stopped_early": "badge-warn",
           "training": "badge-run"}.get(s, "badge-other")
    label = s.replace("_", " ").title()
    return f'<span class="badge {cls}">{label}</span>'


def build_table_rows(experiments: list) -> str:
    rows = ""
    sorted_exp = sorted(experiments, key=lambda e: e["metrics"]["auc"], reverse=True)
    full_exps = [e for e in sorted_exp if not e["is_smoke"]]
    best_name = full_exps[0]["name"] if full_exps else ""
    for rank, exp in enumerate(sorted_exp, 1):
        m = exp["metrics"]
        s = exp["status"]
        cls = "smoke-row" if exp["is_smoke"] else ""
        if exp["name"] == best_name:
            cls += " best-row"
        rows += f"""
  <tr class="{cls.strip()}" onclick="scrollToCard('{exp['name']}')">
    <td style="color:#666">{rank}</td>
    <td style="color:{exp['color']};font-weight:700">{exp['display_name']}</td>
    <td class="td-auc">{m['auc']:.4f}</td>
    <td>{m['balanced_accuracy']:.4f}</td>
    <td>{m['f1']:.4f}</td>
    <td>{m['precision']:.4f}</td>
    <td>{m['recall']:.4f}</td>
    <td>{s.get('best_epoch','?')} / {s.get('total_epochs','?')}</td>
    <td>{_status_badge(s.get('status',''))}</td>
    <td style="color:#555;font-size:11px">{s.get('timestamp','')[:10]}</td>
  </tr>"""
    return rows


def build_cards(experiments: list) -> str:
    cards = ""
    for exp in experiments:
        m = exp["metrics"]
        s = exp["status"]
        cm = exp["confusion_matrix"]
        tn, fp_v = cm[0][0], cm[0][1]
        fn_v, tp_v = cm[1][0], cm[1][1]
        img_html = (
            f'<img src="data:image/png;base64,{exp["curves_b64"]}" '
            f'class="curves-img" alt="training curves" />'
            if exp["curves_b64"]
            else '<p class="no-data">No training curves image</p>'
        )
        cards += f"""
<div class="exp-card" id="card-{exp['name']}">
  <div class="card-hdr" style="border-left:3px solid {exp['color']}">
    <span style="color:{exp['color']};font-weight:700;font-size:15px">{exp['display_name']}</span>
    <span class="card-auc-badge">AUC {m['auc']:.4f}</span>
    {_status_badge(s.get('status',''))}
  </div>
  <div class="card-body">
    <div class="card-stats">
      <div class="stat-row"><span class="sk">Accuracy</span><span class="sv">{m['accuracy']:.4f}</span></div>
      <div class="stat-row"><span class="sk">Balanced Acc</span><span class="sv">{m['balanced_accuracy']:.4f}</span></div>
      <div class="stat-row"><span class="sk">F1</span><span class="sv">{m['f1']:.4f}</span></div>
      <div class="stat-row"><span class="sk">Precision</span><span class="sv">{m['precision']:.4f}</span></div>
      <div class="stat-row"><span class="sk">Recall</span><span class="sv">{m['recall']:.4f}</span></div>
      <div class="stat-row"><span class="sk">Threshold</span><span class="sv">{exp['threshold']:.3f}</span></div>
      <div class="stat-row"><span class="sk">Best Epoch</span><span class="sv">{s.get('best_epoch','?')} / {s.get('total_epochs','?')}</span></div>
      <div class="stat-row"><span class="sk">Train Loss</span><span class="sv">{s.get('last_train_loss',0):.4f}</span></div>
      <div class="stat-row"><span class="sk">Val Loss</span><span class="sv">{s.get('last_val_loss',0):.4f}</span></div>
      <div class="stat-row"><span class="sk">Last AUC</span><span class="sv">{s.get('last_auc',0):.4f}</span></div>
    </div>
    <div class="card-cm">
      <div class="cm-title">Confusion Matrix</div>
      <table class="cm-tbl">
        <tr><th></th><th>Pred Real</th><th>Pred Fake</th></tr>
        <tr><th>True Real</th><td class="cm-tn">{tn:,}</td><td class="cm-fp">{fp_v:,}</td></tr>
        <tr><th>True Fake</th><td class="cm-fn">{fn_v:,}</td><td class="cm-tp">{tp_v:,}</td></tr>
      </table>
    </div>
    <div class="card-curves">{img_html}</div>
  </div>
</div>"""
    return cards


def build_html(experiments: list) -> str:
    full = [e for e in experiments if not e["is_smoke"]]
    best = max(full, key=lambda e: e["metrics"]["auc"]) if full else None
    training_now = [e for e in full if e["status"].get("status") == "training"]

    best_auc    = f"{best['metrics']['auc']:.4f}" if best else "—"
    best_name   = best["display_name"] if best else "—"
    n_training  = len(training_now)
    first_auc   = full[0]["metrics"]["auc"] if full else 0
    last_auc    = best["metrics"]["auc"] if best else 0
    delta_str   = f"+{last_auc - first_auc:.4f}"

    table_rows = build_table_rows(experiments)
    cards_html = build_cards(experiments)

    # Chart datasets — bar chart (AUC per experiment)
    bar_labels  = json.dumps([e["display_name"] for e in experiments])
    bar_aucs    = json.dumps([e["metrics"]["auc"] for e in experiments])
    bar_balacc  = json.dumps([e["metrics"]["balanced_accuracy"] for e in experiments])
    bar_f1      = json.dumps([e["metrics"]["f1"] for e in experiments])
    bar_colors  = json.dumps([e["color"] for e in experiments])

    # ROC datasets (full exps only, skip smoke for clarity)
    roc_datasets_js = "[\n"
    for exp in experiments:
        if exp["is_smoke"]:
            continue
        xy_data = json.dumps([{"x": x, "y": y} for x, y in zip(exp["fpr"], exp["tpr"])])
        label   = json.dumps(exp["display_name"])
        color   = json.dumps(exp["color"])
        roc_datasets_js += (
            f"  {{\n"
            f"    label: {label},\n"
            f"    data: {xy_data},\n"
            f"    borderColor: {color},\n"
            f"    borderWidth: 2,\n"
            f"    pointRadius: 0,\n"
            f"    fill: false,\n"
            f"    tension: 0.1,\n"
            f"  }},\n"
        )
    roc_datasets_js += "]"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>TALVAD — Experiment Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {{
  --bg:     #0f1117;
  --bg2:    #1a1d27;
  --bg3:    #21253a;
  --border: #2e3148;
  --text:   #e0e0e0;
  --muted:  #7a7a9a;
  --accent: #4fc3f7;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: var(--bg); color: var(--text); font-family: "Consolas","Courier New",monospace; font-size: 13px; }}
a {{ color: var(--accent); }}

/* Header */
.hdr {{ background: var(--bg2); border-bottom: 1px solid var(--border); padding: 18px 32px; }}
.hdr h1 {{ font-size: 20px; color: var(--accent); letter-spacing: 1px; }}
.hdr p  {{ color: var(--muted); font-size: 11px; margin-top: 4px; }}

/* KPI cards */
.kpi-row {{ display: flex; gap: 14px; padding: 20px 32px; flex-wrap: wrap; }}
.kpi {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
        padding: 14px 20px; min-width: 160px; flex: 1; }}
.kpi .kv {{ font-size: 26px; font-weight: 700; color: var(--accent); }}
.kpi .kl {{ font-size: 10px; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }}

/* Sections */
.section {{ padding: 0 32px 28px; }}
.section-title {{ font-size: 13px; font-weight: 700; color: var(--muted);
                  text-transform: uppercase; letter-spacing: 1px;
                  border-bottom: 1px solid var(--border); padding-bottom: 8px; margin-bottom: 16px; }}

/* Table */
table.exp-table {{ width: 100%; border-collapse: collapse; }}
table.exp-table th {{ color: var(--muted); font-size: 11px; text-transform: uppercase;
                      letter-spacing: 0.5px; padding: 6px 10px; text-align: left;
                      border-bottom: 1px solid var(--border); }}
table.exp-table td {{ padding: 8px 10px; border-bottom: 1px solid #1e2035; vertical-align: middle; }}
table.exp-table tr {{ cursor: pointer; transition: background 0.15s; }}
table.exp-table tr:hover {{ background: var(--bg3); }}
table.exp-table .best-row td {{ background: #14201a; }}
table.exp-table .smoke-row {{ opacity: 0.45; }}
.td-auc {{ font-weight: 700; color: #a5ffb0; }}

/* Badges */
.badge {{ display: inline-block; font-size: 10px; padding: 2px 8px; border-radius: 20px;
          font-weight: 700; letter-spacing: 0.5px; }}
.badge-ok   {{ background: #1a3320; color: #81c784; }}
.badge-warn {{ background: #302010; color: #ffb74d; }}
.badge-run  {{ background: #102030; color: #4fc3f7; }}
.badge-other{{ background: #252535; color: #aaa; }}

/* Charts row */
.charts-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 0 32px 28px; }}
.chart-box  {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }}
.chart-box h3 {{ color: var(--muted); font-size: 11px; text-transform: uppercase;
                 letter-spacing: 1px; margin-bottom: 14px; }}

/* ROC full-width */
.roc-box {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
            padding: 20px; margin: 0 32px 28px; }}
.roc-box h3 {{ color: var(--muted); font-size: 11px; text-transform: uppercase;
               letter-spacing: 1px; margin-bottom: 14px; }}

/* Experiment cards */
.cards-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(700px, 1fr));
               gap: 20px; padding: 0 32px 40px; }}
.exp-card {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
.card-hdr {{ padding: 12px 16px; display: flex; align-items: center; gap: 12px;
             background: #1e2135; flex-wrap: wrap; }}
.card-auc-badge {{ background: #1a3320; color: #81c784; font-size: 11px; font-weight: 700;
                   padding: 2px 10px; border-radius: 20px; }}
.card-body {{ display: grid; grid-template-columns: 200px 200px 1fr; gap: 0; }}
.card-stats {{ padding: 14px; border-right: 1px solid var(--border); }}
.stat-row {{ display: flex; justify-content: space-between; padding: 3px 0;
             border-bottom: 1px solid #1e2035; }}
.sk {{ color: var(--muted); font-size: 11px; }}
.sv {{ color: var(--text); font-size: 11px; font-weight: 600; }}
.card-cm {{ padding: 14px; border-right: 1px solid var(--border); }}
.cm-title {{ color: var(--muted); font-size: 10px; text-transform: uppercase;
             letter-spacing: 1px; margin-bottom: 10px; }}
.cm-tbl {{ border-collapse: collapse; font-size: 11px; width: 100%; }}
.cm-tbl th {{ color: var(--muted); padding: 4px 6px; font-weight: normal; font-size: 10px; }}
.cm-tbl td {{ padding: 6px 8px; text-align: center; font-weight: 700; border-radius: 4px; }}
.cm-tn  {{ background: #1a3320; color: #81c784; }}
.cm-fp  {{ background: #3a1a1a; color: #ef9a9a; }}
.cm-fn  {{ background: #3a2a1a; color: #ffb74d; }}
.cm-tp  {{ background: #0e2040; color: #4fc3f7; }}
.card-curves {{ padding: 14px; overflow: hidden; }}
.curves-img  {{ width: 100%; border-radius: 4px; display: block; }}
.no-data {{ color: var(--muted); font-size: 11px; padding: 20px; text-align: center; }}

/* Toggle */
.toggle-row {{ padding: 8px 32px 0; display: flex; gap: 10px; align-items: center; }}
.toggle-btn {{ background: var(--bg2); border: 1px solid var(--border); color: var(--muted);
               padding: 4px 14px; border-radius: 20px; cursor: pointer; font-size: 11px;
               font-family: inherit; }}
.toggle-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
</style>
</head>
<body>

<div class="hdr">
  <h1>TALVAD — Experiment Dashboard</h1>
  <p>LAV-DF deepfake detection &nbsp;·&nbsp; {len(experiments)} experiments tracked</p>
</div>

<!-- KPI row -->
<div class="kpi-row">
  <div class="kpi"><div class="kv">{best_auc}</div><div class="kl">Best AUC ({best_name})</div></div>
  <div class="kpi"><div class="kv">{len(full)}</div><div class="kl">Full experiments</div></div>
  <div class="kpi"><div class="kv" style="color:#4fc3f7">{n_training}</div><div class="kl">Currently training</div></div>
  <div class="kpi"><div class="kv" style="color:#a5ffb0">{delta_str}</div><div class="kl">AUC gain (first → best)</div></div>
  <div class="kpi"><div class="kv" style="color:#ffb74d">{f"{best['metrics']['f1']:.4f}" if best else '—'}</div><div class="kl">Best F1</div></div>
  <div class="kpi"><div class="kv" style="color:#ce93d8">{f"{best['metrics']['balanced_accuracy']:.4f}" if best else '—'}</div><div class="kl">Best Balanced Acc</div></div>
</div>

<!-- Leaderboard -->
<div class="toggle-row">
  <button class="toggle-btn" onclick="toggleSmoke()">Toggle smoke tests</button>
  <span style="color:var(--muted);font-size:11px">Click any row to jump to its card &nbsp;·&nbsp; Smoke tests are dimmed</span>
</div>
<div class="section" style="padding-top:14px">
  <div class="section-title">Leaderboard (sorted by AUC at best checkpoint)</div>
  <table class="exp-table">
    <thead>
      <tr>
        <th>#</th><th>Experiment</th><th>AUC ★</th><th>Bal.Acc</th>
        <th>F1</th><th>Precision</th><th>Recall</th>
        <th>Best / Total Epochs</th><th>Status</th><th>Date</th>
      </tr>
    </thead>
    <tbody id="leaderboard-body">
      {table_rows}
    </tbody>
  </table>
</div>

<!-- Charts -->
<div class="charts-row">
  <div class="chart-box">
    <h3>AUC Comparison (all experiments, sorted by AUC)</h3>
    <canvas id="barAuc" height="260"></canvas>
  </div>
  <div class="chart-box">
    <h3>Multi-metric comparison (AUC / Balanced Acc / F1)</h3>
    <canvas id="barMetrics" height="260"></canvas>
  </div>
</div>

<!-- ROC overlay -->
<div class="roc-box">
  <h3>ROC Curves overlay (full experiments)</h3>
  <canvas id="rocChart" height="120"></canvas>
</div>

<!-- Per-experiment cards -->
<div class="section">
  <div class="section-title">Per-Experiment Details</div>
</div>
<div class="cards-grid">
  {cards_html}
</div>

<script>
const DARK = "#0f1117";
const GRID = "#252535";
const MUTED = "#7a7a9a";
Chart.defaults.color = MUTED;
Chart.defaults.borderColor = GRID;

// ── AUC bar chart ─────────────────────────────────────────────────────────────
const labels  = {bar_labels};
const aucs    = {bar_aucs};
const balacc  = {bar_balacc};
const f1s     = {bar_f1};
const colors  = {bar_colors};

// Sort by AUC descending for the bar chart
const sortIdx = [...aucs.keys()].sort((a,b) => aucs[b]-aucs[a]);
const sLabels = sortIdx.map(i=>labels[i]);
const sAucs   = sortIdx.map(i=>aucs[i]);
const sColors = sortIdx.map(i=>colors[i]);

new Chart(document.getElementById("barAuc"), {{
  type: "bar",
  data: {{
    labels: sLabels,
    datasets: [{{
      label: "AUC",
      data: sAucs,
      backgroundColor: sColors.map(c => c + "cc"),
      borderColor: sColors,
      borderWidth: 1,
    }}]
  }},
  options: {{
    indexAxis: "y",
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ min: 0, max: 1, grid: {{ color: GRID }}, ticks: {{ color: MUTED }} }},
      y: {{ grid: {{ display: false }}, ticks: {{ color: MUTED, font: {{ size: 11 }} }} }},
    }},
    animation: false,
  }}
}});

// ── Multi-metric grouped bar ──────────────────────────────────────────────────
new Chart(document.getElementById("barMetrics"), {{
  type: "bar",
  data: {{
    labels: labels,
    datasets: [
      {{ label: "AUC",          data: aucs,   backgroundColor: "#4fc3f799", borderColor: "#4fc3f7", borderWidth:1 }},
      {{ label: "Balanced Acc", data: balacc, backgroundColor: "#ce93d899", borderColor: "#ce93d8", borderWidth:1 }},
      {{ label: "F1",           data: f1s,    backgroundColor: "#81c78499", borderColor: "#81c784", borderWidth:1 }},
    ]
  }},
  options: {{
    plugins: {{ legend: {{ labels: {{ color: MUTED, font:{{size:11}} }} }} }},
    scales: {{
      x: {{ grid:{{display:false}}, ticks:{{color:MUTED, font:{{size:10}}, maxRotation:30}} }},
      y: {{ min:0, max:1, grid:{{color:GRID}}, ticks:{{color:MUTED}} }},
    }},
    animation: false,
  }}
}});

// ── ROC curves ────────────────────────────────────────────────────────────────
const rocDatasets = {roc_datasets_js};

// Add diagonal reference line
rocDatasets.push({{
  label: "Random",
  data: [{{x:0,y:0}},{{x:1,y:1}}],
  borderColor: "#444455",
  borderWidth: 1,
  borderDash: [4,4],
  pointRadius: 0,
  fill: false,
}});

new Chart(document.getElementById("rocChart"), {{
  type: "line",
  data: {{ datasets: rocDatasets }},
  options: {{
    parsing: false,
    plugins: {{ legend: {{ labels: {{ color: MUTED, font:{{size:11}}, boxWidth:16 }} }} }},
    scales: {{
      x: {{ type:"linear", min:0, max:1, title:{{display:true,text:"False Positive Rate",color:MUTED}},
            grid:{{color:GRID}}, ticks:{{color:MUTED}} }},
      y: {{ min:0, max:1, title:{{display:true,text:"True Positive Rate",color:MUTED}},
            grid:{{color:GRID}}, ticks:{{color:MUTED}} }},
    }},
    animation: false,
    elements: {{ line:{{tension:0.1}} }},
  }}
}});

// ── Interactions ──────────────────────────────────────────────────────────────
function scrollToCard(name) {{
  const el = document.getElementById("card-" + name);
  if (el) el.scrollIntoView({{ behavior: "smooth", block: "start" }});
}}

let smokeVisible = true;
function toggleSmoke() {{
  smokeVisible = !smokeVisible;
  document.querySelectorAll(".smoke-row").forEach(r => {{
    r.style.display = smokeVisible ? "" : "none";
  }});
  document.querySelectorAll(".exp-card").forEach(c => {{
    if (c.id.includes("compact_smoke") || c.id.includes("compact_subset")) {{
      c.style.display = smokeVisible ? "" : "none";
    }}
  }});
}}
</script>
</body>
</html>"""


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    experiments = []
    for i, name in enumerate(EXPERIMENT_ORDER):
        exp = load_experiment(name)
        if exp is not None:
            exp["color"] = PALETTE[i % len(PALETTE)]
            experiments.append(exp)

    if not experiments:
        print("No experiment logs found. Run from project root.")
        return

    html = build_html(experiments)
    OUT_FILE.write_text(html, encoding="utf-8")
    size_kb = OUT_FILE.stat().st_size // 1024
    print(f"Dashboard saved: {OUT_FILE.resolve()}  ({size_kb} KB)")
    print(f"  Experiments loaded: {len(experiments)}")
    for e in experiments:
        print(f"  {e['display_name']:22s}  AUC {e['metrics']['auc']:.4f}  {e['status'].get('status','?')}")


if __name__ == "__main__":
    main()
