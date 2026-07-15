"""Pennywise observability dashboard.

Builds a self-contained HTML dashboard from MetricsSources. Two views:
  1. Training dynamics — KL, reward(±std), policy stability (grad-norm), SFT loss.
     Fully populated NOW from the real Phase-2 metrics (replay).
  2. System health — rollout throughput, GPU util. Live-ready PLACEHOLDERS that
     populate when a live source (AWSMetricsSource) is connected.

Alerting (ShopRL approach): KL-blowup -> CRITICAL, reward-stall -> WARNING.

    python -m shoprl.observability.dashboard \
        --rloo results/metrics/rloo50_metrics.jsonl \
        --sft  results/metrics/sft_ground_metrics.jsonl \
        --out  results/dashboard.html
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

from shoprl.observability.datasource import (AWSMetricsSource, MetricsSource,
                                             StaticFileSource)


def _series(rows, x, y):
    return [{"x": r[x], "y": r[y]} for r in rows if x in r and y in r and r[y] is not None]


def build_model(rloo: MetricsSource, sft: MetricsSource,
                health: MetricsSource | None = None, *,
                kl_critical: float = 0.5, reward_target: float = 0.8,
                stall_window: int = 10, stall_eps: float = 0.02) -> dict:
    R = rloo.read()
    S = sft.read()

    charts = []
    if R:
        charts.append({"id": "kl", "title": "KL divergence vs. reference",
                       "subtitle": "RLOO — per-step KL to the frozen SFT reference",
                       "yfmt": ".4f", "threshold": kl_critical,
                       "threshold_label": f"blowup ≥ {kl_critical}",
                       "points": _series(R, "step", "kl_mean")})
        rpts = [{"x": r["step"], "y": r["reward_mean"],
                 "lo": r["reward_mean"] - r.get("reward_std", 0),
                 "hi": r["reward_mean"] + r.get("reward_std", 0)} for r in R]
        charts.append({"id": "reward", "title": "Reward vs. step",
                       "subtitle": "RLOO — mean trajectory reward (±1 std band)",
                       "yfmt": ".3f", "points": rpts, "band": True})
        charts.append({"id": "gradnorm", "title": "Policy stability (grad-norm)",
                       "subtitle": "RLOO — pre-clip gradient norm per step",
                       "yfmt": ".2f", "points": _series(R, "step", "grad_norm")})
    if S:
        charts.append({"id": "loss", "title": "SFT warmup loss",
                       "subtitle": "supervised warmup — loss (agent-token masked)",
                       "yfmt": ".3f", "points": _series(S, "step", "loss")})

    # --- stat tiles (headline) ---
    tiles = []
    if R:
        last = R[-1]
        kls = [r["kl_mean"] for r in R if "kl_mean" in r]
        rwd = [r["reward_mean"] for r in R]
        tiles += [
            {"label": "Final KL", "value": f"{last.get('kl_mean', 0):.4f}", "hint": f"max {max(kls):.4f}"},
            {"label": "Mean reward", "value": f"{sum(rwd)/len(rwd):+.3f}", "hint": f"final {last['reward_mean']:+.3f}"},
            {"label": "Ask-rate", "value": f"{last.get('ask_rate', 0):.2f}", "hint": "clarify before recommend"},
            {"label": "Violation-rate", "value": f"{last.get('violation_rate', 0):.2f}", "hint": "added w/o permission"},
        ]
    if S:
        tiles.append({"label": "SFT loss", "value": f"{S[-1]['loss']:.3f}",
                      "hint": f"from {S[0]['loss']:.2f}"})

    # --- alerts ---
    alerts = []
    if R:
        kls = [r["kl_mean"] for r in R if "kl_mean" in r]
        maxkl = max(kls) if kls else 0.0
        if maxkl > kl_critical:
            alerts.append({"rule": "KL blowup", "level": "critical", "icon": "✖",
                           "msg": f"max KL {maxkl:.4f} exceeded {kl_critical} — policy diverging from reference"})
        else:
            alerts.append({"rule": "KL blowup", "level": "good", "icon": "✓",
                           "msg": f"KL bounded (max {maxkl:.4f} ≪ {kl_critical}) — stable, RLOO-regime"})
        rwd = [r["reward_mean"] for r in R]
        win = rwd[-stall_window:]
        plateaued = (max(win) - min(win)) < stall_eps and (win[-1] - win[0]) <= stall_eps
        mean_win = sum(win) / len(win)
        if plateaued and mean_win < reward_target:
            alerts.append({"rule": "Reward stall", "level": "warning", "icon": "▲",
                           "msg": f"reward plateaued at {mean_win:.3f} < target {reward_target} over last {len(win)} steps — stuck low"})
        elif plateaued:
            alerts.append({"rule": "Reward stall", "level": "good", "icon": "✓",
                           "msg": f"reward plateaued at {mean_win:.3f} ≥ target {reward_target} — healthy (task near-saturated), not stalled-low"})
        else:
            alerts.append({"rule": "Reward stall", "level": "good", "icon": "✓",
                           "msg": f"reward moving (last-window Δ {win[-1]-win[0]:+.3f})"})

    # --- system health (placeholders unless a live source is connected) ---
    health_live = bool(health and health.available())
    health_rows = (health.read() if health_live else [])
    system = {
        "live": health_live,
        "source": (health.name if health else "none"),
        "tiles": [
            {"label": "Rollout throughput", "unit": "traj/sec",
             "value": (f"{health_rows[-1].get('traj_per_s','—')}" if health_live else None)},
            {"label": "GPU utilization", "unit": "%",
             "value": (f"{health_rows[-1].get('gpu_util','—')}" if health_live else None)},
            {"label": "Agent-gens throughput", "unit": "gen/sec",
             "value": (f"{health_rows[-1].get('gens_per_s','—')}" if health_live else None)},
        ],
    }

    return {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "rloo_source": rloo.name, "rloo_steps": len(R),
        "sft_source": sft.name, "sft_steps": len(S),
        "charts": charts, "tiles": tiles, "alerts": alerts, "system": system,
    }


_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pennywise — training & system dashboard</title>
<style>
:root{color-scheme:light dark}
.viz-root{
  --page:#f9f9f7; --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --series-1:#2a78d6;
  --good:#0ca30c; --warning:#fab219; --critical:#d03b3b; --border:rgba(11,11,11,.10);
  --band:rgba(42,120,214,.14);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--text-primary);
  background:var(--page); min-height:100vh; margin:0;
}
:root[data-theme="dark"] .viz-root, @media (prefers-color-scheme:dark){
  :root:where(:not([data-theme="light"])) .viz-root{
    --page:#0d0d0d; --surface-1:#1a1a19; --text-primary:#fff; --text-secondary:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --axis:#383835; --series-1:#3987e5;
    --border:rgba(255,255,255,.10); --band:rgba(57,135,229,.18);
}}
.wrap{max-width:1080px;margin:0 auto;padding:24px}
header{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap}
h1{font-size:20px;margin:0}
.meta{color:var(--muted);font-size:12px}
.tabs{display:flex;gap:8px;margin:16px 0 8px}
.tab{padding:6px 12px;border:1px solid var(--border);border-radius:8px;background:var(--surface-1);
  color:var(--text-secondary);cursor:pointer;font-size:13px}
.tab[aria-selected="true"]{color:var(--text-primary);border-color:var(--series-1);font-weight:600}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:12px 0}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.tile .l{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:26px;margin-top:4px}
.tile .h{color:var(--text-secondary);font-size:12px;margin-top:2px}
.tile.await .v{color:var(--muted)}
.alerts{display:flex;flex-direction:column;gap:8px;margin:12px 0}
.alert{display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:10px;
  border:1px solid var(--border);background:var(--surface-1);font-size:13px}
.alert .badge{display:inline-flex;align-items:center;gap:6px;font-weight:700;padding:2px 8px;border-radius:6px}
.alert.good .badge{color:var(--good)} .alert.warning .badge{color:var(--warning)}
.alert.critical .badge{color:var(--critical)}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:14px 14px 6px;margin:12px 0}
.card h3{margin:0;font-size:14px} .card .sub{color:var(--muted);font-size:12px;margin:2px 0 6px}
svg{width:100%;height:200px;display:block}
.axis text{fill:var(--muted);font-size:10px} .grid line{stroke:var(--grid);stroke-width:1}
.await-note{color:var(--muted);font-size:12px;padding:8px 0}
.tog{cursor:pointer;border:1px solid var(--border);background:var(--surface-1);color:var(--text-secondary);
  border-radius:8px;padding:4px 10px;font-size:12px}
.hidden{display:none}
.tt{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--border);
  border-radius:6px;padding:4px 8px;font-size:11px;color:var(--text-primary);opacity:0;transition:opacity .08s}
</style></head>
<body data-palette="#2a78d6"><div class="viz-root"><div class="wrap">
<header>
  <div><h1>Pennywise — training &amp; system dashboard</h1>
  <div class="meta" id="meta"></div></div>
  <button class="tog" id="tog">Toggle theme</button>
</header>
<div class="tabs">
  <div class="tab" id="tab-train" aria-selected="true" onclick="showView('train')">Training dynamics</div>
  <div class="tab" id="tab-sys" aria-selected="false" onclick="showView('sys')">System health</div>
</div>
<div id="view-train">
  <div class="alerts" id="alerts"></div>
  <div class="tiles" id="tiles"></div>
  <div id="charts"></div>
</div>
<div id="view-sys" class="hidden">
  <div class="await-note" id="sysnote"></div>
  <div class="tiles" id="systiles"></div>
</div>
</div></div>
<div class="tt" id="tt"></div>
<script id="model" type="application/json">__MODEL__</script>
<script>
const M=JSON.parse(document.getElementById('model').textContent);
document.getElementById('meta').textContent=
  `RLOO ${M.rloo_steps} steps (${M.rloo_source}) · SFT ${M.sft_steps} steps (${M.sft_source}) · generated ${M.generated}`;
const cs=getComputedStyle(document.querySelector('.viz-root'));
function col(n){return cs.getPropertyValue(n).trim()}
// alerts
document.getElementById('alerts').innerHTML=M.alerts.map(a=>
  `<div class="alert ${a.level}"><span class="badge">${a.icon} ${a.level.toUpperCase()}</span>
   <b>${a.rule}</b> — ${a.msg}</div>`).join('');
// tiles
document.getElementById('tiles').innerHTML=M.tiles.map(t=>
  `<div class="tile"><div class="l">${t.label}</div><div class="v">${t.value}</div><div class="h">${t.hint||''}</div></div>`).join('');
// system tiles (placeholders)
document.getElementById('sysnote').textContent = M.system.live
  ? `Live system metrics from ${M.system.source}.`
  : `System-health panels are live-ready placeholders — awaiting live data (connect AWSMetricsSource: A10G rollout throughput + GPU util from Steps 2–3).`;
document.getElementById('systiles').innerHTML=M.system.tiles.map(t=>
  `<div class="tile ${M.system.live?'':'await'}"><div class="l">${t.label} (${t.unit})</div>
   <div class="v">${M.system.live? t.value : 'awaiting'}</div>
   <div class="h">${M.system.live?'':'populates when live source connected'}</div></div>`).join('');
// charts
const W=980,H=200,PL=48,PR=14,PT=12,PB=26;
function chart(c){
  const pts=c.points; if(!pts.length) return '';
  const xs=pts.map(p=>p.x), ys=pts.map(p=>p.y);
  let ymin=Math.min(...ys), ymax=Math.max(...ys);
  if(c.band){ymin=Math.min(ymin,...pts.map(p=>p.lo)); ymax=Math.max(ymax,...pts.map(p=>p.hi));}
  if(c.threshold!=null) ymax=Math.max(ymax,c.threshold);
  const xmin=Math.min(...xs), xmax=Math.max(...xs);
  const pad=(ymax-ymin)*0.08||1; ymin-=pad; ymax+=pad;
  const X=x=>PL+(xmax===xmin?0:(x-xmin)/(xmax-xmin))*(W-PL-PR);
  const Y=y=>PT+(1-(y-ymin)/(ymax-ymin))*(H-PT-PB);
  const line=pts.map((p,i)=>`${i?'L':'M'}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join(' ');
  // gridlines (recessive) + y ticks
  let grid='',ticks='';
  for(let i=0;i<=4;i++){const yy=ymin+(ymax-ymin)*i/4; const py=Y(yy);
    grid+=`<line x1="${PL}" y1="${py.toFixed(1)}" x2="${W-PR}" y2="${py.toFixed(1)}"/>`;
    ticks+=`<text x="${PL-6}" y="${(py+3).toFixed(1)}" text-anchor="end">${fmt(yy,c.yfmt)}</text>`;}
  // x ticks (first,last)
  ticks+=`<text x="${X(xmin)}" y="${H-8}" text-anchor="start">step ${xmin}</text>`+
         `<text x="${W-PR}" y="${H-8}" text-anchor="end">step ${xmax}</text>`;
  let band=''; if(c.band){const up=pts.map((p,i)=>`${i?'L':'M'}${X(p.x).toFixed(1)},${Y(p.hi).toFixed(1)}`).join(' ');
    const dn=pts.slice().reverse().map(p=>`L${X(p.x).toFixed(1)},${Y(p.lo).toFixed(1)}`).join(' ');
    band=`<path d="${up} ${dn} Z" fill="var(--band)" stroke="none"/>`;}
  let thr=''; if(c.threshold!=null){const ty=Y(c.threshold);
    thr=`<line x1="${PL}" y1="${ty.toFixed(1)}" x2="${W-PR}" y2="${ty.toFixed(1)}" stroke="var(--critical)" stroke-width="1.5" stroke-dasharray="4 4"/>
         <text x="${W-PR}" y="${(ty-4).toFixed(1)}" text-anchor="end" fill="var(--critical)">${c.threshold_label||''}</text>`;}
  const dots=pts.map(p=>`<circle cx="${X(p.x).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="0" data-x="${p.x}" data-y="${p.y}"/>`).join('');
  return `<div class="card"><h3>${c.title}</h3><div class="sub">${c.subtitle}</div>
    <svg viewBox="0 0 ${W} ${H}" data-yfmt="${c.yfmt}" onmousemove="hov(event,this)" onmouseleave="document.getElementById('tt').style.opacity=0">
    <g class="grid">${grid}${thr}</g><g class="axis">${ticks}</g>
    ${band}<path d="${line}" fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round"/>
    <g class="pts">${dots}</g><line class="cross hidden" stroke="var(--axis)" stroke-width="1"/></svg></div>`;
}
function fmt(v,f){if(f==='.4f')return v.toFixed(4); if(f==='.3f')return v.toFixed(3); if(f==='.2f')return v.toFixed(2); return v.toFixed(2);}
function hov(e,svg){const r=svg.getBoundingClientRect(); const pts=[...svg.querySelectorAll('.pts circle')];
  if(!pts.length)return; const sx=W/r.width;
  const mx=(e.clientX-r.left)*sx; let best=null,bd=1e9;
  pts.forEach(c=>{const cx=+c.getAttribute('cx'); const d=Math.abs(cx-mx); if(d<bd){bd=d;best=c;}});
  if(!best)return; const cx=+best.getAttribute('cx'),cy=+best.getAttribute('cy');
  const cr=svg.querySelector('.cross'); cr.classList.remove('hidden');
  cr.setAttribute('x1',cx);cr.setAttribute('x2',cx);cr.setAttribute('y1',PT);cr.setAttribute('y2',H-PB);
  const tt=document.getElementById('tt'); tt.style.opacity=1;
  tt.style.left=(e.clientX+12)+'px'; tt.style.top=(e.clientY-10)+'px';
  tt.textContent=`step ${best.getAttribute('data-x')} · ${fmt(+best.getAttribute('data-y'),svg.getAttribute('data-yfmt'))}`;}
document.getElementById('charts').innerHTML=M.charts.map(chart).join('');
function showView(v){const t=v==='train';
  document.getElementById('view-train').classList.toggle('hidden',!t);
  document.getElementById('view-sys').classList.toggle('hidden',t);
  document.getElementById('tab-train').setAttribute('aria-selected',t);
  document.getElementById('tab-sys').setAttribute('aria-selected',!t);}
document.getElementById('tog').onclick=()=>{const r=document.documentElement;
  const cur=r.getAttribute('data-theme'); r.setAttribute('data-theme', cur==='dark'?'light':'dark');
  document.getElementById('charts').innerHTML=M.charts.map(chart).join('');};
</script></body></html>"""


def render_html(model: dict) -> str:
    return _HTML.replace("__MODEL__", json.dumps(model))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rloo", required=True, help="RLOO metrics.jsonl")
    ap.add_argument("--sft", required=True, help="SFT metrics.jsonl")
    ap.add_argument("--out", default="results/dashboard.html")
    ap.add_argument("--kl-critical", type=float, default=0.5)
    ap.add_argument("--reward-target", type=float, default=0.8)
    args = ap.parse_args()
    model = build_model(StaticFileSource(args.rloo, name="replay"),
                        StaticFileSource(args.sft, name="replay"),
                        AWSMetricsSource(),  # pre-wired, unavailable -> placeholders
                        kl_critical=args.kl_critical, reward_target=args.reward_target)
    with open(args.out, "w") as f:
        f.write(render_html(model))
    print(f"[dashboard] {args.out} | RLOO {model['rloo_steps']} steps, SFT {model['sft_steps']} steps")
    print(f"[dashboard] alerts: " + "; ".join(f"{a['rule']}={a['level']}" for a in model["alerts"]))


if __name__ == "__main__":
    main()
