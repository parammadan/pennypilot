"""PennyData console — the platform's single-page UI (stdlib-served).

Live feed + stat tiles + per-label bars + tag bars + self-service SQL +
one-click dataset export. Palette = the repo's validated dataviz set
(#2a78d6/#eb6834/#1baf7a categorical, status colors reserved); bars carry
direct labels so identity/value are never color-alone.
"""

CONSOLE_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>PennyData — open data platform</title>
<style>
:root{ --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --border:rgba(11,11,11,.10);
  --c1:#2a78d6; --c2:#eb6834; --c3:#1baf7a;
  --good:#0ca30c; --warning:#fab219; --critical:#d03b3b; }
@media (prefers-color-scheme: dark){ :root{
  --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7;
  --grid:#2c2c2a; --border:rgba(255,255,255,.12);
  --c1:#3987e5; --c2:#d95926; --c3:#199e70; } }
*{box-sizing:border-box} body{margin:0;background:var(--page);color:var(--ink);
  font:14px/1.45 -apple-system,system-ui,sans-serif}
header{padding:14px 22px;border-bottom:1px solid var(--border);display:flex;
  align-items:baseline;gap:12px}
header h1{font-size:17px;margin:0} header .sub{color:var(--muted);font-size:12px}
.live{color:var(--good);font-size:12px}
main{display:grid;grid-template-columns:1.25fr .9fr;gap:16px;padding:16px 22px;
  max-width:1280px}
section{background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:14px 16px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--ink2);margin:0 0 10px}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;grid-column:1/3}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:12px 14px}
.tile .v{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}
.tile .k{font-size:11px;color:var(--muted);text-transform:uppercase;
  letter-spacing:.05em}
.tile .v.good{color:var(--good)} .tile .v.bad{color:var(--critical)}
.bar{display:grid;grid-template-columns:130px 1fr 52px;gap:8px;align-items:center;
  margin:6px 0;font-variant-numeric:tabular-nums}
.bar .name{font-size:12px;color:var(--ink2);overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.bar .track{height:14px;border-radius:4px;background:var(--grid);
  position:relative;overflow:hidden}
.bar .fill{position:absolute;inset:0 auto 0 0;border-radius:4px;min-width:2px}
.bar .val{font-size:12px;text-align:right;color:var(--ink)}
#feed{max-height:340px;overflow:auto;font-size:12px}
#feed .ev{padding:5px 0;border-bottom:1px solid var(--grid);display:flex;gap:8px}
#feed .kind{flex:0 0 108px;color:var(--muted)}
#feed .ev.turn .kind{color:var(--c1)} #feed .ev.feedback .kind{color:var(--c2)}
#feed .ev.episode_end .kind{color:var(--c3)}
#feed .body{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
textarea{width:100%;background:var(--page);color:var(--ink);
  border:1px solid var(--border);border-radius:6px;padding:8px;font:12px monospace}
button{background:var(--c1);border:0;color:#fff;border-radius:6px;
  padding:7px 14px;font-size:13px;cursor:pointer;margin-top:8px}
table{border-collapse:collapse;width:100%;font-size:12px;margin-top:10px}
th,td{padding:4px 8px;border-bottom:1px solid var(--grid);text-align:left;
  font-variant-numeric:tabular-nums}
th{color:var(--muted);font-weight:600}
.note{font-size:11px;color:var(--muted);margin-top:6px}
</style></head><body>
<header><h1>🧮 PennyData</h1>
  <span class="sub">open data platform · every event below arrived from the PennyMart store</span>
  <span class="live" id="live">● live</span></header>
<main>
  <div class="tiles">
    <div class="tile"><div class="v" id="t-ep">–</div><div class="k">episodes ingested</div></div>
    <div class="tile"><div class="v" id="t-viol">–</div><div class="k">permission-violation rate</div></div>
    <div class="tile"><div class="v" id="t-inv">–</div><div class="k">invalid-action rate</div></div>
    <div class="tile"><div class="v" id="t-fb">–</div><div class="k">human feedback 👍 / 👎</div></div>
  </div>
  <section><h2>Live event feed</h2><div id="feed"></div></section>
  <section><h2>Episodes by model</h2><div id="bylabel"></div>
    <h2 style="margin-top:16px">Behavior tags (failure modes are named, not guessed)</h2>
    <div id="tags"></div></section>
  <section style="grid-column:1/3"><h2>Self-service SQL (SELECT-only)</h2>
    <textarea id="sql" rows="2">SELECT label, COUNT(*) episodes, SUM(violation) violations FROM episodes GROUP BY label</textarea>
    <button onclick="runq()">Run query</button>
    <button style="background:var(--c3)" onclick="doExport()">Export SFT dataset (validated)</button>
    <div id="qout"></div>
    <div class="note">Export excludes violating episodes by default and reports
    hard-example candidates (thumbs-down turns) for the next data recipe.</div>
  </section>
</main>
<script>
let lastRow = 0;
const CAT = [getComputedStyle(document.body).getPropertyValue('--c1'),
             getComputedStyle(document.body).getPropertyValue('--c2'),
             getComputedStyle(document.body).getPropertyValue('--c3')];
const labelColor = {};   // color follows the entity, never its rank
let nextSlot = 0;
function colorFor(label){
  if(!(label in labelColor)) labelColor[label] = CAT[Math.min(nextSlot++, CAT.length-1)];
  return labelColor[label]; }
function bars(el, entries, colorfn){
  const max = Math.max(1, ...entries.map(e=>e[1]));
  el.innerHTML = entries.map(([k,v])=>`<div class="bar">
    <div class="name" title="${k}">${k}</div>
    <div class="track"><div class="fill" style="width:${100*v/max}%;
      background:${colorfn(k)}"></div></div>
    <div class="val">${v}</div></div>`).join('') || '<div class="note">no data yet</div>'; }
async function tick(){
  try{
    const st = await (await fetch('stats')).json();
    document.getElementById('t-ep').textContent = st.episodes_total;
    let viol=0, ep=0, inv=0, turns=0, up=0, down=0;
    const byl = [];
    for(const [k,v] of Object.entries(st.per_label)){
      viol+=v.violations; ep+=v.episodes; turns+=v.turns;
      inv+=v.invalid_action_rate*v.turns; up+=v.thumbs_up; down+=v.thumbs_down;
      byl.push([k, v.episodes]); }
    const vr = ep? (100*viol/ep) : 0;
    const tv = document.getElementById('t-viol');
    tv.textContent = vr.toFixed(1)+'%'; tv.className = 'v '+(viol? 'bad':'good');
    document.getElementById('t-inv').textContent =
      (turns? (100*inv/turns):0).toFixed(1)+'%';
    document.getElementById('t-fb').textContent = up+' / '+down;
    bars(document.getElementById('bylabel'), byl, colorFor);
    bars(document.getElementById('tags'),
         Object.entries(st.tags).sort((a,b)=>b[1]-a[1]), ()=>CAT[0]);
    const fd = await (await fetch('feed?since='+lastRow)).json();
    const feed = document.getElementById('feed');
    for(const e of fd.reverse()){
      lastRow = Math.max(lastRow, e.rowid);
      const d = document.createElement('div'); d.className = 'ev '+e.kind;
      const p = e.payload;
      const body = e.kind==='turn' ? (p.agent||'').slice(0,110)
        : e.kind==='feedback' ? (p.vote==='up'?'👍':'👎')+' on turn '+p.i
        : e.kind==='episode_end' ? (p.verdict||'').slice(0,110)
        : (p.label||'')+' — '+(p.brief||'');
      d.innerHTML = `<span class="kind">${e.kind}</span>
        <span class="body">${body.replace(/</g,'&lt;')}</span>`;
      feed.prepend(d); }
    while(feed.children.length>200) feed.lastChild.remove();
    document.getElementById('live').style.opacity =
      1.5 - (document.getElementById('live').style.opacity||0.5);
  }catch(e){ document.getElementById('live').textContent='● reconnecting…'; }
}
async function runq(){
  const sql = document.getElementById('sql').value;
  const r = await (await fetch('query?sql='+encodeURIComponent(sql))).json();
  const out = document.getElementById('qout');
  if(r.error){ out.innerHTML = `<div class="note">⚠ ${r.error}</div>`; return; }
  out.innerHTML = '<table><tr>'+r.columns.map(c=>`<th>${c}</th>`).join('')+'</tr>'
    + r.rows.map(row=>'<tr>'+row.map(c=>`<td>${String(c).replace(/</g,'&lt;')
        .slice(0,80)}</td>`).join('')+'</tr>').join('')+'</table>'; }
async function doExport(){
  const r = await (await fetch('export')).json();
  const blob = new Blob([r.data.map(x=>JSON.stringify(x)).join('\n')],
                        {type:'application/jsonl'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'pennydata_sft.jsonl'; a.click();
  document.getElementById('qout').innerHTML =
    '<pre style="font-size:12px">'+JSON.stringify(r.report,null,1)+'</pre>'; }
setInterval(tick, 2000); tick();
</script></body></html>"""
