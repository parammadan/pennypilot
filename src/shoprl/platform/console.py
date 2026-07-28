"""PennyData console v2 — a real analytics surface (stdlib-served, no build).

Views: Overview (tiles + time-series) · Funnel · Slices · Cohorts · Sessions
(drill-down to full transcripts) · Data Quality · SQL. Global label filter.
Palette = validated dataviz set; single-axis charts; hover tooltips; identity
never color-alone (direct labels everywhere); table views throughout.
"""

CONSOLE_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>PennyData — open data platform</title>
<style>
:root{ --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --c1:#2a78d6; --c2:#eb6834; --c3:#1baf7a;
  --good:#0ca30c; --warning:#fab219; --critical:#d03b3b; }
@media (prefers-color-scheme: dark){ :root{
  --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.12);
  --c1:#3987e5; --c2:#d95926; --c3:#199e70; } }
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
  font:14px/1.45 -apple-system,system-ui,sans-serif}
header{padding:12px 22px;border-bottom:1px solid var(--border);display:flex;
  align-items:center;gap:14px;flex-wrap:wrap}
header h1{font-size:17px;margin:0}
.badge{font-size:11px;padding:3px 10px;border-radius:12px;font-weight:600}
.badge.ok{background:rgba(12,163,12,.12);color:var(--good)}
.badge.bad{background:rgba(208,59,59,.12);color:var(--critical)}
nav{display:flex;gap:2px;padding:0 22px;border-bottom:1px solid var(--border)}
nav button{background:none;border:0;border-bottom:2px solid transparent;
  color:var(--ink2);padding:10px 14px;font-size:13px;cursor:pointer}
nav button.on{color:var(--ink);border-bottom-color:var(--c1);font-weight:600}
.filters{display:flex;gap:10px;align-items:center;padding:12px 22px;
  font-size:13px;color:var(--ink2)}
select{background:var(--surface);color:var(--ink);border:1px solid var(--border);
  border-radius:6px;padding:5px 8px;font-size:13px}
main{padding:6px 22px 30px;max-width:1380px}
.view{display:none}.view.on{display:block}
.tiles{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:10px 0 18px}
.tile{background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:12px 14px}
.tile .v{font-size:24px;font-weight:650;font-variant-numeric:tabular-nums}
.tile .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.tile .v.good{color:var(--good)}.tile .v.bad{color:var(--critical)}
.card{background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:14px 16px;margin-bottom:16px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--ink2);margin:0 0 10px}
.duo{display:grid;grid-template-columns:1fr 1fr;gap:16px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 10px;border-bottom:1px solid var(--grid);text-align:left;
  font-variant-numeric:tabular-nums}
th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;
  letter-spacing:.04em}
tr.click{cursor:pointer} tr.click:hover{background:rgba(42,120,214,.06)}
.bar{display:grid;grid-template-columns:150px 1fr 90px;gap:8px;
  align-items:center;margin:6px 0;font-variant-numeric:tabular-nums}
.bar .name{font-size:12px;color:var(--ink2)}
.bar .track{height:15px;border-radius:4px;background:var(--grid);
  position:relative;overflow:hidden}
.bar .fill{position:absolute;inset:0 auto 0 0;border-radius:4px;min-width:2px}
.bar .val{font-size:12px;color:var(--ink)}
.dev{display:inline-block;min-width:52px}
.dev.up{color:var(--critical)}.dev.dn{color:var(--good)}
#tip{position:fixed;display:none;background:var(--surface);color:var(--ink);
  border:1px solid var(--border);border-radius:6px;padding:6px 10px;
  font-size:12px;pointer-events:none;z-index:99;box-shadow:0 4px 14px rgba(0,0,0,.15)}
svg text{fill:var(--muted);font-size:10px}
.note{font-size:11px;color:var(--muted);margin-top:8px}
textarea{width:100%;background:var(--page);color:var(--ink);
  border:1px solid var(--border);border-radius:6px;padding:8px;font:12px monospace}
button.act{background:var(--c1);border:0;color:#fff;border-radius:6px;
  padding:7px 14px;font-size:13px;cursor:pointer;margin:8px 8px 0 0}
#modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.45);
  z-index:50;align-items:center;justify-content:center}
#modal .box{background:var(--surface);border-radius:12px;max-width:760px;
  width:92%;max-height:80vh;overflow:auto;padding:18px 20px}
.turn{margin:8px 0;padding:8px 10px;border-radius:8px;font-size:13px}
.turn.agent{background:rgba(42,120,214,.08)}
.turn.user{background:var(--grid)}
.fb{margin-left:6px}
</style></head><body>
<header><h1>🧮 PennyData</h1>
  <span style="color:var(--muted);font-size:12px">open data platform · fed live by the PennyMart store via Kafka</span>
  <span class="badge ok" id="dq">data: …</span>
  <span class="badge" id="live" style="color:var(--good)">● live</span></header>
<nav id="nav"></nav>
<div class="filters">cohort <select id="f-label"><option value="">all</option></select>
  &nbsp; slice metric <select id="f-metric">
    <option>abandoned</option><option>reformulated</option>
    <option>repeated</option><option>violation</option></select>
  <span class="note" id="updated"></span></div>
<main>
<div class="view" id="v-overview">
  <div class="tiles">
    <div class="tile"><div class="v" id="t-ep">–</div><div class="k">episodes</div></div>
    <div class="tile"><div class="v" id="t-viol">–</div><div class="k">violation rate</div></div>
    <div class="tile"><div class="v" id="t-aband">–</div><div class="k">abandonment</div></div>
    <div class="tile"><div class="v" id="t-reform">–</div><div class="k">reformulation</div></div>
    <div class="tile"><div class="v" id="t-fb">–</div><div class="k">feedback 👍/👎</div></div>
  </div>
  <div class="duo">
    <div class="card"><h2>Sessions over time</h2><div id="ch-vol"></div></div>
    <div class="card"><h2>Abandonment over time <span id="anom" style="color:var(--critical)"></span></h2><div id="ch-rate"></div></div>
  </div>
  <div class="card"><h2>Live event feed</h2><div id="feed"
    style="max-height:260px;overflow:auto;font-size:12px"></div></div>
</div>
<div class="view" id="v-funnel"><div class="card">
  <h2>Behavioral funnel <span id="fun-lb" style="color:var(--c1)"></span></h2>
  <div id="funnel"></div>
  <div class="note">conversion % = share of the PREVIOUS stage that reached this one</div>
</div></div>
<div class="view" id="v-slices"><div class="card">
  <h2>Ranked slices — where behavior deviates</h2>
  <div id="slices"></div>
  <div class="note" id="slice-note"></div>
</div></div>
<div class="view" id="v-cohorts"><div class="card">
  <h2>Cohorts (model versions / arms / personas)</h2><div id="cohorts"></div>
</div></div>
<div class="view" id="v-sessions"><div class="card">
  <h2>Session explorer <span class="note">click a row for the full transcript</span></h2>
  <div class="filters" style="padding:6px 0">show
    <select id="s-filter"><option value="">all</option>
      <option value="abandoned=1">abandoned</option>
      <option value="reformulated=1">reformulated</option>
      <option value="violation=1">violations</option>
      <option value="carted=1">carted</option></select></div>
  <div id="sessions"></div>
</div></div>
<div class="view" id="v-quality"><div class="duo">
  <div class="card"><h2>Data-quality checks (validate before blaming the model)</h2>
    <div id="dqchecks"></div></div>
  <div class="card"><h2>Time-bucket anomalies (median/MAD, low-n never flagged)</h2>
    <div id="anomalies"></div></div>
</div></div>
<div class="view" id="v-sql"><div class="card">
  <h2>Self-service SQL (SELECT-only) — tables: episodes, turns, ui_events, events</h2>
  <textarea id="sql" rows="3">SELECT label, COUNT(*) episodes, SUM(violation) violations FROM episodes GROUP BY label ORDER BY 2 DESC</textarea>
  <button class="act" onclick="runq()">Run query</button>
  <button class="act" style="background:var(--c3)" onclick="doExport()">Export SFT dataset (validated)</button>
  <div id="qout"></div>
  <div class="note">Export excludes violating episodes; thumbs-down turns surface as hard-example candidates.</div>
</div></div>
</main>
<div id="tip"></div>
<div id="modal" onclick="this.style.display='none'"><div class="box"
  onclick="event.stopPropagation()"><h2 id="m-title"></h2><div id="m-body"></div></div></div>
<script>
const $ = id => document.getElementById(id);
const VIEWS = ["overview","funnel","slices","cohorts","sessions","quality","sql"];
let view = "overview", lastRow = 0;
const nav = $("nav");
VIEWS.forEach(v=>{const b=document.createElement("button");
  b.textContent=v[0].toUpperCase()+v.slice(1); b.id="nav-"+v;
  b.onclick=()=>{view=v;VIEWS.forEach(x=>{$("v-"+x).classList.toggle("on",x===v);
    $("nav-"+x).classList.toggle("on",x===v)});refresh()};nav.appendChild(b);});
$("v-overview").classList.add("on");$("nav-overview").classList.add("on");
const CAT=[getComputedStyle(document.body).getPropertyValue('--c1').trim(),
           getComputedStyle(document.body).getPropertyValue('--c2').trim(),
           getComputedStyle(document.body).getPropertyValue('--c3').trim()];
const labelColor={};let nextSlot=0;
const colorFor=l=>{if(!(l in labelColor))labelColor[l]=CAT[Math.min(nextSlot++,2)];
  return labelColor[l]};
const esc=t=>String(t??"").replace(/&/g,"&amp;").replace(/</g,"&lt;");
const pct=v=>v==null?"–":(100*v).toFixed(1)+"%";

function lineChart(el,pts,color,fmt){
  if(!pts.length){el.innerHTML='<div class="note">no data yet</div>';return;}
  const W=560,H=150,P=28;
  const xs=pts.map(p=>p.t),ys=pts.map(p=>p.v);
  const x0=Math.min(...xs),x1=Math.max(...xs)||1,y1=Math.max(...ys)||1;
  const X=t=>P+(W-2*P)*(t-x0)/Math.max(x1-x0,1),Y=v=>H-P-(H-2*P)*v/y1;
  const d=pts.map((p,i)=>(i?"L":"M")+X(p.t).toFixed(1)+","+Y(p.v).toFixed(1)).join(" ");
  const grid=[0,.5,1].map(f=>`<line x1="${P}" x2="${W-P}" y1="${Y(y1*f)}" y2="${Y(y1*f)}"
    stroke="var(--grid)" stroke-width="1"/><text x="4" y="${Y(y1*f)+3}">${fmt(y1*f)}</text>`).join("");
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" style="width:100%">${grid}
    <path d="${d}" fill="none" stroke="${color}" stroke-width="2"
      stroke-linejoin="round"/>${pts.map(p=>`<circle cx="${X(p.t)}" cy="${Y(p.v)}"
      r="7" fill="transparent" data-tip="${new Date(p.t*1000).toLocaleTimeString()} — ${fmt(p.v)} (n=${p.n})"/>`).join("")}</svg>`;
  el.querySelectorAll("circle").forEach(c=>{
    c.onmousemove=e=>{const t=$("tip");t.style.display="block";
      t.textContent=c.dataset.tip;t.style.left=e.clientX+12+"px";
      t.style.top=e.clientY-10+"px";};
    c.onmouseout=()=>$("tip").style.display="none";});
}
function bars(el,rows,colorfn){
  const max=Math.max(1,...rows.map(r=>r.v));
  el.innerHTML=rows.map(r=>`<div class="bar"><div class="name">${esc(r.k)}</div>
    <div class="track"><div class="fill" style="width:${100*r.v/max}%;
    background:${colorfn(r.k)}"></div></div><div class="val">${r.txt??r.v}</div></div>`)
    .join("")||'<div class="note">no data</div>';
}
function tbl(el,cols,rows,onclick){
  el.innerHTML='<table><tr>'+cols.map(c=>`<th>${esc(c)}</th>`).join("")+'</tr>'
    +rows.map((r,i)=>`<tr class="${onclick?'click':''}" data-i="${i}">`
      +r.map(c=>`<td>${c}</td>`).join("")+'</tr>').join("")+'</table>';
  if(onclick)el.querySelectorAll("tr.click").forEach(tr=>
    tr.onclick=()=>onclick(+tr.dataset.i));
}
async function j(u){return (await fetch(u)).json()}

async function refresh(){
  try{
    const lb=$("f-label").value;
    const b=await j("behavior");
    const dq=$("dq");dq.textContent="data: "+b.data_quality.verdict;
    dq.className="badge "+(b.data_quality.verdict==="clean"?"ok":"bad");
    $("updated").textContent="updated "+new Date().toLocaleTimeString();
    const st=await j("stats");
    const labels=Object.keys(st.per_label);
    const sel=$("f-label");
    if(sel.options.length-1!==labels.length){const cur=sel.value;
      sel.innerHTML='<option value="">all</option>'+labels.map(l=>
        `<option${l===cur?" selected":""}>${esc(l)}</option>`).join("");}
    if(view==="overview"){
      const m=b.metrics;
      $("t-ep").textContent=m.sessions;
      const viol=m.violation.value; const tv=$("t-viol");
      tv.textContent=pct(viol);tv.className="v "+(viol>0?"bad":"good");
      $("t-aband").textContent=pct(m.abandonment.value);
      $("t-reform").textContent=pct(m.reformulation.value);
      let up=0,dn=0;for(const v of Object.values(st.per_label)){up+=v.thumbs_up;dn+=v.thumbs_down;}
      $("t-fb").textContent=up+" / "+dn;
      const ts=await j("timeseries?metric=abandoned&bucket=120");
      lineChart($("ch-vol"),ts.series.map(p=>({t:p.t,v:p.n,n:p.n})),CAT[0],v=>v.toFixed(0));
      lineChart($("ch-rate"),ts.series.map(p=>({t:p.t,v:p.rate,n:p.n})),CAT[1],pct);
      $("anom").textContent=ts.anomalies.length?`⚠ ${ts.anomalies.length} anomaly`:``;
      const fd=await j("feed?since="+lastRow);const feed=$("feed");
      for(const e of fd.reverse()){lastRow=Math.max(lastRow,e.rowid);
        const d=document.createElement("div");
        const p=e.payload;const body=e.kind==="turn"?(p.agent||"").slice(0,110)
          :e.kind==="feedback"?(p.vote==="up"?"👍":"👎")+" turn "+p.i
          :e.kind==="ui"?(p.type+" "+(p.target||"")).slice(0,60)
          :e.kind==="episode_end"?(p.verdict||"").slice(0,110)
          :(p.label||"")+" — "+(p.brief||"");
        d.innerHTML=`<span style="color:var(--muted);display:inline-block;width:106px">${e.kind}</span> ${esc(body)}`;
        d.style.cssText="padding:4px 0;border-bottom:1px solid var(--grid);white-space:nowrap;overflow:hidden;text-overflow:ellipsis";
        feed.prepend(d);}
      while(feed.children.length>150)feed.lastChild.remove();
    }
    if(view==="funnel"){
      const f=await j("funnel"+(lb?"?label="+encodeURIComponent(lb):""));
      $("fun-lb").textContent=f.label;
      bars($("funnel"),f.stages.map(s=>({k:s.stage,v:s.sessions,
        txt:s.sessions+(s.conversion_from_prev!=null?` (${pct(s.conversion_from_prev)})`:"")})),
        ()=>CAT[2]);
    }
    if(view==="slices"){
      const sl=await j("slices?metric="+$("f-metric").value+"&top=12");
      tbl($("slices"),["slice","support","value","baseline","deviation"],
        (sl.top_slices||[]).map(c=>[`<code>${esc(JSON.stringify(c.slice))}</code>`,
          c.support,pct(c.value),pct(c.baseline),
          `<span class="dev ${c.deviation>0?'up':'dn'}">${c.deviation>0?"▲":"▼"} ${pct(Math.abs(c.deviation))}</span>`]));
      $("slice-note").textContent=sl.note+` · min support ${sl.min_support} · ${sl.suppressed_low_n} low-n slices suppressed`;
    }
    if(view==="cohorts"){
      tbl($("cohorts"),["label","episodes","carted","violations","invalid rate","👍","👎"],
        Object.entries(st.per_label).map(([k,v])=>[
          `<span style="color:${colorFor(k)}">●</span> ${esc(k)}`,v.episodes,
          v.carted,v.violations,pct(v.invalid_action_rate),v.thumbs_up,v.thumbs_down]));
    }
    if(view==="sessions"){
      const f=$("s-filter").value;
      const s=await j("sessions?limit=25"+(f?"&"+f:"")+(lb?"&label="+encodeURIComponent(lb):""));
      window.__sess=s.sessions;
      tbl($("sessions"),["session","turns","first agent line"],
        s.sessions.map(x=>[esc(x.session_id),x.turns.length,
          esc((x.turns[0]?.agent||"").slice(0,90))]),
        i=>{const x=window.__sess[i];$("m-title").textContent=x.session_id;
          $("m-body").innerHTML=x.turns.map(t=>
            `<div class="turn agent">🤖 ${esc(t.agent)}${t.feedback?`<span class="fb">${t.feedback==="up"?"👍":"👎"}</span>`:""}</div>`
            +(t.observation?`<div class="turn user">🧑 ${esc(t.observation)}</div>`:"")).join("");
          $("modal").style.display="flex";});
    }
    if(view==="quality"){
      tbl($("dqchecks"),["check","count"],
        Object.entries(b.data_quality.checks).map(([k,v])=>[esc(k),v]));
      const ts=await j("timeseries?metric=abandoned&bucket=120");
      tbl($("anomalies"),["time","n","rate","median","robust deviations"],
        ts.anomalies.map(a=>[new Date(a.t*1000).toLocaleTimeString(),a.n,
          pct(a.rate),pct(a.baseline_median),a.deviations]));
      if(!ts.anomalies.length)$("anomalies").innerHTML+='<div class="note">none — no eligible bucket deviates from the median beyond 4 robust deviations</div>';
    }
  }catch(e){$("live").textContent="● reconnecting…";return;}
  $("live").textContent="● live";
}
async function runq(){
  const r=await j("query?sql="+encodeURIComponent($("sql").value)).catch(()=>null);
  const out=$("qout");
  if(!r||r.error){out.innerHTML=`<div class="note">⚠ ${esc(r?.error||"query failed")}</div>`;return;}
  tbl(out,r.columns,r.rows.map(row=>row.map(c=>esc(String(c).slice(0,90)))));
}
async function doExport(){
  const r=await j("export");
  const blob=new Blob([r.data.map(x=>JSON.stringify(x)).join("\n")],{type:"application/jsonl"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);
  a.download="pennydata_sft.jsonl";a.click();
  $("qout").innerHTML="<pre style='font-size:12px'>"+esc(JSON.stringify(r.report,null,1))+"</pre>";
}
$("f-label").onchange=refresh;$("f-metric").onchange=refresh;
$("s-filter").onchange=refresh;
setInterval(refresh,3000);refresh();
</script></body></html>"""
