"""BrowserDemoEnvironment — visible-Chromium projection of saved trajectories.

DEMO/REPLAY ONLY, never the training path: the browser renders decisions an
episode already made (a `transcript_record`), it does not make them. The
storefront is a self-contained local HTML page generated from the same synthetic
catalog the episode shopped — no real site, no real cart, no network.

Replay contract: every visible browser event maps 1:1 to a recorded abstract
action; the conversation pane shows the recorded turns verbatim; the permission
modal appears before any simulated add-to-cart, showing items / total / savings
/ the user's actual recorded reply.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from shoprl.data.catalog import Product

_SKU_RE = re.compile(r"(?:[A-Z]{2,4}-\d{3,5}|B0[A-Z0-9]{8})")


def skus_in(text: str) -> list[str]:
    """SKUs mentioned in a recorded observation (the search-results parser)."""
    seen, out = set(), []
    for m in _SKU_RE.findall(text or ""):
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def render_storefront_html(catalog,
                           title: str = "PennyMart — simulated storefront") -> str:
    items = [p.model_dump() if hasattr(p, "model_dump") else dict(p)
             for p in catalog]
    products = json.dumps(items)
    return """<!doctype html><html><head><meta charset="utf-8">
<title>__TITLE__</title><style>
  :root { --nav:#131921; --nav2:#232f3e; --link:#007185; --price:#0f1111;
          --star:#de7921; --btn:#ffd814; --btn2:#f7ca00; --ink:#0f1111;
          --muted:#565959; --bg:#eaeded; --ok:#067d62; }
  * { box-sizing:border-box; margin:0; }
  body { font:13px/1.4 "Amazon Ember",Arial,-apple-system,system-ui,sans-serif;
         color:var(--ink); background:var(--bg);
         display:grid; grid-template-columns:340px 1fr; height:100vh; }
  #chat { background:#fff; border-right:1px solid #d5d9d9; padding:14px;
          overflow-y:auto; display:flex; flex-direction:column; }
  #msgs { flex:1; overflow-y:auto; }
  #saybar { display:flex; gap:6px; padding-top:10px; border-top:1px solid #d5d9d9; }
  #say { flex:1; padding:9px 11px; border:1px solid #888c8c; border-radius:8px;
         font-size:13px; }
  #send { padding:9px 14px; border:1px solid #a88734; border-radius:8px;
          background:var(--btn); color:#0f1111; font-weight:600; cursor:pointer; }
  #chat h2 { font-size:13px; color:var(--muted); text-transform:uppercase;
             letter-spacing:.06em; margin-bottom:10px; }
  .bubble { max-width:92%; padding:8px 11px; border-radius:12px; margin:6px 0;
            white-space:pre-wrap; word-break:break-word; font-size:13px; }
  .user { background:#f0f2f2; }
  .agent { background:#eef6ff; border:1px solid #cde5ff; margin-left:auto;
           font-family:ui-monospace,Menlo,monospace; font-size:11.5px; }
  .note { color:var(--muted); font-size:11px; margin:2px 0 8px; }
  #store { display:flex; flex-direction:column; overflow:hidden; background:var(--bg); }
  header { background:var(--nav); display:flex; align-items:center; gap:14px;
           padding:8px 16px; }
  header .logo { color:#fff; font-size:20px; font-weight:800; letter-spacing:-.5px; }
  header .logo em { color:#ff9900; font-style:normal; }
  #search { flex:1; display:flex; height:38px; border-radius:6px; overflow:hidden; }
  #search input { flex:1; border:0; padding:0 12px; font-size:14px; }
  #search .go { width:44px; background:var(--btn2); display:flex; align-items:center;
                justify-content:center; font-size:16px; }
  header .cart { color:#fff; font-weight:700; font-size:13px; white-space:nowrap; }
  header .cart b { color:#f90; font-size:16px; }
  #subnav { background:var(--nav2); color:#fff; font-size:12px; padding:6px 16px;
            display:flex; gap:14px; align-items:center; }
  #subnav .sim { color:#febd69; }
  #resultbar { padding:10px 18px 4px; color:var(--muted); font-size:13px; }
  #resultbar b { color:var(--ink); }
  #hintbar { display:none; margin:8px 16px 0; padding:7px 12px; font-size:12px;
             background:#fff8e5; border:1px dashed #e0c86a; border-radius:6px;
             color:#6a5a1a; }
  #hintbar b { color:#4a3f10; }
  #grid { padding:6px 12px 16px; overflow-y:auto; }
  #empty { display:flex; flex-direction:column; align-items:center;
           justify-content:center; height:70%; color:var(--muted); gap:10px;
           text-align:center; padding:40px; }
  #empty .big { font-size:38px; } #empty h2 { font-size:17px; color:var(--ink);
           font-weight:600; } #empty p { max-width:420px; }
  .card { background:#fff; border-bottom:1px solid #e7e7e7; padding:16px;
          display:grid; grid-template-columns:200px 1fr; gap:18px; }
  .card .thumb { height:170px; display:flex; align-items:center; justify-content:center;
                 background:#fff; position:relative; }
  .card .thumb svg { width:160px; height:110px; }
  .card .info { display:flex; flex-direction:column; gap:4px; }
  .card.hit { background:#fff; }
  .card.selected { background:#fffbea; outline:2px solid #ffd814; }
  .card h3 { font-size:18px; font-weight:500; color:var(--link); line-height:1.3; }
  .card .rateline { display:flex; align-items:center; gap:6px; }
  .card .rateline .st { color:var(--star); font-size:14px; letter-spacing:1px; }
  .card .rateline .n { color:var(--link); font-size:12px; }
  .card .price { color:var(--price); margin:2px 0; }
  .card .price .cur { font-size:13px; vertical-align:top; }
  .card .price .whole { font-size:26px; font-weight:500; }
  .card .price .frac { font-size:13px; vertical-align:top; }
  .card .specs { color:var(--muted); font-size:13px; }
  .card .deliv { font-size:12px; color:var(--ink); }
  .card .deliv b { font-weight:700; }
  .card .stock { color:var(--ok); font-size:13px; margin-top:2px; }
  .card .buy { margin-top:8px; width:150px; text-align:center; padding:8px;
               background:var(--btn); border:1px solid #a88734; border-radius:999px;
               font-size:13px; color:#0f1111; }
  .card.selected .buy { background:#ffa41c; border-color:#ff8f00; font-weight:600; }
  .badge-best { display:inline-block; background:#232f3e; color:#fff; font-size:11px;
                font-weight:700; padding:2px 8px; border-radius:3px; width:fit-content; }
  .badge-best em { color:#ff9900; font-style:normal; }
  #banner { display:none; padding:10px 18px; background:#e3f0e3;
            color:#067d62; font-weight:600; border-bottom:1px solid #067d62; }
  #modal { display:none; position:fixed; inset:0; background:rgba(15,17,17,.5);
           align-items:center; justify-content:center; }
  #modal .box { background:#fff; border-radius:10px; padding:22px; width:440px; }
  #modal h3 { margin-bottom:10px; font-size:18px; }
  #modal ul { margin:8px 0 8px 18px; } #modal .savings { color:#067d62; font-weight:700; }
  #modal .reply { margin-top:10px; font-style:italic; color:var(--muted); }
  #modal .actions { margin-top:16px; display:flex; gap:8px; justify-content:flex-end; }
  #modal button { padding:9px 18px; border-radius:999px; border:1px solid #888c8c;
                  background:#fff; font-weight:600; cursor:pointer; }
  #modal button.approve { background:var(--btn); border-color:#a88734; }
</style></head><body>
<aside id="chat"><h2>Assistant</h2><div id="msgs"></div>
  <div id="saybar"><input id="say" placeholder="…" autocomplete="off">
  <button id="send">Send</button></div></aside>
<main id="store">
  <header><span class="logo">Penny<em>Mart</em></span>
    <div id="search"><input placeholder="Search PennyMart"><div class="go">🔍</div></div>
    <span class="cart">🛒 Cart <b id="cartn">0</b></span></header>
  <div id="subnav"><span>All</span><span>Laptops</span><span>Deals</span>
    <span class="sim">● SIMULATED — no real purchases, stylized images</span></div>
  <div id="banner"></div>
  <div id="hintbar"></div>
  <div id="resultbar"></div>
  <div id="grid"></div>
</main>
<div id="modal"><div class="box"><h3>Add to your cart?</h3>
  <div id="modal-body"></div><div class="reply" id="modal-reply"></div>
  <div class="actions"><button id="hold">Not now</button>
  <button class="approve" id="approve">Add to Cart</button></div></div></div>
<script>
const PRODUCTS = __PRODUCTS__;
const grid = document.getElementById("grid");
const chat = document.getElementById("msgs");
// Deterministic per-brand tint (stylized thumbnail — an honest icon, not a
// faked product photo; real product images are unavailable for this data).
const BRAND_HUE = {Asus:"#2a78d6", Dell:"#0a7d4f", Lenovo:"#c0392b",
  HP:"#00838f", Apple:"#555", Framework:"#eb6834", Razer:"#3fae29",
  Acer:"#8e44ad"};
function laptopSVG(p){
  const c = BRAND_HUE[p.brand] || "#6b7785";
  return `<svg viewBox="0 0 120 80"><rect x="24" y="12" width="72" height="46" rx="4"
    fill="#fff" stroke="${c}" stroke-width="3"/><rect x="30" y="18" width="60"
    height="34" rx="2" fill="${c}" opacity="0.16"/><path d="M14 62 h92 l-6 8 h-80 z"
    fill="${c}" opacity="0.85"/><rect x="52" y="62" width="16" height="3"
    rx="1.5" fill="#fff" opacity="0.7"/></svg>`; }
function rateval(p){
  const s = 3.4 + ((p.battery_hrs||10)/20)*0.9 + ((p.ram_gb||8)>=32?0.5:0)
            + (0.4 - ((p.weight_lbs||4)-2)/12);
  return Math.max(3.2, Math.min(5, s)); }
function rateline(p){
  const r = rateval(p), full = Math.round(r);
  const nrev = 40 + Math.round((p.price||500) % 900);   // deterministic review count
  return `<div class="rateline"><span class="st">${"★".repeat(full)}${"☆".repeat(5-full)}</span>`+
         `<span class="n">${r.toFixed(1)} · ${nrev.toLocaleString()} ratings</span></div>`; }
function priceHTML(v){ const [w,f]=v.toFixed(2).split(".");
  return `<div class="price"><span class="cur">$</span><span class="whole">${(+w).toLocaleString()}</span><span class="frac">${f}</span></div>`; }
function specLine(p){ return p.ram_gb===undefined ? "" :
  `<div class="specs">${p.ram_gb}GB RAM &nbsp;|&nbsp; ${p.weight_lbs} lb &nbsp;|&nbsp; `+
  `${p.battery_hrs} hr battery &nbsp;|&nbsp; ${p.brand}</div>`; }
function card(p, best){ return `<div class="card" data-sku="${p.sku}">
  <div class="thumb">${laptopSVG(p)}</div>
  <div class="info">
    ${best?'<div class="badge-best"><em>Best</em> Match</div>':''}
    <h3>${p.name}</h3>${rateline(p)}${priceHTML(p.price)}${specLine(p)}
    <div class="deliv">FREE delivery <b>tomorrow</b></div>
    <div class="stock">In stock</div>
    <div class="buy">Add to Cart</div>
    <div class="specs" style="font-size:11px;color:#aaa">${p.sku}</div>
  </div></div>`; }
function show(list, bestSku){ grid.innerHTML = list.map(p=>card(p, p.sku===bestSku)).join(""); }
const rbar = document.getElementById("resultbar");
window.pennymart = {
  all(){ show(PRODUCTS); rbar.innerHTML = `Showing <b>${PRODUCTS.length}</b> products`; },
  welcome(){ grid.innerHTML = `<div id="empty"><div class="big">🛍️</div>
    <h2>Your shopping assistant is ready</h2>
    <p>Tell it what you're looking for in the chat on the left — a budget,
    a brand, how much RAM. It'll ask what it needs, then find the
    cheapest option that fits and show it here.</p></div>`;
    rbar.textContent = ""; },
  demoHint(text){ const h=document.getElementById("hintbar");
    h.innerHTML = `🎫 <b>Demo hint (only you see this — the assistant does NOT):</b> `
      + text + ` — reveal these only when it asks.`; h.style.display="block"; },
  bubble(role, text, note){ const d=document.createElement("div");
    d.className="bubble "+role; d.textContent=text; chat.appendChild(d);
    if(role==="agent"){
      // human-feedback capture: one 👍/👎 pair per agent reply, votes land in
      // window.__feedback [{i, vote, text}] for the driver to harvest
      const k=(window.__fbN=(window.__fbN||0)+1)-1;
      const fb=document.createElement("div");
      fb.style.cssText="margin:-2px 0 4px 6px;font-size:13px;user-select:none";
      const mk=(sym)=>{const e=document.createElement("span");e.textContent=sym;
        e.style.cssText="cursor:pointer;opacity:.4;margin-right:10px";return e;};
      const up=mk("👍"), dn=mk("👎");
      const vote=(v,me,other)=>{window.__feedback=(window.__feedback||[])
          .filter(f=>f.i!==k);
        window.__feedback.push({i:k,vote:v,text:text.slice(0,200)});
        me.style.opacity=1; other.style.opacity=.2;};
      up.onclick=()=>vote("up",up,dn); dn.onclick=()=>vote("down",dn,up);
      fb.appendChild(up); fb.appendChild(dn); chat.appendChild(fb);
    }
    if(note){ const n=document.createElement("div"); n.className="note";
      n.textContent=note; chat.appendChild(n);} chat.scrollTop=chat.scrollHeight; },
  results(skus){ const by={}; PRODUCTS.forEach(p=>by[p.sku]=p);
    const hits=skus.map(s=>by[s]).filter(Boolean);   // preserve cheapest-first order
    rbar.innerHTML = `<b>${hits.length}</b> results &nbsp;·&nbsp; sorted by price (lowest first)`;
    show(hits, skus[0]);                             // first = cheapest = best match
    hits.forEach(p=>document.querySelector(`[data-sku="${p.sku}"]`)
      .classList.add("hit")); },
  select(sku){ const el=document.querySelector(`[data-sku="${sku}"]`);
    if(el){ el.classList.add("selected"); el.scrollIntoView({block:"center",behavior:"smooth"}); } },
  permission(items,total,savings,reply){ const b=document.getElementById("modal-body");
    b.innerHTML = `<ul>${items.map(s=>`<li>${s}</li>`).join("")}</ul>
      <div>Estimated total: <b>$${total}</b></div>` +
      (savings!==null?`<div class="savings">Estimated savings: $${savings} vs priciest matching option</div>`:"");
    document.getElementById("modal-reply").textContent = reply?`User: “${reply}”`:"";
    document.getElementById("modal").style.display="flex"; },
  closeModal(){ document.getElementById("modal").style.display="none"; },
  hint(h){ const s=document.getElementById("say"); s.placeholder=h; s.focus(); },
  addToCart(sku){ document.getElementById("cartn").textContent="1";
    const bn=document.getElementById("banner");
    bn.textContent=`✓ Added to Cart — ${sku} (simulated, with your explicit permission)`;
    bn.style.display="block"; },
};
window.__human = null;
// clickstream capture: every click / search input / card hover / modal action
// lands in window.__uiev with a timestamp; the driver drains it each turn.
window.__uiev = [];
(function(){
  const push = (type, target, meta) => window.__uiev.push(
    {type, target, meta: meta||{}, ts: Date.now()/1000});
  document.addEventListener("click", e => {
    const card = e.target.closest("[data-sku]");
    const btn = e.target.closest("button, .buy, .tab, #send, #approve, #deny");
    if (card) push("click", "card:"+card.dataset.sku,
                   {buy: !!e.target.closest(".buy")});
    else if (btn) push("click", btn.id || btn.textContent.trim().slice(0,24));
  }, true);
  const search = document.getElementById("q") || document.querySelector("input[type=search]");
  if (search) search.addEventListener("change",
    () => push("input", "search", {q: search.value.slice(0, 80)}));
  let hoverLast = 0;
  document.addEventListener("mouseover", e => {
    const card = e.target.closest("[data-sku]");
    const now = Date.now();
    if (card && now - hoverLast > 800){ hoverLast = now;
      push("hover", "card:"+card.dataset.sku); }
  }, true);
})();
(function(){
  const say = document.getElementById("say");
  const send = document.getElementById("send");
  function submit(){ const v = say.value.trim(); if(!v) return;
    window.__uiev.push({type:"input", target:"chat",
                        meta:{len:v.length}, ts:Date.now()/1000});
    say.value=""; window.__human = v; }
  send.onclick = submit;
  say.addEventListener("keydown", e => { if (e.key === "Enter") submit(); });
  document.getElementById("approve").addEventListener("click",
    () => { window.__uiev.push({type:"modal", target:"approve", meta:{},
                                ts:Date.now()/1000});
            window.__human = "__yes__"; });
  document.getElementById("hold").addEventListener("click",
    () => { window.__human = "__no__"; });
})();
pennymart.welcome();
</script></body></html>""".replace("__TITLE__", title).replace("__PRODUCTS__", products)


# ---- self-contained chat+shop demo (no Playwright; open in any browser) -----
def render_chat_demo_html(catalog, server_url: str = "http://localhost:8765",
                          system_prompt: str | None = None,
                          title: str = "PennyMart — live assistant") -> str:
    """A storefront page whose OWN JavaScript talks to a serve_policy.py server
    (the 7B chat-face, reached over an SSH tunnel at server_url). You just open
    the file in a normal browser and type — no Playwright, no Python client.

    The page reuses render_storefront_html's markup + window.pennymart API and
    overrides only the input handlers with a fetch-driven chat loop. Shopping is
    projected CLIENT-SIDE from what you typed (an honest demo projection, same
    spirit as the replay path): search filters the embedded catalog by the
    budget / RAM / weight / brand it can read from your messages, and the
    permission gate holds — add_to_cart is refused unless you approved that SKU.
    """
    from shoprl.data.prompts_v2 import SYSTEM_PROMPT_CHAT

    html = render_storefront_html(catalog, title=title)
    sys_json = json.dumps(system_prompt or SYSTEM_PROMPT_CHAT)
    driver = """
<script>
(function(){
  const SERVER = "__SERVER__";
  let messages = [{role:"system", content: __SYS__}];
  const granted = new Set();
  let pendingItems = null, busy = false;
  const say = document.getElementById("say");
  const send = document.getElementById("send");

  // ---- client-side shopping projection (from what YOU typed) ----------------
  function constraints(){
    const text = messages.filter(m=>m.role==="user")
                         .map(m=>m.content).join(" ").toLowerCase();
    const c = {};
    const nums = (text.match(/\\$?\\s*(\\d{3,5})/g)||[])
      .map(s=>parseInt(s.replace(/[^0-9]/g,""))).filter(n=>n>=200 && n<=6000);
    if(nums.length) c.budget = Math.max.apply(null, nums);
    const ram = text.match(/(\\d{1,2})\\s*gb/);   if(ram) c.min_ram = parseInt(ram[1]);
    if(/light|lightweight|ligera|liviana|thin|portable|delgad/.test(text)) c.max_weight = 3.5;
    const wt = text.match(/(?:under|below|less than)\\s*(\\d(?:\\.\\d)?)\\s*(?:lb|lbs|pound)/);
    if(wt) c.max_weight = parseFloat(wt[1]);
    for(const b of ["asus","dell","lenovo","hp","apple","framework","razer","acer"])
      if(new RegExp("\\\\b"+b+"\\\\b").test(text)) c.brand = b;
    return c;
  }
  function filt(cc){
    return PRODUCTS.filter(p=>{
      if(cc.budget && p.price > cc.budget) return false;
      if(cc.min_ram && p.ram_gb < cc.min_ram) return false;
      if(cc.max_weight && p.weight_lbs > cc.max_weight) return false;
      if(cc.brand && String(p.brand).toLowerCase() !== cc.brand) return false;
      return true;
    }).sort((a,b)=>a.price-b.price);
  }
  function search(){                          // progressive relax -> always real hits
    const c = constraints();
    let hits = filt(c);
    if(!hits.length){ const x=Object.assign({},c); delete x.max_weight; hits=filt(x); }
    if(!hits.length){ const x=Object.assign({},c); delete x.max_weight; delete x.brand; hits=filt(x); }
    if(!hits.length){ hits = filt({min_ram:c.min_ram}); }   // keep RAM, drop budget
    if(!hits.length){ hits = PRODUCTS.slice().sort((a,b)=>a.price-b.price); }
    return hits;
  }
  function resultsText(hits){
    if(!hits.length) return "No matching products found.";
    return "Matching products (cheapest first):\\n" + hits.slice(0,6).map(p=>
      "- "+p.sku+": $"+Math.round(p.price)+", "+p.ram_gb+"GB RAM, "+
      p.weight_lbs+"lbs, "+p.battery_hrs+"hrs, "+p.brand).join("\\n");
  }

  // ---- parse the model's optional JSON action out of its prose --------------
  function extractAction(text){
    const i = text.indexOf("{"); if(i<0) return null;
    let depth=0, end=-1;
    for(let j=i;j<text.length;j++){
      if(text[j]==="{") depth++;
      else if(text[j]==="}"){ depth--; if(depth===0){ end=j; break; } } }
    if(end<0) return null;
    try { return JSON.parse(text.slice(i,end+1)); } catch(e){ return null; }
  }
  function prose(text){ const i=text.indexOf("{"); return (i<0?text:text.slice(0,i)).trim(); }

  function setBusy(b){ busy=b; say.disabled=b; send.disabled=b;
    pennymart.hint(b ? "thinking…" : "type your reply…"); if(!b) say.focus(); }

  async function agentStep(){
    setBusy(true);
    try{
      const r = await fetch(SERVER+"/act", {method:"POST",
        headers:{"Content-Type":"application/json"}, body: JSON.stringify({messages})});
      const data = await r.json();
      const text = (data && data.text) || "";
      messages.push({role:"assistant", content:text});
      const p = prose(text); if(p) pennymart.bubble("agent", p);
      const act = extractAction(text);
      setBusy(false);
      if(act) await handle(act);
    }catch(e){
      setBusy(false);
      pennymart.bubble("agent","(couldn't reach the model — is the tunnel up? "+
        "ssh -L 8765:<node>:8765 explorer)");
    }
  }

  async function handle(act){
    if(act.action==="search"){
      const hits = search();
      pennymart.results(hits.map(p=>p.sku));
      messages.push({role:"user", content: resultsText(hits)});
      await agentStep();                       // let it recommend from the hits
    } else if(act.action==="select_product" || act.action==="inspect_product"){
      if(act.product_id) pennymart.select(act.product_id);
    } else if(act.action==="request_cart_permission"){
      pendingItems = act.items || [];
      if(pendingItems[0]) pennymart.select(pendingItems[0]);   // highlight it in the grid
      pennymart.permission(pendingItems, act.estimated_total==null?"":act.estimated_total,
                           null, null);
    } else if(act.action==="add_to_cart"){
      if(granted.has(act.product_id)) pennymart.addToCart(act.product_id);
      else pennymart.bubble("agent",
        "(I can only add it once you approve — tap Add to Cart in the box.)");
    }                                          // ask_user / chat: just wait for you
  }

  function submitUser(){
    if(busy) return;
    const v = say.value.trim(); if(!v) return;
    say.value=""; pennymart.bubble("user", v);
    messages.push({role:"user", content:v});
    agentStep();
  }
  send.onclick = submitUser;
  say.onkeydown = e => { if(e.key==="Enter") submitUser(); };
  document.getElementById("approve").onclick = () => {
    (pendingItems||[]).forEach(s=>granted.add(s));
    pennymart.closeModal();
    pennymart.bubble("user","Yes, please add it.");
    messages.push({role:"user", content:"Yes, please add it."}); agentStep(); };
  document.getElementById("hold").onclick = () => {
    pennymart.closeModal();
    pennymart.bubble("user","No, not yet.");
    messages.push({role:"user", content:"No, not yet."}); agentStep(); };
  pennymart.hint("say hi, or tell me what you're shopping for…");
})();
</script>
""".replace("__SERVER__", server_url.rstrip("/")).replace("__SYS__", sys_json)
    return html.replace("</body>", driver + "</body>")


# ---- replayer ---------------------------------------------------------------
def replay_transcript(bundle: dict, headed: bool = False, slow_mo: int = 0,
                      screenshot_dir: str | None = None,
                      beat_pause_ms: int = 900) -> dict:
    """Replay one saved episode in Chromium. `bundle` is the JSON written by
    scripts/make_demo_transcript.py: {catalog_seed, catalog_n, language, record}.
    Returns a small report (beats shown, screenshots written, cart check)."""
    from playwright.sync_api import sync_playwright

    from shoprl.actions import parse_agent_action
    from shoprl.data.catalog import generate_catalog

    record = bundle["record"]
    catalog = generate_catalog(n=bundle["catalog_n"], seed=bundle["catalog_seed"])
    import tempfile
    workdir = bundle.get("workdir") or tempfile.mkdtemp(prefix="pennymart-")
    html_path = Path(workdir) / "pennymart.html"
    html_path.write_text(render_storefront_html(catalog))

    shots: list[str] = []
    sdir = Path(screenshot_dir) if screenshot_dir else None
    if sdir:
        sdir.mkdir(parents=True, exist_ok=True)

    def snap(page, name: str) -> None:
        if sdir:
            p = str(sdir / f"{len(shots):02d}_{name}.png")
            page.screenshot(path=p)
            shots.append(p)

    state = record["state"]
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headed, slow_mo=slow_mo)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(f"file://{html_path}")
        page.evaluate("pennymart.bubble('user', arguments0)".replace(
            "arguments0", json.dumps(record.get("opener", "(opening message)"))))
        snap(page, "opening")

        for t in record["turns"]:
            r = parse_agent_action(t["action"])
            page.evaluate(f"pennymart.bubble('agent', {json.dumps(t['action'])}, "
                          f"{json.dumps(t['note'])})")
            if r.ok:
                kind = r.action.action
                if kind == "search":
                    page.fill("#search input", "")
                    page.type("#search input", r.action.query, delay=18)
                    page.evaluate(
                        f"pennymart.results({json.dumps(skus_in(t['observation']))})")
                    snap(page, "search_results")
                elif kind in ("inspect_product", "select_product"):
                    page.evaluate(f"pennymart.select({json.dumps(r.action.product_id)})")
                    snap(page, kind)
                elif kind == "request_cart_permission":
                    savings = state.get("estimated_savings")
                    page.evaluate(
                        "pennymart.permission(%s, %s, %s, %s)" % (
                            json.dumps(r.action.items),
                            json.dumps(r.action.estimated_total),
                            json.dumps(savings),
                            json.dumps(t["observation"])))
                    snap(page, "permission_modal")
                    page.wait_for_timeout(beat_pause_ms)
                    granted = "granted" in t["note"]
                    page.click("#approve" if granted else "#hold")
                    page.evaluate("pennymart.closeModal()")
                elif kind == "add_to_cart":
                    if "permitted" in t["note"]:
                        page.evaluate(
                            f"pennymart.addToCart({json.dumps(r.action.product_id)})")
                    snap(page, "cart")
            if t["observation"]:
                page.evaluate(f"pennymart.bubble('user', "
                              f"{json.dumps(t['observation'][:400])})")
            page.wait_for_timeout(beat_pause_ms if headed else 30)

        snap(page, "final")
        cart_shown = page.text_content("#cartn")
        browser.close()

    expected = "1" if state.get("cart_contents") else "0"
    return {"beats": len(record["turns"]), "screenshots": shots,
            "cart_badge": cart_shown, "cart_expected": expected,
            "cart_ok": cart_shown == expected}


# ---- shared projection + live mode -------------------------------------------
def _project_action(page, r, note: str, observation: str, savings,
                    snap, beat_pause_ms: int) -> None:
    """Project ONE parsed action onto the storefront (shared by replay/live)."""
    kind = r.action.action
    if kind == "search":
        page.fill("#search input", "")
        page.type("#search input", r.action.query, delay=18)
        page.evaluate(f"pennymart.results({json.dumps(skus_in(observation))})")
        snap("search_results")
    elif kind in ("inspect_product", "select_product"):
        page.evaluate(f"pennymart.select({json.dumps(r.action.product_id)})")
        snap(kind)
    elif kind == "request_cart_permission":
        page.evaluate("pennymart.permission(%s, %s, %s, %s)" % (
            json.dumps(r.action.items), json.dumps(r.action.estimated_total),
            json.dumps(savings), json.dumps(observation)))
        snap("permission_modal")
        page.wait_for_timeout(beat_pause_ms)
        page.click("#approve" if "granted" in note else "#hold")
        page.evaluate("pennymart.closeModal()")
    elif kind == "add_to_cart":
        if "permitted" in note:
            page.evaluate(f"pennymart.addToCart({json.dumps(r.action.product_id)})")
        snap("cart")


def run_live(env, policy, headed: bool = True, slow_mo: int = 0,
             screenshot_dir: str | None = None, beat_pause_ms: int = 900,
             max_steps: int = 15, policy_label: str = "live policy") -> dict:
    """LIVE mode: the policy decides each turn NOW and the browser mirrors it.
    Same projection as replay — the browser stays a projection of structured
    actions either way; only the source of decisions changes."""
    from playwright.sync_api import sync_playwright

    from shoprl.actions import parse_agent_action

    import tempfile
    workdir = tempfile.mkdtemp(prefix="pennymart-live-")
    if hasattr(env, "catalog"):
        page_items = env.catalog
        store_title = "PennyMart — simulated storefront"
    else:  # WebShopEnvironment: project the fake backend's item list
        page_items = [{"sku": it.asin, "name": it.title, "price": it.price}
                      for it in env.backend.items]
        store_title = "PennyMart × WebShop — simulated storefront"
    html_path = Path(workdir) / "pennymart.html"
    html_path.write_text(render_storefront_html(page_items, title=store_title))

    shots: list[str] = []
    sdir = Path(screenshot_dir) if screenshot_dir else None
    if sdir:
        sdir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headed, slow_mo=slow_mo)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        def snap(name: str) -> None:
            if sdir:
                fp = str(sdir / f"{len(shots):02d}_{name}.png")
                page.screenshot(path=fp)
                shots.append(fp)

        page.goto(f"file://{html_path}")
        opener = env.reset()
        policy.reset(getattr(env, "scenario", None), getattr(env, "idx", None))
        page.evaluate(f"pennymart.bubble('user', {json.dumps(opener)}, "
                      f"{json.dumps('policy: ' + policy_label)})")
        snap("opening")

        obs = env.observe()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action_text = policy.act(obs)
            step = env.execute_text(action_text)
            page.evaluate(f"pennymart.bubble('agent', {json.dumps(action_text)}, "
                          f"{json.dumps(step.note)})")
            r = parse_agent_action(action_text)
            if r.ok:
                _project_action(page, r, step.note, step.observation,
                                env.state.estimated_savings, snap, beat_pause_ms)
            if step.observation:
                page.evaluate(
                    f"pennymart.bubble('user', {json.dumps(step.observation[:400])})")
            obs = step.observation
            done = step.done
            steps += 1
            page.wait_for_timeout(beat_pause_ms if headed else 30)

        snap("final")
        cart_shown = page.text_content("#cartn")
        if headed:
            page.wait_for_timeout(4 * beat_pause_ms)   # hold the final frame
        browser.close()

    out = env.calculate_outcome()
    expected = "1" if env.get_cart() else "0"
    return {"steps": steps, "screenshots": shots, "cart_badge": cart_shown,
            "cart_ok": cart_shown == expected,
            "reward": getattr(out, "total", getattr(out, "score", None)),
            "value_quality": getattr(out, "value_quality",
                                     getattr(out, "score", None)),
            "violation": bool(out.acted_without_permission)}
