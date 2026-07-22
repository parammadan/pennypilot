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
  :root { --accent:#0a7d4f; --ink:#1a2330; --muted:#68737f; --bg:#f5f6f8; }
  * { box-sizing:border-box; margin:0; }
  body { font:14px/1.45 -apple-system,system-ui,sans-serif; color:var(--ink);
         background:var(--bg); display:grid; grid-template-columns:340px 1fr;
         height:100vh; }
  #chat { background:#fff; border-right:1px solid #e3e6ea; padding:14px;
          overflow-y:auto; }
  #chat h2 { font-size:13px; color:var(--muted); text-transform:uppercase;
             letter-spacing:.06em; margin-bottom:10px; }
  .bubble { max-width:92%; padding:8px 11px; border-radius:12px; margin:6px 0;
            white-space:pre-wrap; word-break:break-word; font-size:13px; }
  .user { background:#eef1f5; }
  .agent { background:var(--accent); color:#fff; margin-left:auto;
           font-family:ui-monospace,Menlo,monospace; font-size:12px; }
  .note { color:var(--muted); font-size:11px; margin:2px 0 8px; }
  #store { display:flex; flex-direction:column; overflow:hidden; }
  header { display:flex; gap:10px; align-items:center; padding:12px 18px;
           background:#fff; border-bottom:1px solid #e3e6ea; }
  header h1 { font-size:17px; margin-right:auto; }
  header h1 em { color:var(--accent); font-style:normal; }
  .sim { font-size:11px; color:#b25a00; background:#fff3e2; padding:3px 8px;
         border-radius:999px; }
  #search { flex:0 0 320px; padding:8px 10px; border:1px solid #cfd5db;
            border-radius:8px; font-size:13px; }
  #cart { font-weight:600; } #cart b { color:var(--accent); }
  #grid { padding:16px; display:grid; gap:12px; overflow-y:auto;
          grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); }
  .card { background:#fff; border:1px solid #e3e6ea; border-radius:12px;
          padding:12px; transition:box-shadow .2s,border-color .2s; }
  .card.hit { border-color:var(--accent); box-shadow:0 4px 14px rgba(10,125,79,.18); }
  .card.selected { outline:3px solid var(--accent); }
  .card h3 { font-size:13px; } .card .price { font-size:17px; font-weight:700;
             color:var(--accent); margin:4px 0; }
  .card .specs { color:var(--muted); font-size:12px; }
  #banner { display:none; padding:10px 18px; background:#e7f6ee;
            color:#0a7d4f; font-weight:600; }
  #modal { display:none; position:fixed; inset:0; background:rgba(20,26,34,.45);
           align-items:center; justify-content:center; }
  #modal .box { background:#fff; border-radius:14px; padding:22px; width:430px; }
  #modal h3 { margin-bottom:8px; } #modal ul { margin:8px 0 8px 18px; }
  #modal .savings { color:var(--accent); font-weight:600; }
  #modal .reply { margin-top:10px; font-style:italic; color:var(--muted); }
  #modal .actions { margin-top:14px; display:flex; gap:8px; justify-content:flex-end; }
  #modal button { padding:8px 14px; border-radius:8px; border:1px solid #cfd5db;
                  background:#fff; font-weight:600; }
  #modal button.approve { background:var(--accent); color:#fff; border:none; }
</style></head><body>
<aside id="chat"><h2>Conversation</h2></aside>
<main id="store">
  <header><h1><em>Penny</em>Mart</h1><span class="sim">SIMULATED — no real purchases</span>
    <input id="search" placeholder="Search products…"><div id="cart">Cart: <b>0</b></div>
  </header>
  <div id="banner"></div><div id="grid"></div>
</main>
<div id="modal"><div class="box"><h3>Permission to add to cart?</h3>
  <div id="modal-body"></div><div class="reply" id="modal-reply"></div>
  <div class="actions"><button id="hold">Not yet</button>
  <button class="approve" id="approve">Approve</button></div></div></div>
<script>
const PRODUCTS = __PRODUCTS__;
const grid = document.getElementById("grid");
const chat = document.getElementById("chat");
function card(p){ return `<div class="card" data-sku="${p.sku}"><h3>${p.name}</h3>
  <div class="price">$${p.price.toFixed(2)}</div>
  <div class="specs">${p.ram_gb!==undefined?`${p.ram_gb}GB RAM · ${p.weight_lbs} lbs · ${p.battery_hrs} h · ${p.brand}`:""}</div>
  <div class="specs">${p.sku}</div></div>`; }
function show(list){ grid.innerHTML = list.map(card).join(""); }
window.pennymart = {
  all(){ show(PRODUCTS); },
  bubble(role, text, note){ const d=document.createElement("div");
    d.className="bubble "+role; d.textContent=text; chat.appendChild(d);
    if(note){ const n=document.createElement("div"); n.className="note";
      n.textContent=note; chat.appendChild(n);} chat.scrollTop=chat.scrollHeight; },
  results(skus){ const set=new Set(skus);
    const hits=PRODUCTS.filter(p=>set.has(p.sku));
    show(hits); hits.forEach(p=>document.querySelector(`[data-sku="${p.sku}"]`)
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
  addToCart(sku){ document.querySelector("#cart b").textContent="1";
    const bn=document.getElementById("banner");
    bn.textContent=`✓ ${sku} added to cart (simulated) — with explicit permission`;
    bn.style.display="block"; },
};
pennymart.all();
</script></body></html>""".replace("__TITLE__", title).replace("__PRODUCTS__", products)


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
                    page.fill("#search", "")
                    page.type("#search", r.action.query, delay=18)
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
        cart_shown = page.text_content("#cart b")
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
        page.fill("#search", "")
        page.type("#search", r.action.query, delay=18)
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
        cart_shown = page.text_content("#cart b")
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
