import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { actions, getState, initBridge, subscribe, type Product } from "./bridge";

initBridge();

const HUE: Record<string, string> = {
  Asus: "#2a78d6", Dell: "#0a7d4f", Lenovo: "#c0392b", HP: "#00838f",
  Apple: "#555", Framework: "#eb6834", Razer: "#3fae29", Acer: "#8e44ad" };

const Art = ({ p, w = 160 }: { p: Product; w?: number }) => {
  const c = HUE[p.brand] ?? "#6b7785";
  return (
    <svg viewBox="0 0 120 80" width={w} height={(w * 80) / 120}>
      <rect x="24" y="12" width="72" height="46" rx="4" fill="#fff"
            stroke={c} strokeWidth="3" />
      <rect x="30" y="18" width="60" height="34" rx="2" fill={c} opacity=".16" />
      <path d="M14 62 h92 l-6 8 h-80 z" fill={c} opacity=".85" />
      <rect x="52" y="62" width="16" height="3" rx="1.5" fill="#fff" opacity=".7" />
    </svg>
  );
};

const rate = (p: Product) => Math.max(3.2, Math.min(5,
  3.4 + ((p.battery_hrs ?? 10) / 20) * 0.9 + ((p.ram_gb ?? 8) >= 32 ? 0.5 : 0)
  + (0.4 - ((p.weight_lbs ?? 4) - 2) / 12)));
const Stars = ({ p }: { p: Product }) => {
  const r = rate(p), full = Math.round(r);
  const n = 40 + Math.round((p.price ?? 500) % 900);
  return (
    <div className="rateline">
      <span className="st">{"★".repeat(full)}{"☆".repeat(5 - full)}</span>
      <span className="n">{r.toFixed(1)} · {n.toLocaleString()} ratings</span>
    </div>
  );
};
const Price = ({ v }: { v: number }) => {
  const [w, f] = v.toFixed(2).split(".");
  return (
    <div className="price"><span className="cur">$</span>
      <span className="whole">{(+w).toLocaleString()}</span>
      <span className="frac">{f}</span></div>
  );
};

export default function App() {
  const s = useSyncExternalStore(subscribe, getState);
  const [q, setQ] = useState("");
  const [say, setSay] = useState("");
  const [drawer, setDrawer] = useState(false);
  const msgsRef = useRef<HTMLDivElement>(null);
  const fb = (window.__feedback ?? []);
  useEffect(() => {
    msgsRef.current?.scrollTo(0, msgsRef.current.scrollHeight);
  }, [s.bubbles.length, s.busy]);

  const send = () => { actions.submitChat(say); setSay(""); };
  let agentIdx = -1;

  return (
    <div className="layout">
      <aside id="chat">
        <h2 id="chathead">Penny{" "}
          {s.modelTag && <span id="modeltag">{s.modelTag}</span>}</h2>
        <div id="msgs" ref={msgsRef}>
          {s.bubbles.map((b) => {
            const i = b.role === "agent" ? ++agentIdx : -1;
            const my = fb.find((f) => f.i === i);
            return (
              <div key={b.id}>
                <div className={`bubble ${b.role}`}>
                  <div className="av">{b.role === "agent" ? "🛍️" : "🧑"}</div>
                  <div className="tx">{b.text}
                    <span className="tm">{b.time}</span></div>
                </div>
                {b.role === "agent" && (
                  <div className="fb">
                    <span style={{ opacity: my?.vote === "up" ? 1 : 0.4 }}
                          onClick={() => actions.vote(i, "up", b.text)}>👍</span>
                    <span style={{ opacity: my?.vote === "down" ? 1 : 0.4 }}
                          onClick={() => actions.vote(i, "down", b.text)}>👎</span>
                  </div>
                )}
                {b.note && <div className="note">{b.note}</div>}
              </div>
            );
          })}
          {s.busy && <div id="typing"><span /><span /><span /></div>}
        </div>
        <div id="saybar">
          <input id="say" placeholder={s.hint} autoComplete="off" value={say}
                 disabled={s.busy} onChange={(e) => setSay(e.target.value)}
                 onKeyDown={(e) => e.key === "Enter" && send()} />
          <button id="send" onClick={send}>Send</button>
        </div>
      </aside>

      <main id="store">
        <header>
          <span className="logo">Penny<em>Mart</em></span>
          <div id="search">
            <input id="q" placeholder="Search PennyMart" value={q}
                   onChange={(e) => setQ(e.target.value)}
                   onKeyDown={(e) => e.key === "Enter" && actions.search(q)} />
            <div className="go" onClick={() => actions.search(q)}>🔍</div>
          </div>
          <span className="cart" id="cartbtn"
                onClick={() => { actions.openCart(); setDrawer(true); }}>
            🛒 Cart <b id="cartn">{s.cart.length}</b></span>
        </header>
        <div id="subnav"><span>All</span><span>Laptops</span><span>Deals</span></div>
        <div id="resultbar">{s.resultBar}</div>
        <div id="grid">
          {s.grid === null ? (
            <div id="empty"><div className="big">🛍️</div>
              <h2>Your shopping assistant is ready</h2>
              <p>Tell Penny what you're looking for in the chat — a budget, a
                brand, how much RAM. She'll ask what she needs, then find the
                cheapest option that fits and show it here.</p></div>
          ) : s.grid.map((p, gi) => (
            <div key={p.sku} data-sku={p.sku}
                 className={`card${p.sku === s.selectedSku ? " selected" : ""}`}
                 onMouseOver={() => actions.cardHover(p, gi + 1)}
                 onClick={() => actions.cardClick(p, gi + 1)}>
              <div className="thumb"><Art p={p} /></div>
              <div className="info">
                {p.sku === s.bestSku &&
                  <div className="badge-best"><em>Best</em> Match</div>}
                <h3>{p.name}</h3>
                <Stars p={p} /><Price v={p.price} />
                <div className="specs">{p.ram_gb}GB RAM &nbsp;|&nbsp;
                  {p.weight_lbs} lb &nbsp;|&nbsp; {p.battery_hrs} hr battery
                  &nbsp;|&nbsp; {p.brand}</div>
                <div className="deliv">FREE delivery <b>tomorrow</b></div>
                <div className="stock">In stock</div>
                <div className="buy" onClick={(e) => {
                  e.stopPropagation(); actions.buyClick(p, gi + 1); }}>Add to Cart</div>
                <div className="sku">{p.sku}</div>
              </div>
            </div>
          ))}
        </div>
      </main>

      <div id="drawer" className={drawer ? "open" : ""}>
        <button className="close" onClick={() => setDrawer(false)}>✕</button>
        <h3>🛒 Your Cart</h3>
        <div>
          {s.cart.length === 0 && (
            <div className="empty">Your cart is empty.<br />
              Penny adds items only with your permission.</div>)}
          {s.cart.map((p) => (
            <div className="item" key={p.sku}>
              <Art p={p} w={64} />
              <div><div style={{ fontWeight: 600 }}>{p.name}</div>
                <div className="sub">{p.sku}</div></div>
              <div className="right">
                <div style={{ fontWeight: 700 }}>${p.price.toFixed(2)}</div>
                <button className="rm"
                        onClick={() => actions.removeFromCart(p.sku)}>
                  Remove</button>
              </div>
            </div>
          ))}
        </div>
        <div className="subtotal"><span>Subtotal</span>
          <span>${s.cart.reduce((a, p) => a + p.price, 0).toFixed(2)}</span></div>
        <div className="checkout">Proceed to checkout (simulated)</div>
      </div>

      {s.detail && (
        <div id="detail" onClick={actions.closeDetail}>
          <div className="box" onClick={(e) => e.stopPropagation()}>
            <div><Art p={s.detail} w={200} /></div>
            <div>
              <h3>{s.detail.name}</h3>
              <Stars p={s.detail} /><Price v={s.detail.price} />
              <table><tbody>
                <tr><td>RAM</td><td>{s.detail.ram_gb} GB</td></tr>
                <tr><td>Battery</td><td>{s.detail.battery_hrs} hours</td></tr>
                <tr><td>Weight</td><td>{s.detail.weight_lbs} lb</td></tr>
                <tr><td>Brand</td><td>{s.detail.brand}</td></tr>
                <tr><td>SKU</td><td>{s.detail.sku}</td></tr>
              </tbody></table>
              <div className="deliv">FREE delivery <b>tomorrow</b> ·{" "}
                <span className="stock">In stock</span></div>
              <div className="note">Ask Penny to add it — nothing enters your
                cart without your explicit permission.</div>
            </div>
          </div>
        </div>
      )}

      {s.modal && (
        <div id="modal">
          <div className="box">
            <h3>Add to your cart?</h3>
            <ul>{s.modal.items.map((x) => <li key={x}>{x}</li>)}</ul>
            <div>Estimated total: <b>${s.modal.total}</b></div>
            {s.modal.savings != null && (
              <div className="savings">Estimated savings: ${s.modal.savings} vs
                priciest matching option</div>)}
            {s.modal.reply && <div className="reply">User: “{s.modal.reply}”</div>}
            <div className="actions">
              <button id="hold" onClick={actions.hold}>Not now</button>
              <button id="approve" className="approve"
                      onClick={actions.approve}>Add to Cart</button>
            </div>
          </div>
        </div>
      )}

      {s.toast && <div id="toast">{s.toast}</div>}
    </div>
  );
}
