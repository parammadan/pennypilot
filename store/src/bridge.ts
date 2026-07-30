/** The driver contract. demo_human.py talks to this page exclusively through
 * window.pennymart.* calls and the __human / __uiev / __feedback globals —
 * this bridge preserves that contract exactly, so the Playwright harness,
 * the clickstream pipeline, and every existing test keep working.
 */
export type Product = {
  sku: string; name: string; price: number; brand: string;
  ram_gb?: number; weight_lbs?: number; battery_hrs?: number;
};

declare global {
  interface Window {
    PRODUCTS: Product[];
    pennymart: any;
    __human: string | null;
    __uiev: any[];
    __feedback: { i: number; vote: string; text: string }[];
    __fbN: number;
    __closed?: boolean;
  }
}

export type Bubble = { id: number; role: "user" | "agent"; text: string;
                       note?: string; time: string };

export type StoreState = {
  bubbles: Bubble[];
  grid: Product[] | null;          // null = welcome screen
  resultBar: string;
  bestSku: string | null;
  selectedSku: string | null;
  hint: string;
  demoHint: string;
  busy: boolean;
  cart: Product[];
  toast: string;
  modal: null | { items: string[]; total: string; savings: string | null;
                  reply: string };
  detail: Product | null;
  modelTag: string;
};

let state: StoreState = {
  bubbles: [], grid: null, resultBar: "", bestSku: null, selectedSku: null,
  hint: "…", demoHint: "", busy: false, cart: [], toast: "", modal: null,
  detail: null, modelTag: "",
};
let listeners: (() => void)[] = [];
const emitChange = () => listeners.forEach((l) => l());
export const subscribe = (l: () => void) => {
  listeners.push(l);
  return () => { listeners = listeners.filter((x) => x !== l); };
};
export const getState = () => state;
const set = (patch: Partial<StoreState>) => {
  state = { ...state, ...patch };
  emitChange();
};

export const push = (type: string, target: string, meta: any = {}) =>
  window.__uiev.push({ type, target, meta, ts: Date.now() / 1000 });

const products = () => window.PRODUCTS ?? [];
const bySku = (sku: string) =>
  products().find((p) => p.sku === (sku || "").toUpperCase());

let bubbleId = 0;
let toastTimer: any = null;

export function initBridge() {
  window.__human = null;
  window.__uiev = window.__uiev ?? [];
  window.__feedback = window.__feedback ?? [];
  window.__fbN = window.__fbN ?? 0;

  window.pennymart = {
    all() { set({ grid: products(),
                  resultBar: `Showing ${products().length} products` }); },
    welcome() { set({ grid: null, resultBar: "" }); },
    demoHint(text: string) { set({ demoHint: text }); },
    bubble(role: "user" | "agent", text: string, note?: string) {
      const time = new Date().toLocaleTimeString([], {
        hour: "2-digit", minute: "2-digit" });
      const b: Bubble = { id: bubbleId++, role, text, note, time };
      if (role === "agent") window.__fbN = (window.__fbN ?? 0) + 1;
      set({ bubbles: [...state.bubbles, b] });
    },
    results(skus: string[]) {
      const hits = skus.map(bySku).filter(Boolean) as Product[];
      set({ grid: hits, bestSku: skus[0] ?? null,
            resultBar: `${hits.length} results · sorted by price (lowest first)` });
    },
    select(sku: string) { set({ selectedSku: (sku || "").toUpperCase() }); },
    permission(items: string[], total: string, savings: string | null,
               reply: string) {
      set({ modal: { items, total, savings, reply } });
    },
    closeModal() { set({ modal: null }); },
    hint(h: string) { set({ hint: h }); },
    busy(on: boolean) { set({ busy: !!on }); },
    modelTag(t: string) { set({ modelTag: t }); },
    addToCart(sku: string) {
      const p = bySku(sku);
      const cart = p ? [...state.cart, p] : state.cart;
      set({ cart,
            toast: `✓ Added to cart — ${p ? p.name : sku} (simulated, with your explicit permission)` });
      clearTimeout(toastTimer);
      toastTimer = setTimeout(() => set({ toast: "" }), 4200);
    },
  };
}

// ---- user-side actions (these produce the clickstream) ----------------------
export const actions = {
  submitChat(v: string) {
    if (!v.trim()) return;
    push("input", "chat", { len: v.length });
    window.__human = v.trim();
  },
  approve() { push("modal", "approve"); window.__human = "__yes__"; },
  hold() { window.__human = "__no__"; },
  vote(i: number, voteVal: "up" | "down", text: string) {
    window.__feedback = (window.__feedback ?? []).filter((f) => f.i !== i);
    window.__feedback.push({ i, vote: voteVal, text: text.slice(0, 200) });
    emitChange();
  },
  search(q: string) {
    push("input", "search", { q: q.slice(0, 80) });
    if (!q.trim()) { window.pennymart.all(); return; }
    const hits = products().filter((p) =>
      `${p.name} ${p.brand} ${p.sku}`.toLowerCase().includes(q.toLowerCase()));
    set({ grid: hits, bestSku: null,
          resultBar: `${hits.length} results for “${q}”` });
  },
  // position = 1-based grid slot at interaction time — "clicked LAP-882,
  // shown in position 2" is a different fact than "clicked LAP-882"
  cardClick(p: Product, position: number) {
    push("click", `card:${p.sku}`, { position });
    push("click", `detail:${p.sku}`, { position });
    set({ detail: p });
  },
  cardHover: (() => {
    let last = 0;
    return (p: Product, position: number) => {
      const now = Date.now();
      if (now - last > 800) { last = now;
        push("hover", `card:${p.sku}`, { position }); }
    };
  })(),
  buyClick(p: Product, position: number) {
    // recorded intent — carting itself stays behind Penny's permission gate
    push("click", `card:${p.sku}`, { buy: true, position });
  },
  openCart() { push("click", "cart_open"); },
  removeFromCart(sku: string) {
    push("click", `cart_remove:${sku}`);
    const i = state.cart.findIndex((p) => p.sku === sku);
    if (i >= 0) {
      const cart = [...state.cart];
      cart.splice(i, 1);
      set({ cart });
    }
  },
  closeDetail() { set({ detail: null }); },
  setModelTag(t: string) { set({ modelTag: t }); },
};
