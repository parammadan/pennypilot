// PennyData API client — same-origin in prod (served by the platform),
// direct to :8770 in `npm run dev`.
import { useEffect, useState } from "react";

const BASE = import.meta.env.DEV ? "http://localhost:8770" : "";

export async function get<T = any>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

/** Poll an endpoint on an interval; undefined until first response. */
export function usePoll<T = any>(path: string, ms = 4000): T | undefined {
  const [data, setData] = useState<T>();
  useEffect(() => {
    let live = true;
    const tick = () =>
      get<T>(path).then((d) => live && setData(d)).catch(() => {});
    tick();
    const id = setInterval(tick, ms);
    return () => { live = false; clearInterval(id); };
  }, [path, ms]);
  return data;
}

export const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
// validated dataviz palette (light/dark steps)
export const CAT = dark
  ? ["#3987e5", "#d95926", "#199e70"]
  : ["#2a78d6", "#eb6834", "#1baf7a"];
export const pct = (v: number | null | undefined, d = 1) =>
  v == null ? "–" : `${(100 * v).toFixed(d)}%`;
