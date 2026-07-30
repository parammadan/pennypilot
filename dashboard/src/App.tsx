import { useState } from "react";
import { usePoll } from "./api";
import { Executive, Friction, Investigation, Journey, Platform, Recipes,
         Sql } from "./views";

const TABS = {
  Executive, Journey, Friction, Investigation,
  "Recipes & Lineage": Recipes, Platform, "SQL (admin)": Sql,
} as const;

export default function App() {
  const [tab, setTab] = useState<keyof typeof TABS>("Executive");
  const alerts = usePoll<any[]>("/alerts", 6000) ?? [];
  const View = TABS[tab];
  const crit = alerts.some((a) => a.severity === "CRITICAL");
  return (
    <>
      <header>
        <h1>🧮 PennyData</h1>
        <span className="sub">
          behavioral intelligence for the PennyMart shopping agent · Kafka-fed
        </span>
        <span className="live">● live</span>
      </header>
      {alerts.length > 0 && (
        <div className="alertbar" style={{
          background: crit ? "var(--critical)" : "var(--warning)",
          color: crit ? "#fff" : "#4a3f10" }}>
          {alerts.map((a) =>
            `${a.severity === "CRITICAL" ? "🚨" : "⚠️"} ${a.name}: ${a.detail}`
          ).join("   ·   ")}
        </div>
      )}
      <nav>
        {(Object.keys(TABS) as (keyof typeof TABS)[]).map((t) => (
          <button key={t} className={t === tab ? "on" : ""}
                  onClick={() => setTab(t)}>{t}</button>
        ))}
      </nav>
      <main><View /></main>
    </>
  );
}
