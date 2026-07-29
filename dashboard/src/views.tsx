import { useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid,
         ResponsiveContainer } from "recharts";
import { usePoll, get, pct, CAT } from "./api";
import { Card, HBars, Note, Table, Tile } from "./ui";

// ---------------- Executive ------------------------------------------------
export function Executive() {
  const b = usePoll<any>("/behavior");
  const st = usePoll<any>("/stats");
  const ts = usePoll<any>("/timeseries?metric=abandoned&bucket=300", 8000);
  const m = b?.metrics;
  const fresh = b?.data_quality?.verdict ?? "…";
  let up = 0, dn = 0;
  for (const v of Object.values<any>(st?.per_label ?? {})) {
    up += v.thumbs_up; dn += v.thumbs_down;
  }
  const series = (ts?.series ?? []).map((p: any) => ({
    t: new Date(p.t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    rate: p.rate, n: p.n,
  }));
  return (
    <>
      <div className="tiles">
        <Tile label="episodes" value={m?.sessions ?? "–"} />
        <Tile label="violation rate" tone={m?.violation?.value ? "bad" : "good"}
              value={pct(m?.violation?.value)} />
        <Tile label="task-relevant abandonment" value={pct(m?.abandonment?.value)} />
        <Tile label="recommendation CTR" value={pct(m?.recommendation_ctr?.value)} />
        <Tile label="feedback 👍/👎" value={`${up} / ${dn}`} />
        <Tile label="data quality" tone={fresh === "clean" ? "good" : "bad"}
              value={fresh} />
      </div>
      <Card title={<>abandonment over time {ts?.anomalies?.length
          ? <span className="bad">⚠ {ts.anomalies.length} anomaly</span> : null}</>} wide>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={series}>
            <CartesianGrid stroke="var(--grid)" vertical={false} />
            <XAxis dataKey="t" tick={{ fontSize: 10 }} stroke="var(--axis)" />
            <YAxis tickFormatter={(v) => pct(v, 0)} tick={{ fontSize: 10 }}
                   stroke="var(--axis)" width={44} />
            <Tooltip formatter={(v: any) => pct(v)}
                     contentStyle={{ background: "var(--surface)",
                                     border: "1px solid var(--border)" }} />
            <Line dataKey="rate" stroke={CAT[1]} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <Note>each point carries its own n; low-n buckets are never flagged as anomalies</Note>
      </Card>
    </>
  );
}

// ---------------- Journey (milestone funnel) --------------------------------
export function Journey() {
  const f = usePoll<any>("/funnel2", 8000);
  const [mv, setMv] = useState<string>("");
  if (!f) return <Note>loading…</Note>;
  const models = Object.keys(f.by_model_version ?? {});
  const fun = mv ? f.by_model_version[mv] : f.overall;
  return (
    <>
      <div className="filters">cohort{" "}
        <select value={mv} onChange={(e) => setMv(e.target.value)}>
          <option value="">all ({f.overall.sessions})</option>
          {models.map((m) => <option key={m}>{m}</option>)}
        </select>
        <span className="note">{f.excluded_no_goal} sessions without a structured
          goal are excluded and counted</span>
      </div>
      <Card title="conversation-aware funnel (milestones from authoritative events)" wide>
        <HBars color={CAT[2]} rows={fun.stages.map((s: any) => ({
          k: s.stage, v: s.reached,
          txt: `${s.reached}/${s.eligible}` +
            (s.conversion_from_prev != null ? ` (${pct(s.conversion_from_prev, 0)})` : ""),
        }))} />
      </Card>
      <Card title="failure reason codes between stages" wide>
        <Table cols={["failed stage", "reason", "sessions"]}
          rows={Object.entries<any>(fun.failure_reasons ?? {}).flatMap(
            ([stage, rs]) => Object.entries<any>(rs).map(
              ([reason, n]) => [stage, <code key={reason}>{reason}</code>, n]))} />
      </Card>
    </>
  );
}

// ---------------- Friction ---------------------------------------------------
export function Friction() {
  const m = usePoll<any>("/friction", 8000);
  if (!m) return <Note>loading…</Note>;
  const metric = (x: any, fmt: (v: any) => string) => (
    <Card key={x.metric} title={x.metric} wide>
      <div className="tiles" style={{ gridTemplateColumns: "repeat(4,1fr)" }}>
        {Object.entries<any>(x.by_model_version ?? {}).map(([k, v]) => (
          <Tile key={k} label={k + (v.low_support ? " (low n)" : "")}
                value={fmt(v.value)} />
        ))}
      </div>
      <Note>numerator: {x.numerator} · denominator: {x.denominator} ·
        exclusions: {x.exclusions}</Note>
    </Card>
  );
  return (
    <>
      {metric(m.premature_search_rate, (v) => pct(v))}
      {metric(m.customer_correction_rate, (v) => pct(v))}
      {metric(m.turns_to_goal_understood,
              (v) => (v == null ? "–" : `${v} turns`))}
      <Note>turns-to-goal is a flawed proxy when corrections reveal constraints
        (customer labor runs the clock faster) — see docs 2026-07-28.</Note>
    </>
  );
}

// ---------------- Investigation ---------------------------------------------
export function Investigation() {
  const sl = usePoll<any>("/slices?metric=abandoned&top=8", 10000);
  const at = usePoll<any>("/attribution", 10000);
  const [sess, setSess] = useState<any | null>(null);
  const open = async (sid: string) =>
    setSess((await get<any>(`/sessions?limit=200`)).sessions
      .find((x: any) => x.session_id === sid) ?? null);
  const counts: Record<string, number> = {};
  for (const a of at ?? []) counts[a.primary_category] =
    (counts[a.primary_category] ?? 0) + 1;
  return (
    <>
      <Card title="where behavior deviates (ranked slices — hypotheses, not causes)" wide>
        <Table cols={["slice", "n", "value", "baseline", "deviation"]}
          rows={(sl?.top_slices ?? []).map((c: any) => [
            <code key="s">{JSON.stringify(c.slice)}</code>, c.support,
            pct(c.value), pct(c.baseline),
            <span key="d" className={c.deviation > 0 ? "bad" : "good"}>
              {c.deviation > 0 ? "▲" : "▼"} {pct(Math.abs(c.deviation))}</span>,
          ])} />
        <Note>{sl?.note} · {sl?.suppressed_low_n} low-n slices suppressed</Note>
      </Card>
      <Card title="failure attribution (deterministic rules, evidence-cited, abstains)">
        <HBars color={CAT[1]} rows={Object.entries(counts).map(
          ([k, v]) => ({ k, v }))} />
      </Card>
      <Card title="attributed sessions — click for transcript + evidence">
        <Table cols={["session", "category", "conf", "evidence"]}
          onRow={(i) => open((at ?? [])[i].session_id)}
          rows={(at ?? []).slice(0, 12).map((a: any) => [
            a.session_id, a.primary_category, a.confidence,
            (a.evidence_event_ids ?? []).length + " events"])} />
      </Card>
      {sess && (
        <div className="modal" onClick={() => setSess(null)}>
          <div className="box" onClick={(e) => e.stopPropagation()}>
            <h2>{sess.session_id}</h2>
            {sess.turns.map((t: any, i: number) => (
              <div key={i}>
                <div className="turn agent">🤖 {t.agent}</div>
                {t.observation && <div className="turn user">🧑 {t.observation}</div>}
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}

// ---------------- Recipes + lineage ------------------------------------------
export function Recipes() {
  const recipes = usePoll<any[]>("/recipes", 15000);
  const lin = usePoll<any[]>("/lineage", 15000);
  return (
    <>
      {(recipes ?? []).map((r) => (
        <Card key={r.recipe_id} wide title={<>
          dataset proposal: {r.recipe_id}{" "}
          <span className={r.approval_status === "approved" ? "good" : "bad"}>
            [{r.approval_status}]</span></>}>
          <Note>target: {r.target_failure} · labeling: {r.labeling_policy} ·
            approved by: {r.approved_by || "—"}</Note>
          <p style={{ fontSize: 13 }}>{r.finding}</p>
          <Note>mixture: {JSON.stringify(r.mixture)} · eval slices:{" "}
            {(r.evaluation_slices ?? []).join(", ")} · risks:{" "}
            {(r.regression_risks ?? []).join("; ")}</Note>
        </Card>
      ))}
      <Card title="lineage — finding → dataset → training → evaluation" wide>
        <Table cols={["recipe", "dataset sha", "sequences", "dropped (dup/invalid)", "created"]}
          rows={(lin ?? []).map((l) => [
            l.recipe_id, <code key="s">{l.dataset_sha}</code>,
            l.manifest.sequences,
            `${l.manifest.dropped_duplicates}/${l.manifest.dropped_outcome_invalid}`,
            new Date(l.created * 1000).toLocaleString()])} />
        <Note>v1 → sft7b-C: guardrail breach (action grammar lost), ROLLED BACK ·
          v2 → sft7b-C2: primary met (premature 24% vs 55%), cannot-redirect
          guardrail breached → NOT SHIPPED. Two catches, zero bad ships.</Note>
      </Card>
    </>
  );
}

// ---------------- SQL ---------------------------------------------------------
export function Sql() {
  const [q, setQ] = useState(
    "SELECT model_version, COUNT(*) episodes, SUM(violation) violations FROM episodes GROUP BY 1");
  const [out, setOut] = useState<any>(null);
  const run = async () => {
    try { setOut(await get(`/query?sql=${encodeURIComponent(q)}`)); }
    catch (e: any) { setOut({ error: String(e) }); }
  };
  return (
    <Card title="self-service SQL (SELECT-only) — episodes, turns, ui_events, semantic_events, lineage" wide>
      <textarea rows={3} value={q} onChange={(e) => setQ(e.target.value)} />
      <div><button onClick={run}>Run query</button></div>
      {out?.error && <Note>⚠ {out.error}</Note>}
      {out?.columns && <Table cols={out.columns}
        rows={out.rows.map((r: any[]) => r.map(String))} />}
    </Card>
  );
}
