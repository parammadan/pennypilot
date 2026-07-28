"""PennyData automated analysis — the investigation flow as a runnable report.

Executes the discipline end to end and WRITES THE FINDINGS:
  1. validate the data (before reading any metric)
  2. headline behavioral metrics, definitions attached
  3. funnel: where do sessions drop?
  4. cohort comparison (labels = model versions / arms)
  5. ranked slices: WHERE does behavior deviate?
  6. evidence: example sessions per finding (the failure-analysis handoff)
  7. hypotheses WITH caveats — the report never claims causation; it names
     what to check next (traffic mix, latency, tools, data) per finding

Output = markdown for humans + JSON for pipelines.
"""
from __future__ import annotations

import time

from shoprl.platform.behavior import (data_quality, funnel_by, metrics,
                                      session_features, slice_report)
from shoprl.platform.store import PlatformStore

_CAVEATS = ("observed behavior, not proven cause — before blaming the model, "
            "check: traffic/persona composition of the slice, data quality in "
            "that slice, latency/tool health, and whether the pattern holds "
            "within matched sub-segments")


def analyze(store: PlatformStore, min_support: int = 30,
            evidence_per_finding: int = 2) -> dict:
    sessions = session_features(store)
    dq = data_quality(store)
    m = metrics(sessions)
    labels = sorted({s["label"] for s in sessions})

    # cohort table (labels are whatever the traffic was stamped with:
    # model versions, arms, personas)
    cohorts = {}
    for lb in labels:
        group = [s for s in sessions if s["label"] == lb]
        gm = metrics(group)
        cohorts[lb] = {"sessions": len(group),
                       "abandonment": gm["abandonment"]["value"],
                       "reformulation": gm["reformulation"]["value"],
                       "repeat": gm["repeat_question"]["value"],
                       "invalid": gm["invalid_action"]["value"],
                       "violation": gm["violation"]["value"]}

    findings = []
    for metric in ("abandoned", "reformulated", "repeated", "violation"):
        rep = slice_report(sessions, metric=metric, min_support=min_support,
                           top=3)
        candidates = list(rep["top_slices"])
        # cohort (label) slices are first-class: labels ARE the model
        # versions/arms under comparison, so an adverse label never gets
        # crowded out by larger-deviation minor slices
        seen = {tuple(sorted(c["slice"].items())) for c in candidates}
        for c in slice_report(sessions, metric=metric,
                              min_support=min_support, top=20)["top_slices"]:
            if set(c["slice"]) == {"label"} and                     tuple(sorted(c["slice"].items())) not in seen:
                candidates.append(c)
        for c in candidates:
            if c["deviation"] <= 0.03:      # only adverse, material deviations
                continue
            sids = [s["session_id"] for s in sessions
                    if all(str(s[k]) == str(v) for k, v in c["slice"].items())
                    and s[metric]][:evidence_per_finding]
            evidence = []
            for sid in sids:
                turns = store.db.execute(
                    "SELECT agent, observation FROM turns WHERE session_id=?"
                    " ORDER BY i LIMIT 4", (sid,)).fetchall()
                evidence.append({"session_id": sid,
                                 "excerpt": [{"agent": a[:120],
                                              "user": (o or "")[:120]}
                                             for a, o in turns]})
            findings.append({
                "metric": metric, "slice": c["slice"],
                "support": c["support"], "value": c["value"],
                "baseline": c["baseline"], "deviation": c["deviation"],
                "hypothesis": (f"{metric} is elevated in {c['slice']} "
                               f"({c['value']:.0%} vs {c['baseline']:.0%} "
                               f"baseline, n={c['support']})"),
                "caveats": _CAVEATS, "evidence_sessions": evidence})
    findings.sort(key=lambda f: -(f["deviation"] * f["support"]))

    return {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "data_quality": dq, "metrics": m,
            "funnel": funnel_by(None, store),
            "cohorts": cohorts, "findings": findings[:8]}


def to_markdown(r: dict) -> str:
    L = [f"# PennyData analysis — {r['generated']}", ""]
    dq = r["data_quality"]
    L += [f"## 1. Data quality: **{dq['verdict']}**",
          "", "| check | count |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in dq["checks"].items()]
    if dq["verdict"] != "clean":
        L += ["", "⚠ **STOP: fix instrumentation before reading further.**"]
    m = r["metrics"]
    L += ["", f"## 2. Headline metrics ({m['sessions']} sessions)", "",
          "| metric | value | population |", "|---|---|---|"]
    for k, v in m.items():
        if isinstance(v, dict):
            L.append(f"| {k} | {v['value']} | {v['denominator']} |")
    L += ["", "## 3. Funnel", "", "| stage | sessions | conv. from prev |",
          "|---|---|---|"]
    for st in r["funnel"]["stages"]:
        L.append(f"| {st['stage']} | {st['sessions']} | "
                 f"{st['conversion_from_prev'] or '—'} |")
    L += ["", "## 4. Cohorts", "",
          "| label | n | abandon | reformulate | repeat | invalid | viol |",
          "|---|---|---|---|---|---|---|"]
    for lb, c in r["cohorts"].items():
        L.append(f"| {lb} | {c['sessions']} | {c['abandonment']} | "
                 f"{c['reformulation']} | {c['repeat']} | {c['invalid']} | "
                 f"{c['violation']} |")
    L += ["", "## 5. Findings (ranked; hypotheses, not causes)", ""]
    if not r["findings"]:
        L.append("No slice deviates adversely above the support threshold.")
    for i, f in enumerate(r["findings"], 1):
        L += [f"### Finding {i}: {f['hypothesis']}",
              f"- slice: `{f['slice']}` · support {f['support']}",
              f"- caveats: {f['caveats']}",
              f"- evidence sessions: "
              f"{', '.join(e['session_id'] for e in f['evidence_sessions'])}",
              ""]
    return "\n".join(L)
