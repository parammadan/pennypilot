"""Conversation-aware funnel + friction metrics — computed ONLY from
authoritative semantic events and deterministic outcomes (never from action
strings or text regexes).

Eligibility: a session enters this analysis only if it carries a structured
CustomerGoal (so "required constraints" is defined truth). Sessions without
one are counted as excluded, never silently dropped.
"""
from __future__ import annotations

import json
import time

from shoprl.platform.store import PlatformStore

STAGES = ["session_started", "goal_understood", "valid_search",
          "grounded_recommendation", "permission_requested",
          "permission_granted", "correct_cart", "task_satisfied"]
MIN_SUPPORT = 10


def _sessions(store: PlatformStore) -> list[dict]:
    """One record per goal-carrying session: goal truth, semantic-event
    milestones, outcome."""
    db = store.db
    out = {}
    for sid, goal, outcome, mv, fam in db.execute(
            "SELECT session_id, goal, outcome, COALESCE(model_version, label),"
            " COALESCE(scenario_family, '') FROM episodes"):
        if not goal:
            continue
        g = json.loads(goal)
        required = set(g.get("must_have_constraints", {})) | {"budget"}
        out[sid] = {"session_id": sid, "model_version": mv or "",
                    "scenario_family": fam,
                    "required": required, "revealed": set(),
                    "revealed_at": {}, "searches": [], "recs": [],
                    "perm_requested": False, "perm_granted": False,
                    "corrections": [], "abandoned": False,
                    "outcome": json.loads(outcome) if outcome else None,
                    "event_ids": {"premature": [], "correction": [],
                                  "bad_rec": []},
                    "max_ts": 0.0}
    for sid, eid, typ, ti, attrs, ts in db.execute(
            "SELECT session_id, event_id, type, turn_index, attributes, ts"
            " FROM semantic_events ORDER BY ts, turn_index"):
        s = out.get(sid)
        if s is None:
            continue
        a = json.loads(attrs or "{}")
        s["max_ts"] = max(s["max_ts"], ts or 0)
        if typ == "constraint_revealed":
            s["revealed"].add(a["key"])
            s["revealed_at"].setdefault(a["key"], ti)
        elif typ == "search_executed":
            s["searches"].append(a)
            if a.get("premature_search"):
                s["event_ids"]["premature"].append(eid)
        elif typ == "recommendation_shown":
            s["recs"].append(a)
            if not (a.get("satisfies_known_constraints")
                    and a.get("grounded_in_catalogue")):
                s["event_ids"]["bad_rec"].append(eid)
        elif typ == "permission_requested":
            s["perm_requested"] = True
        elif typ == "permission_granted":
            s["perm_granted"] = True
        elif typ == "customer_correction":
            s["corrections"].append(a)
            s["event_ids"]["correction"].append(eid)
        elif typ == "conversation_abandoned":
            s["abandoned"] = True
    for s in out.values():
        s["goal_understood"] = s["required"] <= s["revealed"]
        s["goal_understood_turn"] = (
            max(s["revealed_at"].get(k, 0) for k in s["required"])
            if s["goal_understood"] else None)
        s["valid_search"] = any(not x.get("premature_search")
                                for x in s["searches"])
        s["grounded_rec"] = any(x.get("satisfies_known_constraints")
                                and x.get("grounded_in_catalogue")
                                for x in s["recs"])
        o = s["outcome"] or {}
        s["correct_cart"] = bool(o.get("correct_cart_action"))
        s["task_satisfied"] = bool(o.get("task_satisfied"))
    return list(out.values())


def _stage_flags(s: dict) -> dict:
    return {"session_started": True, "goal_understood": s["goal_understood"],
            "valid_search": s["valid_search"],
            "grounded_recommendation": s["grounded_rec"],
            "permission_requested": s["perm_requested"],
            "permission_granted": s["perm_granted"],
            "correct_cart": s["correct_cart"],
            "task_satisfied": s["task_satisfied"]}


def _fail_reason(s: dict, failed_stage: str) -> str:
    if failed_stage == "goal_understood":
        missing = sorted(s["required"] - s["revealed"])
        if s["event_ids"]["premature"] and not s["searches"][0:1] == []:
            return f"searched_before_discovery(missing={missing})"
        return f"constraints_never_revealed(missing={missing})" \
            if s["searches"] or s["revealed"] else "no_discovery_attempted"
    if failed_stage == "valid_search":
        return ("only_premature_searches" if s["searches"]
                else "never_searched")
    if failed_stage == "grounded_recommendation":
        return ("ungrounded_or_constraint_violating_recommendation"
                if s["recs"] else "no_recommendation")
    if failed_stage == "permission_requested":
        return "no_permission_request"
    if failed_stage == "permission_granted":
        return "permission_denied_no_recovery"
    if failed_stage == "correct_cart":
        return "granted_but_not_carted_or_wrong_item"
    if failed_stage == "task_satisfied":
        return "carted_but_not_cheapest_or_flow_incomplete"
    return "unknown"


def milestone_funnel(store: PlatformStore, by: str | None = None) -> dict:
    """Reach + eligible + conversion per stage, failure reason codes, and
    breakdowns by model_version / scenario_family."""
    sess = _sessions(store)
    total_eps = store.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    def funnel_for(group: list[dict]) -> dict:
        stages_out, prev = [], None
        reasons: dict[str, dict[str, int]] = {}
        for st in STAGES:
            reached = [s for s in group if _stage_flags(s)[st]]
            conv = round(len(reached) / len(prev), 3) if prev else None
            if prev is not None:
                for s in prev:
                    if not _stage_flags(s)[st]:
                        r = _fail_reason(s, st)
                        reasons.setdefault(st, {})
                        reasons[st][r] = reasons[st].get(r, 0) + 1
            stages_out.append({"stage": st, "reached": len(reached),
                               "eligible": len(group),
                               "reach_rate": round(len(reached) / len(group), 3)
                               if group else None,
                               "conversion_from_prev": conv})
            prev = reached
        return {"stages": stages_out, "failure_reasons": reasons,
                "sessions": len(group)}

    out = {"overall": funnel_for(sess),
           "excluded_no_goal": total_eps - len(sess),
           "note": "milestones from authoritative semantic events + "
                   "deterministic outcomes; sessions without a structured "
                   "goal are excluded and counted"}
    for dim in ("model_version", "scenario_family"):
        vals = sorted({s[dim] for s in sess if s[dim]})
        out[f"by_{dim}"] = {v: funnel_for([s for s in sess if s[dim] == v])
                            for v in vals}
    return out


def _prov(name, num_desc, den_desc, num, den, group, exclusions):
    return {"metric": name, "value": round(num / den, 4) if den else None,
            "numerator": f"{num_desc} ({num})",
            "denominator": f"{den_desc} ({den})",
            "eligible_population": "goal-carrying sessions",
            "exclusions": exclusions, "window": "all-time",
            "min_support": MIN_SUPPORT,
            "low_support": den < MIN_SUPPORT,
            "by_model_version": group}


def friction_metrics(store: PlatformStore) -> dict:
    sess = _sessions(store)
    fresh = max((s["max_ts"] for s in sess), default=0)

    def per_model(fn_num, fn_den):
        res = {}
        for mv in sorted({s["model_version"] for s in sess}):
            g = [s for s in sess if s["model_version"] == mv]
            n, d = fn_num(g), fn_den(g)
            res[mv] = {"value": round(n / d, 4) if d else None,
                       "n": n, "eligible": d, "low_support": d < MIN_SUPPORT}
        return res

    discovery = [s for s in sess if len(s["required"]) > 1]
    premature = _prov(
        "premature_search_rate",
        "sessions with a search before all required constraints known",
        "sessions requiring constraint discovery (>1 required constraint)",
        sum(1 for s in discovery if s["event_ids"]["premature"]),
        len(discovery),
        per_model(lambda g: sum(1 for s in g if s["event_ids"]["premature"]
                                and len(s["required"]) > 1),
                  lambda g: sum(1 for s in g if len(s["required"]) > 1)),
        "sessions without structured goal; single-constraint scenarios")
    corrections = _prov(
        "customer_correction_rate",
        "sessions where the customer restated a missed/violated requirement",
        "goal-carrying sessions",
        sum(1 for s in sess if s["corrections"]), len(sess),
        per_model(lambda g: sum(1 for s in g if s["corrections"]),
                  lambda g: len(g)),
        "sessions without structured goal")
    understood = [s for s in sess if s["goal_understood_turn"] is not None]
    turns_vals = sorted(s["goal_understood_turn"] for s in understood)
    ttg = {"metric": "turns_to_goal_understood",
           "value": turns_vals[len(turns_vals) // 2] if turns_vals else None,
           "numerator": "median turn index at which the LAST required "
                        f"constraint was revealed ({len(turns_vals)} sessions)",
           "denominator": f"sessions reaching goal_understood ({len(understood)})",
           "eligible_population": "goal-carrying sessions",
           "exclusions": "sessions that never reached goal understanding "
                         f"({len(sess) - len(understood)})",
           "window": "all-time", "min_support": MIN_SUPPORT,
           "low_support": len(understood) < MIN_SUPPORT,
           "by_model_version": {
               mv: (lambda v: {"value": v[len(v) // 2] if v else None,
                               "n": len(v), "eligible": len(v),
                               "low_support": len(v) < MIN_SUPPORT})(
                   sorted(s["goal_understood_turn"] for s in sess
                          if s["model_version"] == mv
                          and s["goal_understood_turn"] is not None))
               for mv in sorted({s["model_version"] for s in sess})}}
    return {"freshness_ts": fresh,
            "freshness_age_s": round(time.time() - fresh, 1) if fresh else None,
            "premature_search_rate": premature,
            "customer_correction_rate": corrections,
            "turns_to_goal_understood": ttg}
