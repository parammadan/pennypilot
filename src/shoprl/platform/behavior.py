"""PennyData behavioral intelligence — defined metrics, slices, quality gaps.

Discipline (the part that matters more than any single number):
- Every metric states its NUMERATOR and DENOMINATOR (eligible population),
  because "20% of WHAT" is the first question a number must answer.
- Slices report support (session count) and are suppressed below MIN_SUPPORT —
  a 100%-abandonment slice of 3 sessions is noise, not signal.
- Outputs separate OBSERVED BEHAVIOR from HYPOTHESIS: slice ranking surfaces
  where behavior deviates; the failure-analysis endpoint hands the underlying
  sessions to a human/scientist — the platform never claims causation.
- Data-quality checks run FIRST: before blaming the model, validate the data.
"""
from __future__ import annotations

import re

from shoprl.platform.store import PlatformStore

MIN_SUPPORT = 30          # sessions; below this a slice is reported as low-N
_REPEAT_MIN_LEN = 12      # chars; tiny user lines ("ok") never count as repeats

# Slice dimensions available on every session row (label = model/persona
# cohort; the rest derive from the session's own turns).
DIMENSIONS = ("label", "carted", "violation", "turn_bucket")


def _turn_bucket(n: int) -> str:
    return "1-3" if n <= 3 else "4-6" if n <= 6 else "7+"


def session_features(store: PlatformStore) -> list[dict]:
    """One behavioral record per session (Lesson 2: the analysis unit).
    Derived, rebuildable — never a source of truth."""
    db = store.db
    out = {}
    for sid, label, viol, cart, started, ended in db.execute(
            "SELECT session_id, COALESCE(label,''), COALESCE(violation,0),"
            " COALESCE(cart,'[]'), started, ended FROM episodes"):
        out[sid] = {"session_id": sid, "label": label, "violation": bool(viol),
                    "carted": cart not in (None, "", "[]"),
                    "duration_s": round(ended - started, 1)
                    if started and ended else None,
                    "turns": 0, "asks": 0, "searches": 0, "invalid": 0,
                    "user_msgs": [], "reformulated": False, "repeated": False}
    for sid, action_kind, obs in db.execute(
            "SELECT session_id, action_kind, observation FROM turns ORDER BY i"):
        s = out.get(sid)
        if s is None:
            continue
        s["turns"] += 1
        if action_kind == "ask_user":
            s["asks"] += 1
        elif action_kind == "search":
            s["searches"] += 1
        elif action_kind == "invalid":
            s["invalid"] += 1
        if obs:
            s["user_msgs"].append(obs)
    for s in out.values():
        msgs = [re.sub(r"\s+", " ", m).strip().lower()
                for m in s.pop("user_msgs") if len(m) >= _REPEAT_MIN_LEN]
        # repeat = the SAME normalized user line appears twice (Lesson 3:
        # stronger signal); reformulation = high lexical overlap, not equal
        s["repeated"] = len(msgs) != len(set(msgs))
        s["reformulated"] = s["repeated"] or _any_reformulation(msgs)
        s["turn_bucket"] = _turn_bucket(s["turns"])
        s["abandoned"] = (not s["carted"]) and s["turns"] > 0
    return list(out.values())


def _any_reformulation(msgs: list[str]) -> bool:
    for a, b in zip(msgs, msgs[1:]):
        wa, wb = set(a.split()), set(b.split())
        if wa and wb and len(wa & wb) / len(wa | wb) >= 0.6 and a != b:
            return True
    return False


def metrics(sessions: list[dict]) -> dict:
    """Behavioral metrics WITH their definitions attached (Lesson 3)."""
    n = len(sessions)
    eligible_search = [s for s in sessions if s["searches"] > 0]

    def rate(name, num, den, num_n, den_n):
        return {"value": round(num_n / den_n, 4) if den_n else None,
                "numerator": f"{name}: {num} ({num_n})",
                "denominator": f"{den} ({den_n})"}

    return {
        "sessions": n,
        "abandonment": rate("abandonment", "sessions with turns, no cart",
                            "all sessions with >=1 turn",
                            sum(s["abandoned"] for s in sessions),
                            sum(1 for s in sessions if s["turns"] > 0)),
        "cart_rate_after_search": rate(
            "carted", "sessions that carted",
            "sessions that reached a search (eligible population)",
            sum(s["carted"] for s in eligible_search), len(eligible_search)),
        "reformulation": rate("reformulation",
                              "sessions with a reformulated user message",
                              "all sessions",
                              sum(s["reformulated"] for s in sessions), n),
        "repeat_question": rate("repeat", "sessions repeating a user line",
                                "all sessions",
                                sum(s["repeated"] for s in sessions), n),
        "invalid_action": rate("invalid", "invalid agent turns",
                               "all agent turns",
                               sum(s["invalid"] for s in sessions),
                               sum(s["turns"] for s in sessions)),
        "violation": rate("violation", "sessions with a permission violation",
                          "all sessions",
                          sum(s["violation"] for s in sessions), n),
    }


def slice_report(sessions: list[dict], metric: str = "abandoned",
                 min_support: int = MIN_SUPPORT, top: int = 10) -> dict:
    """Single- and two-dimension slices ranked by |deviation| × support share
    (Lesson 6). Slices under min_support are counted but never ranked."""
    n = len(sessions)
    base = sum(s[metric] for s in sessions) / n if n else 0.0

    def val(group):
        return sum(s[metric] for s in group) / len(group)

    candidates, suppressed = [], 0
    # a dimension that DEFINES the metric is a tautology, not a finding
    tautology = {"abandoned": {"carted"}, "carted": {"carted"}}
    dims = [d for d in DIMENSIONS
            if d != metric and d not in tautology.get(metric, set())]
    keys = [(d,) for d in dims] + [(a, b) for i, a in enumerate(dims)
                                   for b in dims[i + 1:]]
    # a dimension VALUE covering ~everything (e.g. violation=False when no
    # violations exist) adds no information — prune slices containing one
    universal = {(d, v) for d in dims
                 for v in {s[d] for s in sessions}
                 if sum(1 for s in sessions if s[d] == v) > 0.9 * n}
    for key in keys:
        groups: dict[tuple, list] = {}
        for s in sessions:
            groups.setdefault(tuple(s[k] for k in key), []).append(s)
        for gv, group in groups.items():
            if any((k, v) in universal for k, v in zip(key, gv)):
                continue
            if len(group) < min_support:
                suppressed += 1
                continue
            v = val(group)
            candidates.append({
                "slice": dict(zip(key, gv)), "support": len(group),
                "value": round(v, 4), "baseline": round(base, 4),
                "deviation": round(v - base, 4),
                "score": round(abs(v - base) * len(group) / max(n, 1), 5)})
    candidates.sort(key=lambda c: -c["score"])
    return {"metric": metric, "baseline": round(base, 4), "sessions": n,
            "min_support": min_support, "suppressed_low_n": suppressed,
            "top_slices": candidates[:top],
            "note": "observed behavior, ranked by deviation x share — "
                    "hypotheses, not causes; inspect sessions before blaming "
                    "the model"}


def funnel_by(sessions_rows: list[dict], store: PlatformStore,
              label: str | None = None) -> dict:
    """Funnel with STAGE CONVERSIONS (Lesson 4), optionally for one cohort."""
    db = store.db
    where, args = "", []
    if label is not None:
        where, args = ("JOIN episodes e USING(session_id) "
                       "WHERE COALESCE(e.label,'')=?"), [label]
    stages = [("engaged", "ask_user"), ("searched", "search"),
              ("selected", "select_product"),
              ("permission_asked", "request_cart_permission"),
              ("carted", "add_to_cart")]
    counts = []
    for name, action in stages:
        counts.append((name, db.execute(
            f"SELECT COUNT(DISTINCT t.session_id) FROM turns t {where}"
            f" {'AND' if where else 'WHERE'} t.action_kind=?",
            args + [action]).fetchone()[0]))
    out = {"label": label or "(all)", "stages": []}
    prev = None
    for name, c in counts:
        conv = round(c / prev, 3) if prev else None
        out["stages"].append({"stage": name, "sessions": c,
                              "conversion_from_prev": conv})
        prev = c or 1
    return out


def data_quality(store: PlatformStore) -> dict:
    """Lesson 20: the platform can lie — check the data before the model."""
    db = store.db
    n_ep = db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    checks = {
        "episodes_missing_label": db.execute(
            "SELECT COUNT(*) FROM episodes WHERE label IS NULL OR label=''"
        ).fetchone()[0],
        "episodes_never_ended": db.execute(
            "SELECT COUNT(*) FROM episodes WHERE ended IS NULL").fetchone()[0],
        "orphan_turns_no_episode": db.execute(
            "SELECT COUNT(*) FROM turns t LEFT JOIN episodes e"
            " USING(session_id) WHERE e.session_id IS NULL").fetchone()[0],
        "episodes_with_zero_turns": db.execute(
            "SELECT COUNT(*) FROM episodes e LEFT JOIN turns t"
            " USING(session_id) WHERE t.session_id IS NULL").fetchone()[0],
        "duplicate_events_exact": db.execute(
            "SELECT COALESCE(SUM(c - 1), 0) FROM (SELECT COUNT(*) c FROM"
            " events GROUP BY kind, session_id, payload)").fetchone()[0],
    }
    flags = [k for k, v in checks.items()
             if n_ep and v / max(n_ep, 1) > 0.05 and k != "duplicate_events_exact"]
    return {"episodes": n_ep, "checks": checks,
            "flags_over_5pct": flags,
            "verdict": "SUSPECT — validate instrumentation before reading "
                       "any behavioral metric" if flags else "clean"}
