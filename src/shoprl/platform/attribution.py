"""Failure attribution v1 — ONE deterministic category, evidence-first.

Rule (CONSTRAINT_EXTRACTION): the agent searched before the required
constraints were known AND the customer had to correct it AND there is
missed-constraint evidence (an unrevealed required constraint at correction
time, or a recommendation that violated known constraints / ground truth).
Everything else ABSTAINS as UNKNOWN — no LLM judge, no guessing.
"""
from __future__ import annotations

import json

from shoprl.platform.milestones import _sessions
from shoprl.platform.store import PlatformStore

TAXONOMY_VERSION = "v2"


def attribute_session(store: PlatformStore, session_id: str) -> dict:
    sess = {s["session_id"]: s for s in _sessions(store)}
    s = sess.get(session_id)
    base = {"session_id": session_id, "taxonomy_version": TAXONOMY_VERSION,
            "attribution_method": "DETERMINISTIC_RULE"}
    if s is None:
        return {**base, "primary_category": "UNKNOWN", "confidence": 0.0,
                "evidence_event_ids": [],
                "reason": "no structured goal — attribution ineligible"}
    premature = s["event_ids"]["premature"]
    corrections = s["event_ids"]["correction"]
    missed = sorted(s["required"] - s["revealed"])
    bad_rec = s["event_ids"]["bad_rec"]
    if premature and corrections and (missed or bad_rec):
        return {**base, "primary_category": "CONSTRAINT_EXTRACTION",
                "confidence": 0.9,
                "evidence_event_ids": premature + corrections + bad_rec,
                "evidence": {"premature_searches": len(premature),
                             "customer_corrections": len(corrections),
                             "constraints_never_revealed": missed,
                             "constraint_violating_recommendations":
                                 len(bad_rec)}}
    # v2 (added AFTER run 1 measured v1's abstention pattern, BEFORE the
    # confirmation run's evaluation — see docs pre-registration): the
    # customer walked away under accumulated friction although every
    # constraint was eventually disclosed.
    redundant = s["event_ids"]["redundant"]
    abandoned = s["event_ids"]["abandoned"]
    if abandoned and (len(redundant) >= 2 or corrections)             and not (missed or bad_rec):
        return {**base, "primary_category": "EXCESSIVE_FRICTION",
                "confidence": 0.85,
                "evidence_event_ids": redundant + corrections + abandoned,
                "evidence": {"redundant_questions": len(redundant),
                             "customer_corrections": len(corrections),
                             "premature_searches": len(premature),
                             "abandoned": True}}
    return {**base, "primary_category": "UNKNOWN", "confidence": 0.0,
            "evidence_event_ids": [],
            "reason": "insufficient deterministic evidence "
                      f"(premature={bool(premature)}, "
                      f"correction={bool(corrections)}, "
                      f"missed_or_bad_rec={bool(missed or bad_rec)})"}


def validate_against_simulator(store: PlatformStore,
                               simlog_paths: list[str]) -> dict:
    """Agreement between platform attributions and the simulator's hidden
    abandonment reasons — the analytics-vs-truth validation loop."""
    import json as _json
    sim_reason = {}
    for path in simlog_paths:
        for line in open(path):
            rec = _json.loads(line)
            reasons = [l["reason"] for l in rec["log"]
                       if l["customer_action"] == "ABANDON"]
            if reasons:
                sim_reason[rec["session_id"]] = reasons[0]
    agree = total = 0
    detail = []
    for a in attribute_all(store, only_failed=True):
        truth = sim_reason.get(a["session_id"])
        if truth is None:
            continue
        total += 1
        ok = (truth == "EXCESSIVE_FRICTION"
              and a["primary_category"] in ("EXCESSIVE_FRICTION",
                                            "CONSTRAINT_EXTRACTION"))
        agree += ok
        detail.append({"session_id": a["session_id"],
                       "platform": a["primary_category"],
                       "simulator": truth, "agree": ok})
    return {"agreement": round(agree / total, 3) if total else None,
            "n_abandoned": total, "detail": detail}


def attribute_all(store: PlatformStore,
                  only_failed: bool = True) -> list[dict]:
    out = []
    for s in _sessions(store):
        o = s["outcome"] or {}
        if only_failed and o.get("task_satisfied"):
            continue
        out.append(attribute_session(store, s["session_id"]))
    return out
