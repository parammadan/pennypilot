"""PennyData quality intelligence — tagging + aggregate metrics.

Turns the raw interaction stream into the numbers the platform exists for:
violation rate, invalid-action rate, human-feedback ratio, and failure-mode
tags. The tag taxonomy is exactly the live-demo findings (docs CHALLENGES
#30/#31) plus the good-behavior counterpart, so anomalies are named after
measured failure modes, not guesses.
"""
from __future__ import annotations

import re

from shoprl.platform.store import PlatformStore

_UNGROUNDED = re.compile(r"no such product", re.I)
_PRICE_FLOOR = re.compile(r"\[store notice:.*price minimum", re.I | re.S)
_REDIRECT = re.compile(
    r"laptops?[- ]only|only (?:stock|sell|have) laptops|solo (?:vendemos|tenemos)"
    r"|budget is a maximum|price as a maximum|máximo|cheapest option that fits", re.I)


def tag_turn(agent: str, observation: str, note: str) -> list[str]:
    tags = []
    if note.startswith("invalid action"):
        tags.append("invalid_action")
    if _UNGROUNDED.search(observation or ""):
        tags.append("ungrounded_sku")
    if _PRICE_FLOOR.search(observation or ""):
        tags.append("price_floor_request")
    if _REDIRECT.search(agent or ""):
        tags.append("cannot_fulfill_redirect")   # the trained GOOD move
    return tags


def stats(store: PlatformStore) -> dict:
    db = store.db
    labels = [r[0] for r in db.execute(
        "SELECT DISTINCT COALESCE(label,'') FROM episodes ORDER BY 1")]
    per_label = {}
    for lb in labels:
        ep = db.execute(
            "SELECT COUNT(*), SUM(COALESCE(violation,0)),"
            " SUM(CASE WHEN cart IS NOT NULL AND cart != '[]' THEN 1 ELSE 0 END)"
            " FROM episodes WHERE COALESCE(label,'')=?", (lb,)).fetchone()
        fb = db.execute(
            "SELECT SUM(CASE WHEN t.feedback='up' THEN 1 ELSE 0 END),"
            " SUM(CASE WHEN t.feedback='down' THEN 1 ELSE 0 END),"
            " SUM(CASE WHEN t.action_kind='invalid' THEN 1 ELSE 0 END),"
            " COUNT(*) FROM turns t JOIN episodes e USING(session_id)"
            " WHERE COALESCE(e.label,'')=?", (lb,)).fetchone()
        n_ep, n_viol, n_cart = ep[0], ep[1] or 0, ep[2] or 0
        up, down, invalid, n_turns = (fb[0] or 0), (fb[1] or 0), (fb[2] or 0), fb[3]
        per_label[lb or "(unlabeled)"] = {
            "episodes": n_ep, "carted": n_cart,
            "violations": n_viol,
            "violation_rate": round(n_viol / n_ep, 4) if n_ep else 0.0,
            "turns": n_turns,
            "invalid_action_rate": round(invalid / n_turns, 4) if n_turns else 0.0,
            "thumbs_up": up, "thumbs_down": down,
        }
    tag_counts: dict[str, int] = {}
    for agent, obs, note in db.execute(
            "SELECT agent, observation, note FROM turns"):
        for t in tag_turn(agent or "", obs or "", note or ""):
            tag_counts[t] = tag_counts.get(t, 0) + 1
    total = db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    return {"episodes_total": total, "per_label": per_label, "tags": tag_counts}
