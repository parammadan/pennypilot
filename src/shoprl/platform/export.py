"""PennyData dataset generation — filtered interactions → SFT-ready sequences.

The export closes the flywheel: captured conversations become training
sequences in the exact chat format the SFT pipeline consumes (system prompt +
alternating user/assistant), with a validation report attached. Violating
episodes are excluded BY DEFAULT (a training set must never teach a gate
breach); thumbs-down turns can be surfaced separately as hard-example
candidates for the next data recipe.
"""
from __future__ import annotations

from shoprl.actions import parse_agent_action
from shoprl.data.prompts_v2 import SYSTEM_PROMPT_CHAT_MIN
from shoprl.platform.store import PlatformStore


def build_dataset(store: PlatformStore, label: str | None = None,
                  feedback: str | None = None,
                  exclude_violations: bool = True) -> dict:
    db = store.db
    where, args = [], []
    if label:
        where.append("COALESCE(e.label,'')=?"); args.append(label)
    if exclude_violations:
        where.append("COALESCE(e.violation,0)=0")
    cond = ("WHERE " + " AND ".join(where)) if where else ""
    sessions = [r[0] for r in db.execute(
        f"SELECT session_id FROM episodes e {cond} ORDER BY started", args)]

    sequences, skipped_unparseable = [], 0
    hard_candidates = []
    for sid in sessions:
        turns = db.execute(
            "SELECT i, agent, observation, feedback FROM turns"
            " WHERE session_id=? ORDER BY i", (sid,)).fetchall()
        if not turns:
            continue
        opener = db.execute("SELECT brief FROM episodes WHERE session_id=?",
                            (sid,)).fetchone()
        msgs = [{"role": "system", "content": SYSTEM_PROMPT_CHAT_MIN}]
        ok = True
        for i, agent, obs, fb in turns:
            if fb == "down":
                hard_candidates.append({"session_id": sid, "i": i,
                                        "agent": agent, "observation": obs})
            if feedback and fb != feedback:
                ok = False
                break
            if not parse_agent_action(agent).ok:
                skipped_unparseable += 1
            if i == 0:
                msgs.append({"role": "user", "content": "(shopper opens the chat)"})
            msgs.append({"role": "assistant", "content": agent})
            if obs:
                msgs.append({"role": "user", "content": obs})
        if ok and len(msgs) > 2:
            sequences.append({"session_id": sid, "messages": msgs})

    report = {
        "sessions_considered": len(sessions),
        "sequences": len(sequences),
        "turns_with_unparseable_action": skipped_unparseable,
        "hard_example_candidates": len(hard_candidates),
        "filters": {"label": label, "feedback": feedback,
                    "exclude_violations": exclude_violations},
        "format": "chat messages, system=SYSTEM_PROMPT_CHAT_MIN "
                  "(matches train/sft.py consumption)",
    }
    return {"report": report, "data": sequences,
            "hard_candidates": hard_candidates}
