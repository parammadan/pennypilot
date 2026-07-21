"""Transcript rendering for v2 episodes — the human-readable artifact.

One episode → one readable text block (agent actions, user/store replies,
info-gain, permission trail) + a structured dict twin for JSONL. Used by the
eval harness, the demo replay, and the docs (measured transcripts, never
hand-written ones).
"""
from __future__ import annotations

import json
from dataclasses import asdict


def render_transcript(env, outcome=None) -> str:
    s = env.state
    lines = [f"=== {s.conversation_id} | languages={','.join(s.detected_languages)}"
             f"{' (code-switched)' if s.code_switched else ''} ==="]
    opener = getattr(env, "opener", None)
    if opener:
        lines.append(f"USER: {opener}")
    gloss = getattr(env, "opener_intent", "")
    if gloss:
        lines.append(f"  {gloss}")
    for t in env.turns:
        lines.append(f"[{t.turn:>2}] AGENT: {t.action}")
        flags = []
        if t.info_gain:
            flags.append(f"info+{t.info_gain:.3f}")
        if not t.valid:
            flags.append("INVALID")
        suffix = f"   ({t.note}{'; ' + ' '.join(flags) if flags else ''})"
        lines.append(f"     ENV/USER: {t.observation}{suffix}")
    lines.append(
        f"--- state: budget={s.budget_total} {s.currency or ''} | "
        f"constraints={s.hard_constraints} | selected={s.selected_products} | "
        f"savings={s.estimated_savings} (baseline={s.baseline_total}) | "
        f"permission={s.permission_status} | cart={s.cart_contents} | "
        f"ended={s.termination_reason}")
    if outcome is not None:
        lines.append(
            f"--- reward: total={outcome.total:+.3f} (value={outcome.value_quality:.3f} "
            f"accepted={outcome.accepted:.0f} asked_perm={outcome.asked_permission:.0f} "
            f"violation={outcome.acted_without_permission:.0f} "
            f"info={outcome.info_gain:.3f})")
    return "\n".join(lines)


def transcript_record(env, outcome=None) -> dict:
    """Structured twin of the rendered transcript (one JSONL row)."""
    rec = {
        "scenario_id": env.scenario.scenario_id,
        "turns": [asdict(t) for t in env.turns],
        "state": env.state.model_dump(),
    }
    if outcome is not None:
        rec["reward"] = asdict(outcome)
    return rec


def write_transcripts_jsonl(records: list[dict], path) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
