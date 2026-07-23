"""PennyData event schema — everything the store emits, one envelope.

The platform ingests four event kinds produced by live demo sessions (and the
backfill loader): episode_start, turn, feedback, episode_end. Events are
append-only; the relational view (store.py) is derived and rebuildable.
"""
from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field


class EpisodeStart(BaseModel):
    kind: Literal["episode_start"] = "episode_start"
    session_id: str
    label: str = ""                  # which model/arm served it (e.g. "7b-B2")
    policy: dict = Field(default_factory=dict)   # serve_policy /health payload
    brief: str = ""                  # the shopper's hidden brief (synthetic)
    ts: float = Field(default_factory=time.time)


class Turn(BaseModel):
    kind: Literal["turn"] = "turn"
    session_id: str
    i: int                           # 0-based agent-turn index
    agent: str                       # full agent text (prose + action JSON)
    observation: str = ""            # what came back (user words / store output)
    note: str = ""                   # env note ("revealed budget", "invalid …")
    ts: float = Field(default_factory=time.time)


class Feedback(BaseModel):
    kind: Literal["feedback"] = "feedback"
    session_id: str
    i: int                           # agent-turn index the vote refers to
    vote: Literal["up", "down"]
    ts: float = Field(default_factory=time.time)


class EpisodeEnd(BaseModel):
    kind: Literal["episode_end"] = "episode_end"
    session_id: str
    verdict: str = ""
    violation: bool = False
    cart: list[str] = Field(default_factory=list)
    ts: float = Field(default_factory=time.time)


EVENT_TYPES = {"episode_start": EpisodeStart, "turn": Turn,
               "feedback": Feedback, "episode_end": EpisodeEnd}


def parse_event(obj: dict):
    """Validate one raw dict into its event model (raises on unknown/bad)."""
    kind = obj.get("kind")
    if kind not in EVENT_TYPES:
        raise ValueError(f"unknown event kind: {kind!r}")
    return EVENT_TYPES[kind].model_validate(obj)
