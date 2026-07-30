"""PennyData event schema — everything the store emits, one envelope.

The platform ingests four event kinds produced by live demo sessions (and the
backfill loader): episode_start, turn, feedback, episode_end. Events are
append-only; the relational view (store.py) is derived and rebuildable.
"""
from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


def _eid() -> str:
    return uuid.uuid4().hex


class Envelope(BaseModel):
    """Fields every platform event carries. `event_id` is the identity used
    for idempotent application — redelivery/replay of the same id is a no-op.
    `ts` is EVENT time (stamped at the source); ingest time is recorded by
    the store, never by the producer."""
    event_id: str = Field(default_factory=_eid)
    source: str = ""                 # AGENT | CUSTOMER | STORE_UI | SIMULATOR | SYSTEM
    model_version: str = ""
    prompt_version: str = ""
    request_id: str = ""


class CustomerGoal(BaseModel):
    """The customer's TRUE objective. `goal_source` says how we know it:
    SIMULATED_GROUND_TRUTH (scenario truth, exact) | HUMAN_BRIEF (the demo
    hint a human was given) | INFERRED (from conversation; carries
    confidence) | AMBIGUOUS (we admit we don't know). Hidden fields are for
    simulation/evaluation/analytics ONLY — never handed to the agent."""
    goal_text: str = ""
    budget_max: float | None = None
    must_have_constraints: dict = Field(default_factory=dict)
    preferences: dict = Field(default_factory=dict)
    expertise: Literal["NOVICE", "INTERMEDIATE", "EXPERT", "UNKNOWN"] = "UNKNOWN"
    urgency: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"] = "UNKNOWN"
    language: str = ""
    goal_source: Literal["SIMULATED_GROUND_TRUTH", "HUMAN_BRIEF",
                         "INFERRED", "AMBIGUOUS"] = "AMBIGUOUS"
    confidence: float | None = None   # only meaningful for INFERRED

    @classmethod
    def from_scenario(cls, scen, language: str = "en",
                      expertise: str = "UNKNOWN") -> "CustomerGoal":
        """Build ground-truth goal from a scenario object (duck-typed:
        needs .hidden_budget and .all_must_haves)."""
        return cls(goal_text="Find the cheapest laptop meeting my requirements",
                   budget_max=float(scen.hidden_budget),
                   must_have_constraints=dict(scen.all_must_haves),
                   expertise=expertise, language=language,
                   goal_source="SIMULATED_GROUND_TRUTH")


class Outcome(BaseModel):
    """Deterministic task outcome — computed from scenario truth, catalogue,
    env state, and the final cart. NEVER inferred from clicks or thumbs."""
    task_satisfied: bool = False
    constraints_satisfied: bool = False
    cheapest_valid_product_selected: bool = False
    permission_obtained: bool = False
    correct_cart_action: bool = False
    safety_violation: bool = False
    goal_satisfaction: Literal["SATISFIED", "PARTIAL", "UNSATISFIED",
                               "UNKNOWN"] = "UNKNOWN"


class EpisodeStart(Envelope):
    kind: Literal["episode_start"] = "episode_start"
    session_id: str
    label: str = ""                  # which model/arm served it (e.g. "7b-B2")
    policy: dict = Field(default_factory=dict)   # serve_policy /health payload
    brief: str = ""                  # legacy free-text brief (kept for compat)
    goal: CustomerGoal | None = None
    scenario_family: str = ""
    ts: float = Field(default_factory=time.time)


class Turn(Envelope):
    kind: Literal["turn"] = "turn"
    session_id: str
    i: int                           # 0-based agent-turn index
    agent: str                       # full agent text (prose + action JSON)
    observation: str = ""            # what came back (user words / store output)
    note: str = ""                   # env note ("revealed budget", "invalid …")
    latency_ms: float | None = None  # wall time of the policy call (operational)
    ts: float = Field(default_factory=time.time)


class Feedback(Envelope):
    kind: Literal["feedback"] = "feedback"
    session_id: str
    i: int                           # agent-turn index the vote refers to
    vote: Literal["up", "down"]
    ts: float = Field(default_factory=time.time)


class EpisodeEnd(Envelope):
    kind: Literal["episode_end"] = "episode_end"
    session_id: str
    verdict: str = ""
    violation: bool = False
    cart: list[str] = Field(default_factory=list)
    outcome: Outcome | None = None
    ts: float = Field(default_factory=time.time)


class UiEvent(Envelope):
    """One raw interaction tick from the storefront UI (click, input, hover,
    modal action) — the clickstream primitive behavioral analysis runs on."""
    kind: Literal["ui"] = "ui"
    session_id: str
    type: str                        # click | input | hover | modal | tab
    target: str = ""                 # e.g. "card:LAP-0019", "approve", "search"
    meta: dict = Field(default_factory=dict)
    ts: float = Field(default_factory=time.time)


SEMANTIC_TYPES = frozenset({
    "constraint_requested", "constraint_revealed", "search_executed",
    "recommendation_shown", "customer_correction", "permission_requested",
    "permission_denied", "permission_granted", "cart_add_attempted",
    "cart_add_succeeded", "cart_removed", "conversation_abandoned"})


class SemanticEvent(Envelope):
    """Authoritative conversation-milestone event. Produced by the
    ENVIRONMENT (server-side truth: constraints, search, grounding,
    permission, cart), the SIMULATOR (customer_correction — the customer is
    the authority on why it corrected), or the DRIVER (conversation_abandoned
    on walk-away). Never derived from text regexes when env truth exists."""
    kind: Literal["semantic"] = "semantic"
    session_id: str
    type: str
    turn_index: int = 0
    attributes: dict = Field(default_factory=dict)
    ts: float = Field(default_factory=time.time)


EVENT_TYPES = {"episode_start": EpisodeStart, "turn": Turn,
               "feedback": Feedback, "episode_end": EpisodeEnd,
               "ui": UiEvent, "semantic": SemanticEvent}


def parse_event(obj: dict):
    """Validate one raw dict into its event model (raises on unknown/bad)."""
    kind = obj.get("kind")
    if kind not in EVENT_TYPES:
        raise ValueError(f"unknown event kind: {kind!r}")
    ev = EVENT_TYPES[kind].model_validate(obj)
    if kind == "semantic" and ev.type not in SEMANTIC_TYPES:
        raise ValueError(f"unknown semantic type: {ev.type!r}")
    return ev
