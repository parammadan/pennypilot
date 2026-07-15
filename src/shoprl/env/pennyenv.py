"""Pennywise conversational environment: hidden need + permission gate.

Unlike `ShopEnv` (single visible goal, catalog-filtering actions), Pennywise is a
conversation: the shopper's need is HIDDEN, so the agent must ASK to discover it,
RECOMMEND, ASK_PERMISSION, and only then ADD_TO_CART. The env owns the state
machine and delegates the two simulator seams (see `simulator.py`):
  - the accept/reject DECISION → `judge_accept` (programmatic, verifiable);
  - the user's WORDS → a `ConversationModel` (scripted now, LLM later).

Action grammar (structured now; a real policy's natural language is mapped by the
same parser later):
    ASK[budget] | ASK[feature] | RECOMMEND[SKU] | ASK_PERMISSION[SKU?] | ADD_TO_CART[SKU]

Per-turn info-gain measures how much a clarifying question shrinks the set of
catalog items still consistent with what the agent knows — a question that
reveals a hidden field cuts that set; a redundant/irrelevant one does not.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from shoprl.data.catalog import Product
from shoprl.data.prompts import satisfies
from shoprl.env.scenario import Scenario
from shoprl.env.simulator import ConversationModel, ScriptedConversation, judge_accept

_ACTION_RE = re.compile(
    r"(ASK_PERMISSION|ADD_TO_CART|RECOMMEND|SEARCH|ASK)(?:\s*\[(.*?)\])?",
    re.IGNORECASE)
_BUDGET_WORDS = ("budget", "price", "spend", "afford", "cost", "$", "how much")
_FEATURE_WORDS = ("ram", "gb", "memory", "battery", "hour", "weight", "lb",
                  "brand", "make", "spec", "feature")
_SEARCH_TOPK = 10


def search_catalog(catalog: list[Product], scenario: Scenario, known: set[str],
                   k: int = _SEARCH_TOPK) -> list[Product]:
    """Catalog items consistent with the constraints the agent has DISCOVERED
    (`known` ⊆ {'budget','feature'}), cheapest first, top-k. Searching before
    asking (empty `known`) returns the globally cheapest items — which usually
    violate the hidden must-have — so asking first is still the winning move; the
    first result becomes the cheapest VALID item only once both are known."""
    out = []
    for p in catalog:
        if "budget" in known and p.price > scenario.hidden_budget:
            continue
        if "feature" in known:
            if scenario.must_have_key == "brand":
                if p.brand != scenario.must_have_value:
                    continue
            elif not satisfies(p, {scenario.must_have_key: float(scenario.must_have_value)}):
                continue
        out.append(p)
    out.sort(key=lambda p: p.price)
    return out[:k]


def format_candidates(cands: list[Product]) -> str:
    """The store's response the agent reads to pick from (env observation)."""
    if not cands:
        return "No matching products found."
    lines = ["Matching products (cheapest first):"]
    for p in cands:
        lines.append(f"- {p.sku}: ${p.price:.0f}, {p.ram_gb}GB RAM, "
                     f"{p.weight_lbs}lbs, {p.battery_hrs}hrs, {p.brand}")
    return "\n".join(lines)


@dataclass
class PennyTurn:
    turn: int
    action: str          # ASK | RECOMMEND | ASK_PERMISSION | ADD_TO_CART | OTHER
    arg: str
    user_utterance: str
    revealed: str | None  # "budget" | "feature" | None
    info_gain: float
    note: str


@dataclass
class PennyState:
    turn: int = 0
    known: dict[str, float | str] = field(default_factory=dict)  # 'budget'/'feature'
    recommended: str | None = None
    last_search: list[str] = field(default_factory=list)  # SKUs shown by last SEARCH
    permission_for: str | None = None   # SKU the user explicitly accepted
    asked_permission: bool = False      # asked permission for a recommended item
    accepted: bool = False              # last permission decision (programmatic)
    cart: list[str] = field(default_factory=list)
    acted_without_permission: bool = False
    done: bool = False


class PennyEnv:
    def __init__(self, catalog: list[Product], scenario: Scenario,
                 idx: dict[str, Product] | None = None, max_turns: int = 8,
                 info_coef: float = 0.05,
                 conversation: ConversationModel | None = None):
        self.catalog = catalog
        self.scenario = scenario
        self.idx = idx or {p.sku: p for p in catalog}
        self.max_turns = max_turns
        self.info_coef = info_coef
        self.conversation = conversation or ScriptedConversation()
        self.state: PennyState | None = None
        self.turns: list[PennyTurn] = []

    def reset(self) -> str:
        self.state = PennyState()
        self.turns = []
        return self.scenario.opening_utterance

    # -- info-gain: size of the catalog set still consistent with known fields --
    def _consistent_count(self, known: dict[str, float | str]) -> int:
        s = self.scenario
        cnt = 0
        for p in self.catalog:
            if "budget" in known and p.price > s.hidden_budget:
                continue
            if "feature" in known:
                if s.must_have_key == "brand":
                    if p.brand != s.must_have_value:
                        continue
                elif not satisfies(p, {s.must_have_key: float(s.must_have_value)}):
                    continue
            cnt += 1
        return cnt

    def _gain(self, before: int, after: int) -> float:
        if after <= 0 or after >= before:
            return 0.0
        return self.info_coef * math.log2(before / after)

    def _parse(self, text: str) -> tuple[str, str]:
        m = _ACTION_RE.search(text or "")
        if m:
            return m.group(1).upper(), (m.group(2) or "").strip()
        low = (text or "").lower()  # free-text fallback -> infer an ASK intent
        if any(w in low for w in _BUDGET_WORDS):
            return "ASK", "budget"
        if any(w in low for w in _FEATURE_WORDS):
            return "ASK", "feature"
        return "OTHER", ""

    def _ask_field(self, arg: str, text: str) -> str:
        a = (arg or "").lower()
        if a in ("budget", "price", "cost", "spend"):
            return "budget"
        if a in ("feature", "ram", "battery", "weight", "brand", "spec", "specs"):
            return "feature"
        low = (text or "").lower()
        if any(w in low for w in _BUDGET_WORDS):
            return "budget"
        if any(w in low for w in _FEATURE_WORDS):
            return "feature"
        return "other"

    def step(self, agent_text: str) -> tuple[str, bool, PennyTurn]:
        s = self.state
        if s is None:
            raise RuntimeError("call reset() first")
        if s.done:
            rec = PennyTurn(s.turn, "NOOP", "", "", None, 0.0, "episode already done")
            return rec.user_utterance, True, rec

        s.turn += 1
        kind, arg = self._parse(agent_text)
        revealed: str | None = None
        info_gain = 0.0
        note = ""

        if kind == "ASK":
            fld = self._ask_field(arg, agent_text)
            if fld == "budget" and "budget" not in s.known:
                before = self._consistent_count(s.known)
                s.known["budget"] = self.scenario.hidden_budget
                info_gain = self._gain(before, self._consistent_count(s.known))
                revealed, note = "budget", "revealed budget"
                user = self.conversation.utter("budget", self.scenario)
            elif fld == "feature" and "feature" not in s.known:
                before = self._consistent_count(s.known)
                s.known["feature"] = self.scenario.must_have_value
                info_gain = self._gain(before, self._consistent_count(s.known))
                revealed, note = "feature", "revealed must-have"
                user = self.conversation.utter("feature", self.scenario)
            else:
                note = "redundant/irrelevant ask (no new info)"
                user = self.conversation.utter("other", self.scenario)

        elif kind == "SEARCH":
            cands = search_catalog(self.catalog, self.scenario, set(s.known))
            s.last_search = [p.sku for p in cands]
            note = f"search -> {len(cands)} candidates (known={sorted(s.known)})"
            user = format_candidates(cands)   # env/store observation, not the user

        elif kind == "RECOMMEND":
            s.recommended = arg.upper() or s.recommended
            note = f"recommended {s.recommended}"
            user = self.conversation.utter("other", self.scenario)

        elif kind == "ASK_PERMISSION":
            sku = arg.upper() or s.recommended
            if s.recommended is not None:
                s.asked_permission = True
            accepted = judge_accept(self.scenario, sku, self.idx)  # programmatic
            s.accepted = accepted
            if accepted:
                s.permission_for = sku
            note = f"permission for {sku}: {'granted' if accepted else 'denied'}"
            user = self.conversation.utter("permission", self.scenario, accepted=accepted)

        elif kind == "ADD_TO_CART":
            sku = arg.upper() or s.recommended
            if s.permission_for and s.permission_for == sku:
                s.cart.append(sku)
                note = f"added {sku} (permitted)"
            else:
                s.acted_without_permission = True
                note = f"added {sku} WITHOUT permission (violation)"
            s.done = True
            user = self.conversation.utter("other", self.scenario)

        else:
            note = "unrecognized / no-op"
            user = self.conversation.utter("other", self.scenario)

        if s.turn >= self.max_turns:
            s.done = True
        rec = PennyTurn(s.turn, kind, arg, user, revealed, round(info_gain, 6), note)
        self.turns.append(rec)
        return user, s.done, rec

    def reward(self):
        """Score the completed trajectory (delegates to `pennywise_reward`)."""
        from shoprl.env.reward import pennywise_reward
        s = self.state
        chosen = s.cart[-1] if s.cart else None
        return pennywise_reward(
            chosen_sku=chosen,
            accepted=bool(s.accepted),
            asked_permission=bool(s.asked_permission),
            acted_without_permission=bool(s.acted_without_permission),
            info_gains=[t.info_gain for t in self.turns],
            scenario=self.scenario,
            idx=self.idx,
        )
