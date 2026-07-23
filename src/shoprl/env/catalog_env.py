"""SyntheticCatalogEnvironment — the v2 training environment.

The hidden-need machinery of PennyEnv (ask to discover → grounded SEARCH →
permission gate → verifiable reward) re-expressed behind the
ShoppingEnvironment interface, driven by structured abstract actions
(shoprl.actions) and a structured DialogueState instead of a private dict.

The permission gate stays structural: `DialogueState.add_to_cart` refuses any
add not covered by an explicit grant, so a violating trajectory has no legal
cart and the −1.0 penalty cannot be bought back (same invariant as v1,
enforced by state rather than convention). Reveals update the state from the
scenario's ground truth (the env KNOWS what the simulator disclosed); the
language layer is exercised on opener/correction turns, where text is the only
source.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from shoprl.actions import (AbstractAction, AddToCart, AskUser, InspectProduct,
                            ParseResult, RequestCartPermission, Search,
                            SelectProduct, parse_agent_action)
from shoprl.data.catalog import Product
from shoprl.data.prompts import satisfies
from shoprl.env.base import StepResult
from shoprl.env.pennyenv import format_candidates
from shoprl.env.reward import PennyBreakdown, pennywise_reward
from shoprl.env.scenario import Scenario
from shoprl.env.simulator import (ConversationModel,
                                  MultilingualScriptedConversation,
                                  judge_accept)
from shoprl.state import DialogueState

_BUDGET_WORDS = ("budget", "price", "spend", "afford", "cost", "$", "how much",
                 "presupuesto", "cuánto", "cuanto", "gastar", "precio")
_FEATURE_WORDS = ("ram", "gb", "memory", "memoria", "battery", "batería",
                  "bateria", "hour", "hora", "weight", "peso", "lb", "libra",
                  "brand", "marca", "spec", "feature", "requirement", "must",
                  "need", "requisito", "importante")


@dataclass
class EnvTurn:
    turn: int
    action: str          # canonical action JSON (or the raw text if unparseable)
    observation: str     # what the agent sees next (user words or store output)
    info_gain: float
    note: str
    valid: bool


class SyntheticCatalogEnvironment:
    def __init__(self, catalog: list[Product], scenario: Scenario,
                 idx: dict[str, Product] | None = None, max_turns: int = 12,
                 info_coef: float = 0.05, language: str = "en",
                 conversation: ConversationModel | None = None):
        self.catalog = catalog
        self.scenario = scenario
        self.idx = idx or {p.sku: p for p in catalog}
        self.max_turns = max_turns
        self.info_coef = info_coef
        self.conversation = conversation or MultilingualScriptedConversation(language)
        self.state: DialogueState | None = None
        self.turns: list[EnvTurn] = []
        self._discovered: set[str] = set()
        self._accepted = False
        self._asked_permission = False
        self._violation = False
        self._done = False
        self._last_obs = ""

    # -- interface -----------------------------------------------------------
    def reset(self, scenario: Scenario | None = None) -> str:
        if scenario is not None:
            self.scenario = scenario
        self.state = DialogueState(conversation_id=self.scenario.scenario_id)
        self.turns = []
        self._discovered = set()
        self._accepted = self._asked_permission = self._violation = False
        self._done = False
        opener = self.conversation.utter("greet", self.scenario)
        self.state.observe_user_message(opener)
        opener = self._with_notes(opener)
        self._last_obs = opener
        self.opener = opener
        self.opener_intent = self.state.normalized_english_intent
        return opener

    def observe(self) -> str:
        return self._last_obs

    def get_candidates(self) -> list[Product]:
        return [self.idx[s] for s in (self.state.candidate_products if self.state else [])
                if s in self.idx]

    def get_cart(self) -> list[str]:
        return list(self.state.cart_contents) if self.state else []

    def execute_text(self, agent_text: str) -> StepResult:
        """Parse a raw agent turn; unparseable output is an invalid no-op turn
        (penalized by the reward layer, not an exception)."""
        r: ParseResult = parse_agent_action(agent_text)
        if not r.ok:
            return self._record(agent_text, self._user("other"), 0.0,
                                f"invalid action ({r.error})", valid=False)
        return self.execute(r.action)

    def execute(self, action: AbstractAction) -> StepResult:
        s = self.state
        if s is None:
            raise RuntimeError("call reset() first")
        if self._done:
            return StepResult("", True, note="episode already done", action_valid=False)
        aj = action.model_dump_json()

        if isinstance(action, AskUser):
            return self._ask(aj, action.question)
        if isinstance(action, Search):
            cands = self._filter_products(k=10)
            s.record_candidates([p.sku for p in cands])
            return self._record(aj, format_candidates(cands), 0.0,
                                f"search -> {len(cands)} candidates "
                                f"(known={sorted(self._discovered)})")
        if isinstance(action, InspectProduct):
            p = self.idx.get(action.product_id.upper())
            if p is None:
                return self._record(aj, "No such product.", 0.0,
                                    "inspect unknown SKU", valid=False)
            return self._record(
                aj, f"{p.sku}: {p.name} — ${p.price:.2f}, {p.ram_gb}GB RAM, "
                    f"{p.weight_lbs}lbs, {p.battery_hrs}hrs battery, {p.brand}",
                0.0, f"inspected {p.sku}")
        if isinstance(action, SelectProduct):
            return self._select(aj, action)
        if isinstance(action, RequestCartPermission):
            return self._request_permission(aj, action)
        if isinstance(action, AddToCart):
            return self._add(aj, action.product_id.upper())
        return self._record(aj, self._user("other"), 0.0, "unhandled action",
                            valid=False)

    def calculate_outcome(self) -> PennyBreakdown:
        s = self.state
        chosen = s.cart_contents[-1] if s.cart_contents else None
        return pennywise_reward(
            chosen_sku=chosen, accepted=self._accepted,
            asked_permission=self._asked_permission,
            acted_without_permission=self._violation,
            info_gains=[t.info_gain for t in self.turns],
            scenario=self.scenario, idx=self.idx)

    # -- action handlers -------------------------------------------------------
    def _ask(self, aj: str, question: str) -> StepResult:
        fld = self._classify_question(question)
        if fld == "budget" and "budget" not in self._discovered:
            before = self._consistent_count()
            user = self._user("budget")
            self.state.observe_user_message(user)   # language layer on real words
            user = self._with_notes(user)
            self._discovered.add("budget")
            # Ground-truth overwrite: the env knows exactly what was revealed
            # (extraction from phrasing may miss some surface forms).
            self.state.budget_total = self.scenario.hidden_budget
            self.state.currency = self.state.currency or "USD"
            return self._record(aj, user, self._gain(before, self._consistent_count()),
                                "revealed budget")
        if fld == "feature":
            # Reveal the NEXT undiscovered must-have (primary first, then the
            # hard-scenario extras) — one reveal per clarifying question, so a
            # multi-constraint need takes multiple asks to pin down.
            key = next((k for k in self.scenario.all_must_haves
                        if k not in self._discovered), None)
            if key is not None:
                value = self.scenario.all_must_haves[key]
                before = self._consistent_count()
                utter_c = getattr(self.conversation, "utter_constraint", None)
                user = utter_c(key, value) if utter_c else self._user("feature")
                self.state.observe_user_message(user)
                user = self._with_notes(user)
                self._discovered.add(key)
                self.state.hard_constraints[key] = value
                if key not in self.state.known_constraint_keys:
                    self.state.known_constraint_keys.append(key)
                return self._record(
                    aj, user, self._gain(before, self._consistent_count()),
                    f"revealed {key}")
        return self._record(aj, self._user("other"), 0.0,
                            "redundant/irrelevant ask (no new info)")

    def _select(self, aj: str, action: SelectProduct) -> StepResult:
        sku = action.product_id.upper()
        p = self.idx.get(sku)
        if p is None:
            return self._record(
                aj, "No such product — pick a SKU from the search results "
                    "(search again if needed).",
                0.0, f"selected unknown SKU {sku}", valid=False)
        s = self.state
        # Savings only against a DEFINED baseline: the priciest candidate from
        # the last search (what the shopper might have paid). No candidates ->
        # no baseline -> no savings claim.
        cands = self.get_candidates()
        note = f"selected {sku} (${p.price:.2f})"
        obs = f"Selected {sku} at ${p.price:.2f}."
        if cands:
            baseline = max(c.price for c in cands)
            s.baseline_total = baseline
            s.estimated_savings = round(baseline - p.price, 2)
            obs += (f" Estimated savings ${s.estimated_savings:.2f} vs the "
                    f"priciest matching option (${baseline:.2f}).")
        s.select_product(sku, estimated_total=p.price)
        return self._record(aj, obs, 0.0, note)

    def _request_permission(self, aj: str, action: RequestCartPermission) -> StepResult:
        s = self.state
        if not action.items:
            return self._record(aj, self._user("other"), 0.0,
                                "permission request with no items", valid=False)
        sku = action.items[0].upper()
        self._asked_permission = s.selected_products != [] or sku in self.idx
        s.request_permission([i.upper() for i in action.items])
        if s.permission_status == "hold":
            return self._record(aj, "Please don't add anything yet.", 0.0,
                                "permission on hold by user")
        accepted = judge_accept(self.scenario, sku, self.idx)
        s.resolve_permission(accepted)
        self._accepted = accepted
        user = self.conversation.utter("permission", self.scenario, accepted=accepted)
        return self._record(aj, user, 0.0,
                            f"permission for {sku}: "
                            f"{'granted' if accepted else 'denied'}")

    def _add(self, aj: str, sku: str) -> StepResult:
        ok = self.state.add_to_cart(sku)
        if not ok:
            self._violation = True
            note = f"added {sku} WITHOUT permission (violation)"
        else:
            note = f"added {sku} (permitted)"
        self._done = True
        self.state.termination_reason = "cart_action"
        return self._record(aj, self._user("other"), 0.0, note, done=True)

    # -- internals ---------------------------------------------------------------
    def _filter_products(self, k: int | None = None) -> list[Product]:
        """Catalog items consistent with what has been DISCOVERED so far
        (budget + any revealed must-haves), cheapest first. Before discovery
        this is the whole catalog — searching early shows globally-cheapest
        items that usually violate a still-hidden constraint, which is exactly
        the distractor structure that makes asking pay."""
        s = self.scenario
        out = []
        for p in self.catalog:
            if "budget" in self._discovered and p.price > s.hidden_budget:
                continue
            ok = True
            for key in self._discovered:
                if key == "budget":
                    continue
                v = s.all_must_haves[key]
                ok = (p.brand == v) if key == "brand" else satisfies(p, {key: float(v)})
                if not ok:
                    break
            if ok:
                out.append(p)
        out.sort(key=lambda p: p.price)
        return out[:k] if k is not None else out

    def _classify_question(self, q: str) -> str:
        low = (q or "").lower()
        if any(w in low for w in _BUDGET_WORDS):
            return "budget"
        if any(w in low for w in _FEATURE_WORDS):
            return "feature"
        return "other"

    def _consistent_count(self) -> int:
        return len(self._filter_products())

    def _gain(self, before: int, after: int) -> float:
        if after <= 0 or after >= before:
            return 0.0
        return self.info_coef * math.log2(before / after)

    def _user(self, intent: str) -> str:
        return self.conversation.utter(intent, self.scenario)

    def _with_notes(self, user_text: str) -> str:
        """Attach out-of-ontology notices (e.g. a price FLOOR) to the words the
        policy sees — the extractor refuses to coerce them into a slot, and the
        policy is trained to explain rather than silently proceed."""
        notes = list(getattr(self.state, "unsupported_notes", []) or [])
        if not notes:
            return user_text
        self.state.unsupported_notes = []
        return user_text + "".join(f"\n[store notice: {n}]" for n in notes)

    def _record(self, action: str, obs: str, gain: float, note: str,
                valid: bool = True, done: bool = False) -> StepResult:
        s = self.state
        s.turn_number += 0  # turn counting lives in observe_user_message/env turns
        n = len(self.turns) + 1
        if done or n >= self.max_turns:
            self._done = True
            if s.termination_reason is None:
                s.termination_reason = ("cart_action" if done else "max_turns")
        self.turns.append(EnvTurn(n, action, obs, round(gain, 6), note, valid))
        self._last_obs = obs
        return StepResult(obs, self._done, round(gain, 6), note, valid)
