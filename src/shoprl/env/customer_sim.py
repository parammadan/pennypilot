"""AdaptiveCustomer — a reactive simulated shopper (NOT the agent's policy).

Family v1: GRADUAL_CONSTRAINT_REVEAL. The customer opens vague ("I need a
laptop for programming."), reveals a constraint ONLY when the agent asks a
relevant question, corrects the agent when a recommendation violates a
still-hidden requirement, gets frustrated at repeated questions, and
abandons after excessive friction. Every reactive move is logged with its
reason and the env event IDs that triggered it — so analytics can later be
validated against the simulator's hidden causes.

Two integration points (both required):
- as the env's ConversationModel (utter/utter_constraint): reveals flow
  through the environment's truth bookkeeping, phrased per expertise;
- `post_step(parse_result, step, env)`: called by the DRIVER after each
  agent step; returns a reactive move (CORRECT / ABANDON / PROCEED) chosen
  from the env's authoritative semantic events for that step.

Seeded and versioned: identical (scenario, seed, agent behavior) ⇒ identical
customer behavior. It never emits a pre-scripted action sequence — its next
move always depends on what the agent just did.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from shoprl.env.simulator import MultilingualScriptedConversation

POLICY_VERSION = "customer-sim-v1"
FAMILY = "GRADUAL_CONSTRAINT_REVEAL"

_OPENERS = {"en": "I need a laptop for programming.",
            "es": "Necesito una laptop para programar.",
            "es-en": "Necesito una laptop for programming."}
_CORRECT = {
    "budget": "Wait — you never asked my budget. It has to stay under ${v}.",
    "min_ram": "Hold on, that won't work — it needs at least {v}GB of RAM.",
    "max_weight": "No — it has to weigh under {v} lbs, that one is too heavy.",
    "min_battery": "That battery is too weak — I need at least {v} hours.",
    "brand": "I only want a {v} — you never asked about brand.",
}


@dataclass
class CustomerMove:
    action: str                      # PROCEED | CORRECT | ABANDON
    utterance: str = ""
    reason: str = ""
    trigger_event_ids: list = field(default_factory=list)
    violated_key: str | None = None


class AdaptiveCustomer(MultilingualScriptedConversation):
    def __init__(self, scenario, seed: int = 0, language: str = "en",
                 expertise: str = "NOVICE", frustration_limit: int = 3):
        super().__init__(language)
        self.scenario = scenario
        self.rng = random.Random(f"{POLICY_VERSION}-{seed}")
        self.language = language
        self.expertise = expertise
        self.frustration = 0
        self.corrections = 0
        self.frustration_limit = frustration_limit
        self.log: list[dict] = []      # every reactive decision, with reason

    # -- ConversationModel seam: reveals stay on the env truth path ---------
    def utter(self, intent, scenario, accepted=None):
        if intent == "greet":
            return _OPENERS.get(self.language, _OPENERS["en"])
        return super().utter(intent, scenario, accepted=accepted)

    # -- reactive layer -------------------------------------------------------
    def post_step(self, parse_result, step, env) -> CustomerMove:
        """Decide the customer's reactive move from THIS step's authoritative
        events. Driver applies CORRECT (override observation +
        env.reveal_constraint) and ABANDON (end episode)."""
        evs = list(env.pending_events)

        for e in evs:
            if e["type"] == "constraint_requested" and \
                    e["attributes"].get("redundant"):
                self.frustration += 1
                self._log("FRUSTRATED", "AGENT_REPEATED_KNOWN_QUESTION",
                          [e["event_id"]])
        for e in evs:
            if e["type"] == "search_executed" and \
                    e["attributes"].get("premature_search"):
                self.frustration += 0    # allowed — but remembered
                self._log("NOTED_MISTAKE", "AGENT_SEARCHED_PREMATURELY",
                          [e["event_id"]])

        if self.frustration + self.corrections >= self.frustration_limit:
            self._log("ABANDON", "EXCESSIVE_FRICTION",
                      [e["event_id"] for e in evs])
            return CustomerMove("ABANDON", reason="EXCESSIVE_FRICTION")

        for e in evs:
            if e["type"] != "recommendation_shown":
                continue
            a = e["attributes"]
            if a.get("satisfies_full_ground_truth"):
                continue
            violated = self._violated_hidden_key(a.get("product_id"), env)
            if violated is not None:
                self.corrections += 1
                utt = _CORRECT[violated].format(
                    v=self._truth_value(violated))
                self._log("CORRECT", f"AGENT_IGNORED_{violated.upper()}",
                          [e["event_id"]], violated_key=violated)
                return CustomerMove("CORRECT", utterance=utt,
                                    reason=f"AGENT_IGNORED_{violated.upper()}",
                                    trigger_event_ids=[e["event_id"]],
                                    violated_key=violated)
        return CustomerMove("PROCEED")

    def _violated_hidden_key(self, sku, env):
        p = env.idx.get((sku or "").upper())
        if p is None:
            return None
        if "budget" not in env._discovered and \
                p.price > self.scenario.hidden_budget:
            return "budget"
        from shoprl.data.prompts import satisfies
        for key, v in self.scenario.all_must_haves.items():
            if key in env._discovered:
                continue
            ok = (p.brand == v) if key == "brand" else \
                satisfies(p, {key: float(v)})
            if not ok:
                return key
        return None

    def _truth_value(self, key):
        return (f"{self.scenario.hidden_budget:.0f}" if key == "budget"
                else self.scenario.all_must_haves[key])

    def _log(self, action, reason, trigger_ids, **extra):
        self.log.append({"customer_action": action, "reason": reason,
                         "trigger_event_ids": trigger_ids,
                         "simulator_policy_version": POLICY_VERSION,
                         **extra})
