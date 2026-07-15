"""User-simulator — two seams kept strictly separate.

1. ACCEPT/REJECT JUDGMENT (`judge_accept`): always a programmatic check against
   the hidden constraints. This is the ONLY authority on whether the user accepts
   a recommendation, and it is what the reward reads. It never depends on the
   conversation model — so a model cannot be *persuaded* into accepting a bad
   item (reward hacking), and the accept signal stays verifiable.

2. CONVERSATION GENERATION (`ConversationModel`): produces the natural-language
   user turns. `ScriptedConversation` is the CPU/Phase-1 implementation; a frozen
   instruct LLM can drop in behind the same protocol later WITHOUT touching the
   reward or the judgment — it only changes the wording, never the decision.

The env calls `judge_accept` for the decision and the conversation model for the
words; the reward reads the decision, never the words.
"""
from __future__ import annotations

from typing import Callable, Protocol

from shoprl.data.catalog import Product
from shoprl.data.prompts import satisfies
from shoprl.env.scenario import Scenario


def judge_accept(scenario: Scenario, sku: str | None,
                 idx: dict[str, Product]) -> bool:
    """Programmatic accept: True iff the item meets the hidden budget AND the
    hidden must-have. Objective fit only — the reward's `accepted` term and the
    permission gate both read this, nothing else."""
    if not sku:
        return False
    p = idx.get(sku)
    if p is None or p.price > scenario.hidden_budget:
        return False
    if scenario.must_have_key == "brand":
        return p.brand == scenario.must_have_value
    return satisfies(p, {scenario.must_have_key: float(scenario.must_have_value)})


class ConversationModel(Protocol):
    """The generation seam. `utter` phrases a user turn for a given intent; it is
    told the intent (and, for a permission reply, the already-decided accept
    boolean) — it does not decide anything."""

    def utter(self, intent: str, scenario: Scenario,
              accepted: bool | None = None) -> str: ...


class ScriptedConversation:
    """Deterministic templated user turns (no LLM). Reveals a hidden field only
    when the env asks it to (i.e. only in response to a relevant clarifying
    question); phrases accept/reject as already decided by `judge_accept`."""

    def utter(self, intent: str, scenario: Scenario,
              accepted: bool | None = None) -> str:
        if intent == "budget":
            return f"My budget is about ${scenario.hidden_budget:.0f}."
        if intent == "feature":
            return _phrase_must_have(scenario)
        if intent == "permission":
            return ("Yes, please add that one." if accepted
                    else "No, that one doesn't work for me.")
        if intent == "greet":
            return scenario.opening_utterance
        # Irrelevant / unparseable agent turn: reveal nothing.
        return "Could you help me find the right one?"


class FrozenLLMConversation:
    """Drop-in for `ScriptedConversation`, backed by a frozen instruct LLM.

    Same `ConversationModel` interface, so the env, reward, and judge are all
    unchanged — only the WORDS change. Takes a `generate(prompt) -> str` callable
    (a thin wrapper over an HF text-generation pipeline in Phase 2), so it is
    fully testable on CPU with a stub and needs no torch here.

    The seam's key invariant is preserved by construction: the LLM only phrases.
    For a permission reply it is HANDED the already-decided `accepted` boolean
    (from `judge_accept`) and asked to voice it, so the model cannot flip the
    decision — no amount of generated enthusiasm can turn a reject into an
    accept. And a hidden field is put in the prompt only for its matching intent
    (which the env sets after parsing a relevant clarifying question), so the
    "reveal only when asked" rule stays with the env, not the model.
    """

    def __init__(self, generate: Callable[[str], str], system: str | None = None):
        self._generate = generate
        self._system = system or (
            "You are a shopper with a fixed hidden need. Reply in one short, "
            "natural sentence. Only state a fact you are explicitly told to share.")

    def utter(self, intent: str, scenario: Scenario,
              accepted: bool | None = None) -> str:
        return self._generate(self._build_prompt(intent, scenario, accepted)).strip()

    def _build_prompt(self, intent: str, scenario: Scenario,
                      accepted: bool | None) -> str:
        if intent == "budget":
            fact = f"Share that your budget is about ${scenario.hidden_budget:.0f}."
        elif intent == "feature":
            fact = f"Share this requirement: {_phrase_must_have(scenario)}"
        elif intent == "permission":
            fact = ("Agree and ask them to add it." if accepted
                    else "Politely decline; it doesn't fit your need.")
        elif intent == "greet":
            fact = "Greet and say you're looking for a laptop, without details."
        else:
            fact = "Reply vaguely; reveal no specific requirement."
        return f"{self._system}\nInstruction: {fact}\nUser:"


def _phrase_must_have(scenario: Scenario) -> str:
    k, v = scenario.must_have_key, scenario.must_have_value
    if k == "min_ram":
        return f"It needs at least {int(v)}GB of RAM."
    if k == "max_weight":
        return f"It can't weigh more than {v} lbs."
    if k == "min_battery":
        return f"I need at least {int(v)} hours of battery."
    if k == "brand":
        return f"It has to be a {v}."
    return "That's important to me."
