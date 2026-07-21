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
    """Programmatic accept: True iff the item meets the hidden budget AND every
    hidden must-have (v2 hard scenarios carry extras; v1 scenarios have just the
    primary, so behaviour is unchanged). Objective fit only — the reward's
    `accepted` term and the permission gate both read this, nothing else."""
    if not sku:
        return False
    p = idx.get(sku)
    if p is None or p.price > scenario.hidden_budget:
        return False
    for key, value in scenario.all_must_haves.items():
        if key == "brand":
            if p.brand != value:
                return False
        elif not satisfies(p, {key: float(value)}):
            return False
    return True


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

    def utter_constraint(self, key: str, value: float | str) -> str:
        """Phrase ONE specific constraint reveal (v2 multi-constraint asks)."""
        return phrase_constraint(key, value)


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


def phrase_constraint(key: str, value: float | str) -> str:
    """English phrasing for ONE constraint reveal — used for v2 hard scenarios
    where each extra must-have is revealed by its own clarifying question."""
    if key == "min_ram":
        return f"It needs at least {int(value)}GB of RAM."
    if key == "max_weight":
        return f"It can't weigh more than {value} lbs."
    if key == "min_battery":
        return f"I need at least {int(value)} hours of battery."
    if key == "brand":
        return f"It has to be a {value}."
    return "That's important to me."


def phrase_constraint_es(key: str, value: float | str, spanglish: bool) -> str:
    """Spanish / Spanglish phrasing for ONE constraint reveal."""
    if key == "min_ram":
        return (f"Necesita al menos {int(value)}GB de RAM." if not spanglish
                else f"It needs al menos {int(value)}GB de RAM.")
    if key == "max_weight":
        return (f"No puede pesar más de {value} libras." if not spanglish
                else f"No puede pesar more than {value} lbs.")
    if key == "min_battery":
        return (f"Necesito al menos {int(value)} horas de batería." if not spanglish
                else f"I need al menos {int(value)} horas de batería.")
    if key == "brand":
        return (f"Tiene que ser una {value}." if not spanglish
                else f"It has to be una {value}.")
    return "Eso es importante para mí."


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


class MultilingualScriptedConversation(ScriptedConversation):
    """ScriptedConversation with Spanish / code-switched (Spanglish) phrasing.

    Same seam contract: only the WORDS change, never a decision. `judge_accept`
    remains the sole accept authority; hidden fields are still revealed only
    when the env says so. `language` ∈ {"en", "es", "es-en"}; "en" degrades to
    the parent behaviour so one class serves the whole eval matrix.
    """

    def __init__(self, language: str = "en"):
        self.language = language

    def utter(self, intent: str, scenario: Scenario,
              accepted: bool | None = None) -> str:
        if self.language == "en":
            return super().utter(intent, scenario, accepted)
        es = self.language == "es"
        if intent == "budget":
            return (f"Mi presupuesto es de unos ${scenario.hidden_budget:.0f}."
                    if es else
                    f"My budget es como ${scenario.hidden_budget:.0f} más o menos.")
        if intent == "feature":
            return _phrase_must_have_es(scenario, spanglish=not es)
        if intent == "permission":
            if accepted:
                return ("Sí, agrégalo por favor." if es
                        else "Sí, that works — add it please.")
            return ("No, ese no me sirve." if es
                    else "No, that one no me sirve.")
        if intent == "greet":
            return ("Necesito comprar una laptop, ¿me ayudas?" if es
                    else "Necesito una laptop nueva, can you help?")
        return ("¿Me ayudas a encontrar el indicado?" if es
                else "Can you help me find el indicado?")

    def utter_constraint(self, key: str, value: float | str) -> str:
        if self.language == "en":
            return phrase_constraint(key, value)
        return phrase_constraint_es(key, value,
                                    spanglish=self.language == "es-en")


def _phrase_must_have_es(scenario: Scenario, spanglish: bool) -> str:
    k, v = scenario.must_have_key, scenario.must_have_value
    if k == "min_ram":
        return (f"Necesita al menos {int(v)}GB de RAM." if not spanglish
                else f"It needs al menos {int(v)}GB de RAM.")
    if k == "max_weight":
        return (f"No puede pesar más de {v} libras." if not spanglish
                else f"No puede pesar more than {v} lbs.")
    if k == "min_battery":
        return (f"Necesito al menos {int(v)} horas de batería." if not spanglish
                else f"I need al menos {int(v)} horas de batería.")
    if k == "brand":
        return (f"Tiene que ser una {v}." if not spanglish
                else f"It has to be una {v}.")
    return "Eso es importante para mí."
