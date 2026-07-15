"""Slice B: the user-simulator seam.

Proves the two objects are cleanly separated and swappable:
  - the ConversationModel (generation) can be swapped without changing any
    decision the env/reward make -- the scripted model is a no-regression
    passthrough, and a totally different model yields IDENTICAL decisions;
  - judge_accept (judgment) is rule-based and provably independent of whatever
    the ConversationModel says -- an over-budget item is rejected even when the
    conversation model gushes "yes, add it!".
"""
from __future__ import annotations

from shoprl.data.catalog import catalog_index, generate_catalog
from shoprl.env.pennyenv import PennyEnv
from shoprl.env.scenario import generate_scenarios
from shoprl.env.simulator import (FrozenLLMConversation, ScriptedConversation,
                                  judge_accept)

CAT = generate_catalog(n=400, seed=0)
IDX = catalog_index(CAT)
SCEN = generate_scenarios(CAT, n=1, seed=11)[0]
TARGET = min(SCEN.valid_skus, key=lambda s: IDX[s].price)
OVER_BUDGET = next(p.sku for p in CAT if p.price > SCEN.hidden_budget)

_GOOD = ["ASK[budget]", "ASK[feature]", f"RECOMMEND[{TARGET}]",
         "ASK_PERMISSION", f"ADD_TO_CART[{TARGET}]"]


class LiarConversation:
    """A ConversationModel that ALWAYS gushes acceptance, for any intent."""
    def utter(self, intent, scenario, accepted=None):
        return "YES!! Absolutely, add it, that's perfect, I love it!"


def _run(actions, conversation):
    env = PennyEnv(CAT, SCEN, idx=IDX, max_turns=16, conversation=conversation)
    env.reset()
    for a in actions:
        env.step(a)
    return env


def test_scripted_passthrough_is_a_no_regression():
    """Default env (implicit ScriptedConversation) == explicit one."""
    default = PennyEnv(CAT, SCEN, idx=IDX)
    assert isinstance(default.conversation, ScriptedConversation)

    r_default = _run(_GOOD, None).reward()            # None -> ScriptedConversation
    r_explicit = _run(_GOOD, ScriptedConversation()).reward()
    assert r_default == r_explicit
    assert r_default.value_quality == 1.0 and r_default.total > 1.0

    sc = ScriptedConversation()
    assert sc.utter("budget", SCEN) == f"My budget is about ${SCEN.hidden_budget:.0f}."


def test_decisions_are_identical_across_conversation_models():
    """Swapping ScriptedConversation for the Liar changes only the words, never
    the reward's decision fields."""
    scripted = _run(_GOOD, ScriptedConversation())
    liar = _run(_GOOD, LiarConversation())
    a, b = scripted.reward(), liar.reward()
    assert (a.value_quality, a.accepted, a.asked_permission,
            a.acted_without_permission) == (b.value_quality, b.accepted,
                                            b.asked_permission,
                                            b.acted_without_permission)
    assert scripted.state.accepted == liar.state.accepted is True


def test_judge_is_independent_of_the_conversation_model():
    # Direct: judgment is purely rule-based.
    assert judge_accept(SCEN, OVER_BUDGET, IDX) is False
    assert judge_accept(SCEN, TARGET, IDX) is True

    # In the loop: the Liar gushes acceptance for an OVER-BUDGET item, yet the
    # env still denies permission and the add is a violation.
    actions = [f"RECOMMEND[{OVER_BUDGET}]", "ASK_PERMISSION",
               f"ADD_TO_CART[{OVER_BUDGET}]"]
    env = _run(actions, LiarConversation())
    assert env.state.accepted is False
    assert env.state.acted_without_permission is True
    assert env.reward().value_quality == 0.0


def test_frozen_llm_conversation_drops_in_via_the_interface():
    """A non-scripted ConversationModel (LLM-shaped, stubbed generate) works
    through the same interface; the env's decisions are unchanged, and the
    hidden budget is surfaced to the model only via the budget intent's prompt."""
    seen_prompts = []

    def fake_generate(prompt: str) -> str:
        seen_prompts.append(prompt)
        return "  (some natural user reply)  "

    llm = FrozenLLMConversation(fake_generate)
    env = _run(_GOOD, llm)
    r = env.reward()
    # Same decisions as scripted despite totally different words.
    assert r.value_quality == 1.0 and r.acted_without_permission == 0.0
    assert env.state.accepted is True
    # The budget value reaches the model only through the budget-intent prompt.
    assert any(f"${SCEN.hidden_budget:.0f}" in p for p in seen_prompts)
    # utter returns the (stripped) generated text.
    assert llm.utter("other", SCEN) == "(some natural user reply)"
