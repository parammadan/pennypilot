"""Phase-1 tests for the SFT warmup dialogue generator.

Pins the three properties that make the demos usable warmup data:
  1. well-formed: every agent action is in the grammar and references real SKUs;
  2. internally correct: replaying a demo's agent actions through PennyEnv
     legitimately adds the genuinely cheapest valid item (value_quality == 1.0,
     no permission violation) — the demo is not just plausible-looking;
  3. structurally varied: orderings, clarify-turn counts, rejection+recovery, and
     the permission-edge case are all present (not one rigid template).
"""
from __future__ import annotations

import re

import pytest

from shoprl.data.catalog import catalog_index, generate_catalog
from shoprl.data.sft import (cheapest_valid, demo_stats, generate_sft_dialogues)
from shoprl.env.pennyenv import PennyEnv
from shoprl.env.scenario import generate_scenarios

N = 300
SEED = 0
_GRAMMAR = re.compile(
    r"^(ASK\[(budget|feature)\]|RECOMMEND\[LAP-\d{4}\]|ASK_PERMISSION|ADD_TO_CART\[LAP-\d{4}\])$")


@pytest.fixture(scope="module")
def fixture():
    cat = generate_catalog(n=500, seed=SEED)
    idx = catalog_index(cat)
    demos = generate_sft_dialogues(cat, n=N, seed=SEED)
    # Same catalog+seed+n reproduce the exact scenarios the generator used.
    scen_by_id = {s.scenario_id: s for s in generate_scenarios(cat, n=N, seed=SEED)}
    return cat, idx, demos, scen_by_id


def test_agent_turns_are_well_formed_grammar(fixture):
    cat, idx, demos, _ = fixture
    skus = set(idx)
    for d in demos:
        for t in d.turns:
            if t.role != "agent" or t.action is None:
                continue  # user turns + the NL refrain turn are free text
            assert _GRAMMAR.match(t.text), f"bad agent action: {t.text!r}"
            m = re.search(r"\[(LAP-\d{4})\]", t.text)
            if m:
                assert m.group(1) in skus, f"unknown SKU {m.group(1)}"


def test_dialogues_alternate_and_open_with_user(fixture):
    _, _, demos, _ = fixture
    for d in demos:
        assert d.turns[0].role == "user"       # the vague opener
        # roles never repeat 3x in a row (agent may double up recommend/ask,
        # but the conversation stays a conversation).
        roles = [t.role for t in d.turns]
        assert not any(roles[i] == roles[i + 1] == roles[i + 2]
                       for i in range(len(roles) - 2))


def _replay(demo, cat, idx, scen):
    env = PennyEnv(cat, scen, idx=idx, max_turns=32)
    env.reset()
    for t in demo.turns:
        if t.role == "agent":
            env.step(t.text)
    return env


def test_recommended_item_is_the_cheapest_valid(fixture):
    cat, idx, demos, scen_by_id = fixture
    for d in demos:
        scen = scen_by_id[d.scenario_id]
        assert d.target_sku == cheapest_valid(scen, idx)


def test_positive_demos_legitimately_add_cheapest_valid(fixture):
    cat, idx, demos, scen_by_id = fixture
    checked = 0
    for d in demos:
        if d.kind == "permission_edge":
            continue
        scen = scen_by_id[d.scenario_id]
        env = _replay(d, cat, idx, scen)
        r = env.reward()
        assert env.state.cart == [d.target_sku], d.scenario_id
        assert r.value_quality == 1.0            # cheapest valid -> full marks
        assert r.acted_without_permission == 0.0
        assert r.accepted == 1.0
        checked += 1
    assert checked > 0


def test_permission_edge_demos_never_add(fixture):
    cat, idx, demos, scen_by_id = fixture
    edge = [d for d in demos if d.kind == "permission_edge"]
    assert edge, "expected some permission-edge demos"
    for d in edge:
        assert not any(t.action and t.action.startswith("ADD_TO_CART")
                       for t in d.turns)
        env = _replay(d, cat, idx, scen_by_id[d.scenario_id])
        assert env.state.acted_without_permission is False
        assert env.state.cart == []
        assert env.state.recommended == d.target_sku  # still the right pick


def test_structural_variety(fixture):
    _, _, demos, _ = fixture
    s = demo_stats(demos)
    assert set(s["by_order"]) == {"budget_first", "feature_first"}
    assert len(s["by_n_clarify"]) >= 2          # e.g. 2 and 3 clarifying turns
    assert set(s["by_kind"]) >= {"positive", "rejection_recovery", "permission_edge"}
    assert s["with_rejection"] > 0
    assert s["turn_len_max"] > s["turn_len_min"]  # not one rigid length
