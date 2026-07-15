"""End-to-end Phase-1 test for the Pennywise novelty slice.

Drives the conversational env with a scripted agent and pins the behaviours the
reward is supposed to induce:
  - a GOOD trajectory (ask -> discover hidden need -> recommend cheapest valid ->
    ask permission -> add after accept) scores high, with positive info-gain;
  - a BAD trajectory (add without ever getting permission) scores low, with the
    −1.0 permission floor firing;
  - the accept/reject decision is the programmatic judge, not the words.
"""
from __future__ import annotations

from shoprl.data.catalog import catalog_index, generate_catalog
from shoprl.env.pennyenv import PennyEnv
from shoprl.env.scenario import generate_scenarios
from shoprl.env.simulator import judge_accept


def _fixture():
    cat = generate_catalog(n=400, seed=0)
    idx = catalog_index(cat)
    scen = generate_scenarios(cat, n=1, seed=11)[0]
    return cat, idx, scen


def _cheapest_valid(scen, idx) -> str:
    return min(scen.valid_skus, key=lambda s: idx[s].price)


def test_good_trajectory_scores_high_with_info_gain():
    cat, idx, scen = _fixture()
    target = _cheapest_valid(scen, idx)
    env = PennyEnv(cat, scen, idx=idx, max_turns=8)
    env.reset()
    env.step("ASK[budget]")
    env.step("ASK[feature]")
    env.step(f"RECOMMEND[{target}]")
    env.step("ASK_PERMISSION")
    _, done, _ = env.step(f"ADD_TO_CART[{target}]")

    r = env.reward()
    assert done
    assert env.state.cart == [target]
    assert r.acted_without_permission == 0.0
    assert r.value_quality == 1.0        # cheapest valid -> full marks
    assert r.accepted == 1.0
    assert r.asked_permission == 1.0
    assert r.info_gain > 0.0             # asking revealed the hidden need
    assert r.outcome == 1.0              # 0.4 + 0.4 + 0.2
    assert r.total > 1.0                 # outcome + info-gain


def test_add_without_permission_fires_floor():
    cat, idx, scen = _fixture()
    target = _cheapest_valid(scen, idx)
    env = PennyEnv(cat, scen, idx=idx, max_turns=8)
    env.reset()
    env.step(f"RECOMMEND[{target}]")
    _, done, _ = env.step(f"ADD_TO_CART[{target}]")  # never asked permission

    r = env.reward()
    assert done
    assert env.state.cart == []                 # illegit add is not carted
    assert r.acted_without_permission == 1.0
    assert r.value_quality == 0.0               # no legit purchase
    assert r.accepted == 0.0
    assert r.total <= -0.9                       # −1.0 floor dominates


def test_good_beats_bad_by_a_wide_margin():
    cat, idx, scen = _fixture()
    target = _cheapest_valid(scen, idx)

    good = PennyEnv(cat, scen, idx=idx)
    good.reset()
    for a in ["ASK[budget]", "ASK[feature]", f"RECOMMEND[{target}]",
              "ASK_PERMISSION", f"ADD_TO_CART[{target}]"]:
        good.step(a)

    bad = PennyEnv(cat, scen, idx=idx)
    bad.reset()
    bad.step(f"ADD_TO_CART[{target}]")

    assert good.reward().total - bad.reward().total > 1.5


def test_guessing_an_invalid_item_is_rejected_by_judge():
    """An over-budget item is rejected by the programmatic judge regardless of
    the conversation — asking permission for it does not grant it."""
    cat, idx, scen = _fixture()
    over_budget = next(p.sku for p in cat if p.price > scen.hidden_budget)
    assert judge_accept(scen, over_budget, idx) is False

    env = PennyEnv(cat, scen, idx=idx)
    env.reset()
    env.step(f"RECOMMEND[{over_budget}]")
    env.step("ASK_PERMISSION")               # judge denies
    env.step(f"ADD_TO_CART[{over_budget}]")  # -> violation, not carted

    r = env.reward()
    assert env.state.accepted is False
    assert r.acted_without_permission == 1.0
    assert r.value_quality == 0.0


def test_redundant_ask_yields_no_info_gain():
    cat, idx, scen = _fixture()
    env = PennyEnv(cat, scen, idx=idx)
    env.reset()
    _, _, first = env.step("ASK[budget]")
    _, _, second = env.step("ASK[budget]")   # already known -> no new info
    assert first.info_gain > 0.0
    assert second.info_gain == 0.0
