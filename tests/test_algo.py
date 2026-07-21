"""Algorithm seams + v2 rollout: pure-python advantage math (the ONLY thing
the pre-registered comparison varies) and the engine-agnostic rollout loop."""
import json

import pytest

from shoprl.data.catalog import catalog_index, generate_catalog
from shoprl.env.catalog_env import SyntheticCatalogEnvironment
from shoprl.env.scenario import generate_hard_scenarios
from shoprl.train.algo import ALGORITHMS, GRPO, GRPONoStd, RLOO, rollout_v2


def test_rloo_leave_one_out_math():
    advs = RLOO().advantages([1.0, 0.0, 0.0, 0.0])
    assert advs[0] == pytest.approx(1.0)          # 1 - mean(0,0,0)
    assert advs[1] == pytest.approx(-1 / 3)       # 0 - mean(1,0,0)
    assert sum(advs) == pytest.approx(4 / 3 * 1 - 1)  # sums are consistent


def test_flat_group_gives_zero_update_for_all_algos():
    flat = [0.85] * 8
    for name, algo in ALGORITHMS.items():
        assert algo.advantages(flat) == pytest.approx([0.0] * 8), name


def test_grpo_std_division_amplifies_small_spread():
    """The H1 mechanism, in miniature: identical tiny spread, GRPO's std
    division blows it up to O(1) advantages while RLOO/no-std keep it small."""
    tiny_spread = [0.850, 0.851, 0.849, 0.850]
    grpo = GRPO().advantages(tiny_spread)
    nostd = GRPONoStd().advantages(tiny_spread)
    rloo = RLOO().advantages(tiny_spread)
    assert max(abs(a) for a in grpo) > 1.0        # amplified
    assert max(abs(a) for a in nostd) < 0.01      # stays proportional
    assert max(abs(a) for a in rloo) < 0.01


def test_rollout_v2_with_scripted_agent():
    catalog = generate_catalog(n=150, seed=0)
    idx = catalog_index(catalog)
    scen = generate_hard_scenarios(catalog, n=1, seed=11)[0]
    env = SyntheticCatalogEnvironment(catalog, scen, idx=idx, language="es-en")

    plan = ([{"action": "ask_user", "question": "What is your total budget?"}]
            + [{"action": "ask_user", "question": "any other must-have specs?"}
               for _ in scen.all_must_haves]
            + [{"action": "search", "query": "options"}])
    state = {"i": 0}

    def agent_fn(messages):
        assert messages[0]["role"] == "system"
        i = state["i"]
        state["i"] += 1
        if i < len(plan):
            return json.dumps(plan[i])
        cands = env.get_candidates()
        target = cands[0].sku
        follow = [{"action": "select_product", "product_id": target, "reason": "cheapest"},
                  {"action": "request_cart_permission", "items": [target],
                   "estimated_total": cands[0].price},
                  {"action": "add_to_cart", "product_id": target}]
        return json.dumps(follow[min(i - len(plan), 2)])

    traj = rollout_v2(agent_fn, env, system="You are a shopping agent.")
    assert traj.value_quality == 1.0 and not traj.violation
    assert traj.reward > 1.0 and traj.asked == 1 + len(scen.all_must_haves)
    assert traj.messages[0]["role"] == "system"
    # messages alternate assistant/user after the opener — the training mask
    # machinery consumes exactly this shape.
    roles = [m["role"] for m in traj.messages]
    assert roles[1] == "user" and "assistant" in roles


def test_identical_hidden_state_across_group():
    """Group rollouts must reset to the SAME hidden scenario or the advantage
    baseline is meaningless."""
    catalog = generate_catalog(n=150, seed=0)
    idx = catalog_index(catalog)
    scen = generate_hard_scenarios(catalog, n=1, seed=11)[0]
    rewards = []
    for _ in range(3):
        env = SyntheticCatalogEnvironment(catalog, scen, idx=idx)
        env.reset()
        env.execute_text('{"action": "ask_user", "question": "budget?"}')
        rewards.append(env.state.budget_total)
    assert rewards[0] == rewards[1] == rewards[2] == scen.hidden_budget
