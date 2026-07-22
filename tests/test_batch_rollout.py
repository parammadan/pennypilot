"""Batched rollouts: lockstep semantics — per-episode conversations stay
isolated, finished episodes leave the batch, results match sequential."""
import json

import pytest

from shoprl.data.catalog import catalog_index, generate_catalog
from shoprl.env.catalog_env import SyntheticCatalogEnvironment
from shoprl.env.scenario import generate_hard_scenarios
from shoprl.train.batch_rollout import batched_rollouts


@pytest.fixture(scope="module")
def world():
    catalog = generate_catalog(n=150, seed=0)
    scens = generate_hard_scenarios(catalog, n=4, seed=11)
    return catalog, scens, catalog_index(catalog)


def _oracle_batch_fn(envs_by_msgs):
    """Scripted batch 'engine': plays the oracle for whichever episode each
    message-list belongs to, keyed by the opener + turn count."""
    def fn(message_lists):
        outs = []
        for msgs in message_lists:
            env = envs_by_msgs[id(msgs)]
            n_agent = sum(m["role"] == "assistant" for m in msgs)
            n_feat = len(env.scenario.all_must_haves)
            if n_agent == 0:
                outs.append(json.dumps({"action": "ask_user",
                                        "question": "What is your budget?"}))
            elif n_agent <= n_feat:
                outs.append(json.dumps({"action": "ask_user",
                                        "question": "any other must-have specs?"}))
            elif n_agent == n_feat + 1:
                outs.append(json.dumps({"action": "search", "query": "options"}))
            else:
                cands = env.get_candidates()
                t = cands[0].sku
                seq = [{"action": "select_product", "product_id": t, "reason": "cheapest"},
                       {"action": "request_cart_permission", "items": [t],
                        "estimated_total": cands[0].price},
                       {"action": "add_to_cart", "product_id": t}]
                outs.append(json.dumps(seq[min(n_agent - n_feat - 2, 2)]))
        return outs
    return fn


def test_batched_rollouts_all_succeed_and_stay_isolated(world):
    catalog, scens, idx = world
    envs = [SyntheticCatalogEnvironment(catalog, s, idx=idx, language="es-en")
            for s in scens]
    registry = {}

    # Message lists are mutated in place by the driver, so id(msgs) is stable
    # across rounds; on round 1 the live order equals env order — map by
    # position once, then look up by identity forever after.
    def fn(message_lists):
        if not registry:
            for env, msgs in zip(envs, message_lists):
                registry[id(msgs)] = env
        return _oracle_batch_fn(registry)(message_lists)

    trajs = batched_rollouts(fn, envs, system="You are a shopping agent.")
    assert len(trajs) == 4
    for t, env, scen in zip(trajs, envs, scens):
        assert t.value_quality == 1.0, scen.scenario_id
        assert not t.violation
        assert env.get_cart() and env.get_cart()[0] in scen.valid_skus
    # Different scenarios -> different discovered budgets (no state bleed).
    budgets = {e.state.budget_total for e in envs}
    assert len(budgets) == len({s.hidden_budget for s in scens})


def test_finished_episodes_leave_the_batch(world):
    catalog, scens, idx = world
    envs = [SyntheticCatalogEnvironment(catalog, scens[0], idx=idx),
            SyntheticCatalogEnvironment(catalog, scens[1], idx=idx)]
    sizes = []

    def fn(message_lists):
        sizes.append(len(message_lists))
        outs = []
        for msgs in message_lists:
            n_agent = sum(m["role"] == "assistant" for m in msgs)
            # First env finishes immediately (permissionless add -> done+violation);
            # second keeps asking until max_steps.
            if msgs[1]["content"] == getattr(envs[0], "opener", None) and n_agent == 0:
                outs.append(json.dumps({"action": "add_to_cart",
                                        "product_id": "LAP-0001"}))
            else:
                outs.append(json.dumps({"action": "ask_user", "question": "hm?"}))
        return outs

    trajs = batched_rollouts(fn, envs, system="sys", max_steps=5)
    assert sizes[0] == 2 and sizes[-1] == 1     # batch shrank after episode 1 ended
    assert trajs[0].violation and trajs[0].turns == 1
    assert trajs[1].turns == 5
