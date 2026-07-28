"""AdaptiveCustomer: reactive (not pre-scripted), seeded, reasons logged."""
import json

from shoprl.actions import parse_agent_action
from shoprl.data.catalog import generate_catalog
from shoprl.env.catalog_env import SyntheticCatalogEnvironment
from shoprl.env.customer_sim import AdaptiveCustomer
from shoprl.env.scenario import generate_hard_scenarios


def _setup(seed=0):
    cat = generate_catalog(n=300, seed=0)
    scen = generate_hard_scenarios(cat, n=1, seed=11)[0]
    sim = AdaptiveCustomer(scen, seed=seed)
    env = SyntheticCatalogEnvironment(cat, scen, max_turns=32,
                                      conversation=sim)
    env.reset()
    return cat, scen, sim, env


def _step(env, sim, action_json):
    r = parse_agent_action(action_json)
    step = env.execute_text(action_json)
    move = sim.post_step(r, step, env)
    env.pending_events = []
    return step, move


def _polite_agent(env, sim, scen):
    """Discovers everything, recommends valid — the GOOD agent."""
    moves = []
    _, m = _step(env, sim, '{"action": "ask_user", "question": "What is your budget?"}')
    moves.append(m)
    for _ in scen.all_must_haves:
        _, m = _step(env, sim, '{"action": "ask_user", "question": "Any must-have features?"}')
        moves.append(m)
    _, m = _step(env, sim, '{"action": "search", "query": "laptops"}')
    moves.append(m)
    pick = env.get_candidates()[0].sku
    _, m = _step(env, sim, json.dumps({"action": "select_product",
                                       "product_id": pick, "reason": "x"}))
    moves.append(m)
    return moves


def test_vague_opener_reveals_nothing():
    _, _, sim, env = _setup()
    assert env.opener == "I need a laptop for programming."
    assert env._discovered == set()


def test_good_agent_no_corrections():
    _, scen, sim, env = _setup()
    moves = _polite_agent(env, sim, scen)
    assert all(m.action == "PROCEED" for m in moves)
    assert sim.corrections == 0 and sim.frustration == 0


def test_premature_rec_triggers_correction_with_reason_and_reveal():
    cat, scen, sim, env = _setup()
    # rude agent: searches immediately, recommends globally-cheapest item
    _step(env, sim, '{"action": "search", "query": "laptops"}')
    bad = env.get_candidates()[0].sku      # ignores every hidden constraint
    step, move = _step(env, sim, json.dumps(
        {"action": "select_product", "product_id": bad, "reason": "cheap"}))
    assert move.action == "CORRECT"
    assert move.reason.startswith("AGENT_IGNORED_")
    assert move.trigger_event_ids, "correction cites the triggering event"
    key = move.violated_key
    env.reveal_constraint(key)             # what the driver does on CORRECT
    assert key in env._discovered
    assert any(e["type"] == "constraint_revealed"
               and e["attributes"].get("via") == "correction"
               for e in env.pending_events)
    assert sim.log[-1]["reason"] == move.reason
    assert sim.log[-1]["simulator_policy_version"] == "customer-sim-v1"


def test_repeated_question_frustration_then_abandon():
    _, _, sim, env = _setup()
    _step(env, sim, '{"action": "ask_user", "question": "What is your budget?"}')
    moves = []
    for _ in range(4):                     # same question again and again
        _, m = _step(env, sim, '{"action": "ask_user", "question": "What is your budget?"}')
        moves.append(m)
    assert any(m.action == "ABANDON" for m in moves)
    reasons = [l["reason"] for l in sim.log]
    assert "AGENT_REPEATED_KNOWN_QUESTION" in reasons
    assert "EXCESSIVE_FRICTION" in reasons


def test_reactivity_not_prescripted_and_seeded():
    # SAME customer seed, DIFFERENT agent behavior -> different customer trace
    _, scen, sim_a, env_a = _setup(seed=5)
    _polite_agent(env_a, sim_a, scen)
    _, _, sim_b, env_b = _setup(seed=5)
    _step(env_b, sim_b, '{"action": "search", "query": "laptops"}')
    bad = env_b.get_candidates()[0].sku
    _step(env_b, sim_b, json.dumps({"action": "select_product",
                                    "product_id": bad, "reason": "x"}))
    assert [l["customer_action"] for l in sim_a.log] != \
           [l["customer_action"] for l in sim_b.log]
    # same seed + same agent behavior -> identical trace (determinism)
    _, scen2, sim_c, env_c = _setup(seed=5)
    _step(env_c, sim_c, '{"action": "search", "query": "laptops"}')
    bad2 = env_c.get_candidates()[0].sku
    _step(env_c, sim_c, json.dumps({"action": "select_product",
                                    "product_id": bad2, "reason": "x"}))
    assert [l["reason"] for l in sim_b.log] == [l["reason"] for l in sim_c.log]
