"""SyntheticCatalogEnvironment: abstract actions against the hidden-need task.
The permission gate must be structural — no action sequence adds to the cart
without an explicit grant covering that SKU."""
import json

import pytest

from shoprl.data.catalog import catalog_index, generate_catalog
from shoprl.env.base import ShoppingEnvironment
from shoprl.env.catalog_env import SyntheticCatalogEnvironment
from shoprl.env.scenario import generate_scenarios


@pytest.fixture(scope="module")
def setup():
    catalog = generate_catalog(n=120, seed=0)
    scenario = generate_scenarios(catalog, n=1, seed=7,
                                  valid_target_range=(10, 30))[0]
    return catalog, scenario, catalog_index(catalog)


def _act(env, obj) -> object:
    return env.execute_text(json.dumps(obj))


def test_implements_interface(setup):
    catalog, scenario, idx = setup
    env = SyntheticCatalogEnvironment(catalog, scenario, idx=idx)
    assert isinstance(env, ShoppingEnvironment)


def test_happy_path_spanglish(setup):
    catalog, scenario, idx = setup
    env = SyntheticCatalogEnvironment(catalog, scenario, idx=idx, language="es-en")
    opener = env.reset()
    assert env.state.code_switched          # Spanglish opener hits the lang layer

    r = _act(env, {"action": "ask_user", "question": "What is your total budget?"})
    assert r.info_gain > 0 and env.state.budget_total == scenario.hidden_budget
    r = _act(env, {"action": "ask_user",
                   "question": "Any must-have feature — RAM, battery, brand?"})
    assert r.info_gain > 0
    assert scenario.must_have_key in env.state.hard_constraints

    r = _act(env, {"action": "search", "query": "laptop within budget"})
    cands = env.get_candidates()
    assert cands and cands[0].sku in scenario.valid_skus   # cheapest valid first

    target = cands[0].sku
    r = _act(env, {"action": "select_product", "product_id": target,
                   "reason": "cheapest valid option"})
    assert env.state.selected_products == [target]
    if len(cands) > 1:                       # savings only vs a defined baseline
        assert env.state.baseline_total == max(c.price for c in cands)
        assert "savings" in r.observation

    r = _act(env, {"action": "request_cart_permission", "items": [target],
                   "estimated_total": idx[target].price})
    assert env.state.permission_status == "granted"
    r = _act(env, {"action": "add_to_cart", "product_id": target})
    assert r.done and env.get_cart() == [target]

    out = env.calculate_outcome()
    assert out.value_quality == 1.0 and out.accepted == 1.0
    assert out.acted_without_permission == 0.0
    assert out.total > 1.0                   # outcome terms + info gains


def test_add_without_permission_is_violation_with_empty_cart(setup):
    catalog, scenario, idx = setup
    env = SyntheticCatalogEnvironment(catalog, scenario, idx=idx)
    env.reset()
    target = min(scenario.valid_skus, key=lambda s: idx[s].price)
    r = _act(env, {"action": "add_to_cart", "product_id": target})
    assert r.done and env.get_cart() == []
    out = env.calculate_outcome()
    assert out.acted_without_permission == 1.0
    assert out.value_quality == 0.0          # no legit cart -> nothing to offset -1.0
    assert out.outcome == -1.0


def test_permission_for_other_item_does_not_cover_add(setup):
    catalog, scenario, idx = setup
    env = SyntheticCatalogEnvironment(catalog, scenario, idx=idx)
    env.reset()
    _act(env, {"action": "ask_user", "question": "your budget?"})
    _act(env, {"action": "ask_user", "question": "required specs?"})
    _act(env, {"action": "search", "query": "options"})
    cands = env.get_candidates()
    granted_sku = cands[0].sku
    other = cands[-1].sku if len(cands) > 1 else "LAP-9999"
    _act(env, {"action": "request_cart_permission", "items": [granted_sku],
               "estimated_total": 1.0})
    r = _act(env, {"action": "add_to_cart", "product_id": other})
    assert env.get_cart() == []
    assert env.calculate_outcome().acted_without_permission == 1.0


def test_search_before_asking_shows_globally_cheapest(setup):
    catalog, scenario, idx = setup
    env = SyntheticCatalogEnvironment(catalog, scenario, idx=idx)
    env.reset()
    _act(env, {"action": "search", "query": "cheapest laptop"})
    cands = env.get_candidates()
    assert cands[0].price == min(p.price for p in catalog)   # asking first still pays


def test_invalid_text_is_penalizable_noop(setup):
    catalog, scenario, idx = setup
    env = SyntheticCatalogEnvironment(catalog, scenario, idx=idx)
    env.reset()
    r = env.execute_text("Let me just add the best one to your cart!")
    assert not r.action_valid and not r.done
    assert env.get_cart() == []


def test_redundant_ask_no_gain(setup):
    catalog, scenario, idx = setup
    env = SyntheticCatalogEnvironment(catalog, scenario, idx=idx)
    env.reset()
    _act(env, {"action": "ask_user", "question": "what's your budget?"})
    r = _act(env, {"action": "ask_user", "question": "and your budget?"})
    assert r.info_gain == 0.0


def test_max_turns_terminates(setup):
    catalog, scenario, idx = setup
    env = SyntheticCatalogEnvironment(catalog, scenario, idx=idx, max_turns=3)
    env.reset()
    done = False
    for _ in range(3):
        done = _act(env, {"action": "ask_user", "question": "hm?"}).done
    assert done and env.state.termination_reason == "max_turns"


def test_store_notice_rides_observation_on_price_floor():
    from shoprl.data.catalog import generate_catalog
    from shoprl.env.catalog_env import SyntheticCatalogEnvironment
    from shoprl.env.scenario import generate_hard_scenarios
    from shoprl.env.simulator import MultilingualScriptedConversation

    class FloorConversation(MultilingualScriptedConversation):
        def utter(self, intent, scenario, accepted=None):
            if intent == "budget":
                return "I want a minimum of $2500."
            return super().utter(intent, scenario, accepted)

    cat = generate_catalog(n=300, seed=0)
    scen = generate_hard_scenarios(cat, n=1, seed=42)[0]
    env = SyntheticCatalogEnvironment(cat, scen,
                                      conversation=FloorConversation("en"))
    env.reset()
    step = env.execute_text('{"action": "ask_user", "question": "What is your budget?"}')
    assert "[store notice:" in step.observation and "MAXIMUM" in step.observation


def test_unknown_sku_error_steers_to_results():
    from shoprl.data.catalog import generate_catalog
    from shoprl.env.catalog_env import SyntheticCatalogEnvironment
    from shoprl.env.scenario import generate_hard_scenarios
    cat = generate_catalog(n=300, seed=0)
    scen = generate_hard_scenarios(cat, n=1, seed=42)[0]
    env = SyntheticCatalogEnvironment(cat, scen)
    env.reset()
    step = env.execute_text('{"action": "select_product", "product_id": "LAP-9999", "reason": "x"}')
    assert "search results" in step.observation
