"""Hard scenarios (multi-constraint hidden needs) — the Scenario Hardness
Gate's subject. Key property: partial discovery is NOT enough — the cheapest
item shown before the last constraint is revealed frequently violates a
still-hidden one (the distractor structure)."""
import pytest

from shoprl.data.catalog import catalog_index, generate_catalog
from shoprl.env.catalog_env import SyntheticCatalogEnvironment
from shoprl.env.scenario import generate_hard_scenarios, generate_scenarios
from shoprl.env.simulator import judge_accept
from shoprl.eval.v2_policies import OracleGoodV2


@pytest.fixture(scope="module")
def world():
    catalog = generate_catalog(n=300, seed=0)
    hard = generate_hard_scenarios(catalog, n=40, seed=11)
    return catalog, hard, catalog_index(catalog)


def test_generation_shape_and_validity(world):
    catalog, hard, idx = world
    assert len(hard) >= 35                      # near-all sampled successfully
    for s in hard:
        assert 2 <= len(s.all_must_haves) <= 3
        assert s.valid_skus                     # satisfiable
        assert len(s.valid_skus) <= 25          # small valid sets = hard ranking
        for sku in s.valid_skus:                # every valid SKU truly qualifies
            assert judge_accept(s, sku, idx)


def test_budget_stays_binding(world):
    catalog, hard, idx = world
    from shoprl.data.prompts import satisfies

    def feat_ok(p, s):
        return all((p.brand == v) if k == "brand" else satisfies(p, {k: float(v)})
                   for k, v in s.all_must_haves.items())

    for s in hard:
        # At least one item meets every must-have but is priced out -> asking
        # about budget always pays (judge_accept can't express this: it folds
        # the budget in).
        over = [p for p in catalog if p.price > s.hidden_budget and feat_ok(p, s)]
        assert over, f"{s.scenario_id}: budget not binding"


def test_partial_discovery_yields_distractors(world):
    """After discovering budget + primary only, the cheapest visible item must
    violate a hidden extra in a healthy fraction of scenarios — otherwise the
    task degenerates to v1 (pick #1) and the hardness gate will fail high."""
    catalog, hard, idx = world
    distractor = 0
    for s in hard:
        env = SyntheticCatalogEnvironment(catalog, s, idx=idx)
        env.reset()
        env.execute_text('{"action": "ask_user", "question": "your budget?"}')
        env.execute_text('{"action": "ask_user", "question": "any required specs?"}')
        env.execute_text('{"action": "search", "query": "options"}')
        top = env.get_candidates()[0].sku
        if top not in s.valid_skus:
            distractor += 1
    assert distractor / len(hard) >= 0.3, (
        f"only {distractor}/{len(hard)} scenarios have a partial-discovery "
        "distractor — task too easy")


def test_full_discovery_makes_top_item_valid(world):
    catalog, hard, idx = world
    for s in hard[:10]:
        env = SyntheticCatalogEnvironment(catalog, s, idx=idx)
        env.reset()
        env.execute_text('{"action": "ask_user", "question": "your budget?"}')
        for _ in s.all_must_haves:
            env.execute_text('{"action": "ask_user", "question": "any other must-have specs?"}')
        env.execute_text('{"action": "search", "query": "options"}')
        cands = env.get_candidates()
        assert cands and cands[0].sku in s.valid_skus
        assert cands[0].price == min(idx[v].price for v in s.valid_skus)


def test_each_reveal_earns_info_gain(world):
    catalog, hard, idx = world
    s = next(x for x in hard if len(x.all_must_haves) == 3)
    env = SyntheticCatalogEnvironment(catalog, s, idx=idx)
    env.reset()
    gains = []
    gains.append(env.execute_text(
        '{"action": "ask_user", "question": "your budget?"}').info_gain)
    for _ in range(3):
        gains.append(env.execute_text(
            '{"action": "ask_user", "question": "any required specs?"}').info_gain)
    assert all(g > 0 for g in gains)            # four reveals, four payoffs
    extra = env.execute_text(
        '{"action": "ask_user", "question": "more specs?"}')
    assert extra.info_gain == 0.0               # exhausted -> redundant


def test_oracle_solves_hard_scenarios(world):
    catalog, hard, idx = world
    for s in hard[:10]:
        env = SyntheticCatalogEnvironment(catalog, s, idx=idx, language="es-en")
        policy = OracleGoodV2()
        env.reset()
        policy.reset(s, idx)
        done = False
        steps = 0
        while not done and steps < 15:
            done = env.execute_text(policy.act()).done
            steps += 1
        out = env.calculate_outcome()
        assert out.value_quality == 1.0 and out.accepted == 1.0
        assert out.acted_without_permission == 0.0


def test_v1_scenarios_unchanged(world):
    catalog, _, idx = world
    v1 = generate_scenarios(catalog, n=5, seed=3)
    for s in v1:
        assert s.extra_must_haves == {}
        assert len(s.all_must_haves) == 1
