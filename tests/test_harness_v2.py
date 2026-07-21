"""v2 eval harness: reference policies bound the metrics on every language."""
import pytest

from shoprl.data.catalog import generate_catalog
from shoprl.eval.harness_v2 import evaluate_v2, heldout_hard_scenarios
from shoprl.eval.v2_policies import NoAskBaselineV2, OracleGoodV2, ViolationV2


@pytest.fixture(scope="module")
def world():
    catalog = generate_catalog(n=300, seed=0)
    scenarios = heldout_hard_scenarios(catalog, n=20)
    return catalog, scenarios


def test_oracle_ceiling_all_languages(world):
    catalog, scenarios = world
    for lang in ("en", "es", "es-en"):
        r = evaluate_v2(catalog, scenarios, OracleGoodV2, "oracle", language=lang)
        assert r.task_success_rate == 1.0
        assert r.permission_violation_rate == 0.0
        assert r.mean_value_quality == 1.0
        assert r.ask_rate == 1.0


def test_floors_and_ordering(world):
    catalog, scenarios = world
    oracle = evaluate_v2(catalog, scenarios, OracleGoodV2, "oracle")
    noask = evaluate_v2(catalog, scenarios, NoAskBaselineV2, "noask")
    viol = evaluate_v2(catalog, scenarios, ViolationV2, "violation")
    assert viol.permission_violation_rate == 1.0      # safety floor
    assert viol.task_success_rate == 0.0
    assert noask.ask_rate == 0.0                      # value floor never asks
    assert noask.task_success_rate < oracle.task_success_rate
    assert oracle.mean_total > noask.mean_total > viol.mean_total
    assert "success=1.00" in oracle.as_row()


def test_heldout_seed_disjoint_from_training_pools(world):
    catalog, scenarios = world
    from shoprl.env.scenario import generate_hard_scenarios
    train = {s.scenario_id: s for s in generate_hard_scenarios(catalog, n=20, seed=11)}
    # Same id namespace, different hidden needs -> truly held-out content.
    overlap = [s for s in scenarios if s.scenario_id in train
               and train[s.scenario_id].hidden_constraints == s.hidden_constraints]
    assert not overlap
