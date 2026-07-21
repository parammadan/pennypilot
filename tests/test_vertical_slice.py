"""THE VERTICAL SLICE (first deliverable, gates all expansion):

    Spanglish user request → English assistant clarification → updated
    structured state → product search → budget-aware recommendation →
    explicit permission request → simulated add-to-cart → transcript and
    metrics output.

Runs the oracle policy over the Spanglish SyntheticCatalogEnvironment and
asserts every link in that chain, plus the three reference policies bounding
the metrics exactly as in v1 (ceiling / value floor / safety floor).
"""
import pytest

from shoprl.data.catalog import catalog_index, generate_catalog
from shoprl.env.catalog_env import SyntheticCatalogEnvironment
from shoprl.env.scenario import generate_scenarios
from shoprl.eval.v2_policies import NoAskBaselineV2, OracleGoodV2, ViolationV2
from shoprl.transcript import render_transcript, transcript_record


@pytest.fixture(scope="module")
def world():
    catalog = generate_catalog(n=150, seed=0)
    scenarios = generate_scenarios(catalog, n=5, seed=42,
                                   valid_target_range=(10, 30))
    return catalog, scenarios, catalog_index(catalog)


def _run(env, policy, scenario, idx, max_steps=12):
    env.reset(scenario)
    policy.reset(scenario, idx)
    done = False
    steps = 0
    while not done and steps < max_steps:
        done = env.execute_text(policy.act()).done
        steps += 1
    return env.calculate_outcome()


def test_vertical_slice_end_to_end(world):
    catalog, scenarios, idx = world
    scenario = scenarios[0]
    env = SyntheticCatalogEnvironment(catalog, scenario, idx=idx, language="es-en")
    policy = OracleGoodV2()
    out = _run(env, policy, scenario, idx)

    # 1. Spanglish user request hit the language layer.
    assert env.state.code_switched
    assert "spanish" in env.state.detected_languages
    # 2. Clarifications updated the structured state from hidden ground truth.
    assert env.state.budget_total == scenario.hidden_budget
    assert scenario.must_have_key in env.state.hard_constraints
    # 3. Search grounded the candidates in the env, cheapest-valid first.
    assert env.state.candidate_products
    assert env.state.candidate_products[0] in scenario.valid_skus
    # 4. Budget-aware recommendation: cheapest valid, within budget.
    sel = env.state.selected_products
    assert sel and idx[sel[0]].price <= scenario.hidden_budget
    assert out.value_quality == 1.0
    # 5. Explicit permission requested and granted before any cart action.
    assert env.state.permission_status == "granted"
    assert out.asked_permission == 1.0 and out.accepted == 1.0
    # 6. Simulated add-to-cart, no violation.
    assert env.get_cart() == sel
    assert out.acted_without_permission == 0.0
    # 7. Transcript + metrics artifacts render.
    text = render_transcript(env, out)
    assert "code-switched" in text and "add_to_cart" in text
    assert "reward: total=" in text and "permission=granted" in text
    rec = transcript_record(env, out)
    assert rec["scenario_id"] == scenario.scenario_id and rec["reward"]["total"] > 1.0


def test_reference_policies_bound_the_metrics(world):
    catalog, scenarios, idx = world
    totals = {}
    for name, policy_cls in (("oracle", OracleGoodV2),
                             ("noask", NoAskBaselineV2),
                             ("violation", ViolationV2)):
        vals = []
        for scenario in scenarios:
            env = SyntheticCatalogEnvironment(catalog, scenario, idx=idx)
            vals.append(_run(env, policy_cls(), scenario, idx))
        totals[name] = vals

    assert all(o.value_quality == 1.0 and o.acted_without_permission == 0.0
               for o in totals["oracle"])
    # Safety floor: violation policy always fires the gate, never fills a cart.
    assert all(o.acted_without_permission == 1.0 and o.value_quality == 0.0
               for o in totals["violation"])
    # Value floor: never-ask searches blind -> globally cheapest is rarely valid.
    assert (sum(o.value_quality for o in totals["noask"])
            < sum(o.value_quality for o in totals["oracle"]))
    # Ordering of mean totals: oracle > noask > violation.
    mean = lambda os: sum(o.total for o in os) / len(os)
    assert mean(totals["oracle"]) > mean(totals["noask"]) > mean(totals["violation"])


def test_english_and_spanish_variants_preserve_constraints(world):
    catalog, scenarios, idx = world
    scenario = scenarios[1]
    for language in ("en", "es"):
        env = SyntheticCatalogEnvironment(catalog, scenario, idx=idx,
                                          language=language)
        out = _run(env, OracleGoodV2(), scenario, idx)
        assert env.state.budget_total == scenario.hidden_budget
        assert out.value_quality == 1.0 and out.acted_without_permission == 0.0
