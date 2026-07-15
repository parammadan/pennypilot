"""Phase-1 tests for the eval harness.

Runs the harness end-to-end on the three scripted reference policies over a
held-out scenario split and checks the metrics come out as expected:
  - OracleGood:  ~0 violations, value_quality ~1.0 (ceiling);
  - NoAskBaseline: never asks, lower value_quality (floor), no violations;
  - Violation:   ~1.0 violation rate, ~0 value_quality (safety floor).
"""
from __future__ import annotations

from shoprl.data.catalog import generate_catalog
from shoprl.eval.harness import evaluate, heldout_scenarios
from shoprl.eval.policies import (NoAskBaselinePolicy, OracleGoodPolicy,
                                  ViolationPolicy)

CAT = generate_catalog(n=500, seed=0)
SCEN = heldout_scenarios(CAT, n=120, seed=1000)


def test_heldout_split_is_an_iid_holdout_mostly_unseen():
    """Held-out uses a disjoint SEED (independent sampler state), same
    distribution — an i.i.d. test set. Because budgets are pinned to discrete
    catalog prices, a finite hidden-need space means a few tasks coincide by
    chance; the vast majority of hidden needs are unseen, so aggregate metrics
    measure generalization, not train-set recall."""
    from shoprl.env.scenario import generate_scenarios
    train = generate_scenarios(CAT, n=120, seed=0)
    train_needs = {(s.hidden_budget, s.must_have_key, s.must_have_value) for s in train}
    held_needs = {(s.hidden_budget, s.must_have_key, s.must_have_value) for s in SCEN}
    overlap = len(train_needs & held_needs) / len(held_needs)
    assert overlap < 0.20  # measured ~0.14: overwhelmingly unseen needs


def test_oracle_good_is_the_ceiling():
    rep = evaluate(CAT, SCEN, OracleGoodPolicy, "oracle_good")
    assert rep.n == len(SCEN)
    assert rep.violation_rate == 0.0
    assert rep.ask_rate == 1.0
    assert rep.mean_value_quality > 0.99      # always the cheapest valid
    assert rep.accept_rate == 1.0
    assert rep.mean_turns_to_rec == 3.0       # ask, ask, recommend


def test_baseline_is_a_value_floor_below_good():
    good = evaluate(CAT, SCEN, OracleGoodPolicy, "oracle_good")
    base = evaluate(CAT, SCEN, NoAskBaselinePolicy, "no_ask_baseline")
    assert base.ask_rate == 0.0
    assert base.violation_rate == 0.0
    assert base.mean_turns_to_rec == 1.0                 # recommends immediately
    assert base.mean_value_quality < good.mean_value_quality - 0.2
    assert base.mean_value_quality > 0.0                 # random valid, not zero


def test_violation_policy_fires_the_safety_floor():
    rep = evaluate(CAT, SCEN, ViolationPolicy, "violation")
    assert rep.violation_rate > 0.99
    assert rep.mean_value_quality == 0.0
    assert rep.accept_rate == 0.0
    assert rep.mean_total < -0.9


def test_report_row_renders():
    rep = evaluate(CAT, SCEN, OracleGoodPolicy, "oracle_good")
    row = rep.as_row()
    assert "oracle_good" in row and "value=" in row and "viol=" in row
