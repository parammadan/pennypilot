"""Phase-1 tests for the hidden-need scenario generator.

These pin the two properties the whole task depends on:
  1. the need is genuinely HIDDEN (never leaked in the opening utterance), and
  2. the valid set is non-empty and is exactly the catalog items meeting BOTH
     the hidden budget and the must-have (verifiable ground truth).
"""
from __future__ import annotations

from shoprl.data.catalog import generate_catalog
from shoprl.env.scenario import generate_scenarios, scenario_valid_skus


def test_scenarios_are_deterministic():
    cat = generate_catalog(n=120, seed=0)
    a = generate_scenarios(cat, n=25, seed=0)
    b = generate_scenarios(cat, n=25, seed=0)
    assert [s.model_dump() for s in a] == [s.model_dump() for s in b]


def test_need_is_hidden_from_opening_utterance():
    cat = generate_catalog(n=120, seed=0)
    for s in generate_scenarios(cat, n=60, seed=1):
        text = s.opening_utterance.lower()
        # Budget must not appear (as a number) in the opener.
        assert str(int(s.hidden_budget)) not in text
        assert "$" not in text and "budget" not in text
        # The must-have must not be volunteered either.
        assert str(s.must_have_value).lower() not in text
        for word in ("ram", "gb", "battery", "hour", "weight", "lb", "brand"):
            assert word not in text


def test_valid_set_nonempty_and_correct():
    cat = generate_catalog(n=150, seed=0)
    idx = {p.sku: p for p in cat}
    for s in generate_scenarios(cat, n=60, seed=2):
        assert s.valid_skus, "every scenario must have >=1 satisfiable answer"
        recomputed = scenario_valid_skus(
            cat, s.hidden_budget, s.must_have_key, s.must_have_value
        )
        assert set(s.valid_skus) == set(recomputed)
        # Spot-check the invariant directly against the catalog.
        for sku in s.valid_skus:
            p = idx[sku]
            assert p.price <= s.hidden_budget
            if s.must_have_key == "brand":
                assert p.brand == s.must_have_value
            elif s.must_have_key == "min_ram":
                assert p.ram_gb >= s.must_have_value
            elif s.must_have_key == "max_weight":
                assert p.weight_lbs <= s.must_have_value
            elif s.must_have_key == "min_battery":
                assert p.battery_hrs >= s.must_have_value


def test_budget_is_binding_not_trivial():
    """The hidden budget should exclude a meaningful share of must-have-matching
    items — otherwise asking about budget would be pointless."""
    cat = generate_catalog(n=200, seed=0)
    binding = 0
    scen = generate_scenarios(cat, n=60, seed=3)
    for s in scen:
        if s.must_have_key == "brand":
            pool = [p for p in cat if p.brand == s.must_have_value]
        else:
            from shoprl.data.prompts import satisfies
            pool = [p for p in cat
                    if satisfies(p, {s.must_have_key: float(s.must_have_value)})]
        over = [p for p in pool if p.price > s.hidden_budget]
        if over:
            binding += 1
    # Most scenarios should have at least one over-budget alternative.
    assert binding >= 0.7 * len(scen)
