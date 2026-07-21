"""v2 SFT demos: recorded env transcripts — grammar, replay-correctness,
kind/language variety. Mirrors test_sft.py's three pinned properties."""
import pytest

from shoprl.actions import parse_agent_action
from shoprl.data.catalog import catalog_index, generate_catalog
from shoprl.data.sft_v2 import demo_v2_stats, generate_sft_v2_dialogues
from shoprl.env.catalog_env import SyntheticCatalogEnvironment
from shoprl.env.scenario import generate_hard_scenarios

N = 120
SEED = 0


@pytest.fixture(scope="module")
def fixture():
    catalog = generate_catalog(n=300, seed=0)
    idx = catalog_index(catalog)
    demos = generate_sft_v2_dialogues(catalog, n=N, seed=SEED)
    scen_by_id = {s.scenario_id: s
                  for s in generate_hard_scenarios(catalog, n=N, seed=SEED)}
    return catalog, idx, demos, scen_by_id


def test_agent_turns_parse_and_reference_real_skus(fixture):
    catalog, idx, demos, _ = fixture
    for d in demos:
        for t in d.turns:
            if t.role != "agent":
                continue
            r = parse_agent_action(t.text)
            assert r.ok, f"{d.scenario_id}: unparseable agent turn {t.text!r}"
            pid = getattr(r.action, "product_id", None)
            if pid is not None:
                assert pid in idx


def test_demos_open_with_user_and_alternate(fixture):
    _, _, demos, _ = fixture
    for d in demos:
        assert d.turns[0].role == "user"
        roles = [t.role for t in d.turns]
        assert not any(roles[i] == roles[i + 1] == roles[i + 2]
                       for i in range(len(roles) - 2))


def _replay(demo, catalog, idx, scen):
    env = SyntheticCatalogEnvironment(catalog, scen, idx=idx, max_turns=32,
                                      language=demo.language)
    env.reset()
    for t in demo.turns:
        if t.role == "agent":
            env.execute_text(t.text)
        elif t.injected:
            env.state.observe_user_message(t.text)
    return env


def test_cart_demos_replay_to_cheapest_valid_no_violation(fixture):
    catalog, idx, demos, scen_by_id = fixture
    checked = 0
    for d in demos:
        if d.kind == "hold_edge":
            continue
        scen = scen_by_id[d.scenario_id]
        env = _replay(d, catalog, idx, scen)
        out = env.calculate_outcome()
        assert env.get_cart() == [d.target_sku], d.scenario_id
        assert out.value_quality == 1.0
        assert out.acted_without_permission == 0.0
        assert out.accepted == 1.0
        checked += 1
    assert checked > 0


def test_denied_recovery_contains_a_denial_then_recovers(fixture):
    catalog, idx, demos, scen_by_id = fixture
    denied = [d for d in demos if d.kind == "denied_recovery"]
    assert denied, "expected denied_recovery demos"
    for d in denied:
        notes_denied = any("denied" in t.text.lower() or "no me sirve" in t.text.lower()
                           or "doesn't work" in t.text.lower()
                           for t in d.turns if t.role == "user")
        assert notes_denied, f"{d.scenario_id}: no visible denial"
        env = _replay(d, catalog, idx, scen_by_id[d.scenario_id])
        assert env.get_cart() == [d.target_sku]     # recovered to a legit add


def test_hold_edge_never_carts(fixture):
    catalog, idx, demos, scen_by_id = fixture
    holds = [d for d in demos if d.kind == "hold_edge"]
    assert holds, "expected hold_edge demos"
    for d in holds:
        assert not any('"add_to_cart"' in t.text for t in d.turns
                       if t.role == "agent")
        env = _replay(d, catalog, idx, scen_by_id[d.scenario_id])
        assert env.get_cart() == []
        assert env.state.permission_status == "hold"
        assert env.calculate_outcome().acted_without_permission == 0.0


def test_variety(fixture):
    _, _, demos, _ = fixture
    s = demo_v2_stats(demos)
    assert set(s["by_kind"]) >= {"positive", "denied_recovery", "hold_edge"}
    assert set(s["by_language"]) == {"en", "es", "es-en"}
    assert len(s["by_n_asks"]) >= 2             # 2-constraint vs 3-constraint
    assert s["turn_len_max"] > s["turn_len_min"]
