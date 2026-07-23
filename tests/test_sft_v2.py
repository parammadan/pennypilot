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


def test_denied_recovery_varies_pre_denial_depth():
    """Regression for violation es_H-0014: denials must also arrive LATE
    (after >1 discovered constraint), and every recovery re-requests
    permission for the new item before adding."""
    catalog = generate_catalog(n=300, seed=0)
    demos = generate_sft_v2_dialogues(catalog, n=200, seed=1, denied_frac=0.5)
    denied = [d for d in demos if d.kind == "denied_recovery"]
    assert len(denied) >= 20
    # After every denial, a request_cart_permission precedes the add.
    late = 0
    for d in denied:
        agent_texts = [t.text for t in d.turns if t.role == "agent"]
        add_i = next(i for i, t in enumerate(agent_texts) if '"add_to_cart"' in t)
        assert '"request_cart_permission"' in agent_texts[add_i - 1]
        denial_i = next(i for i, t in enumerate(agent_texts)
                        if '"request_cart_permission"' in t)
        asks_before_denial = sum('"ask_user"' in t for t in agent_texts[:denial_i])
        if asks_before_denial > 2:      # budget + >=2 features discovered pre-denial
            late += 1
    assert late > 0, "no late-denial demos generated"


def test_v3_chat_prefix_present_and_safe(fixture):
    """v3 chat+actions: agent turns carry a natural-language prefix before the
    JSON, the FIRST turn greets, and the parser still extracts the structured
    action (safety gate reads JSON, not prose)."""
    from shoprl.actions import parse_agent_action
    _, _, demos, _ = fixture
    greeted = 0
    for d in demos:
        agent_turns = [t for t in d.turns if t.role == "agent"]
        first = agent_turns[0].text
        # first agent turn = greeting prose + a JSON action
        assert parse_agent_action(first).ok
        if any(g in first for g in ("Hi", "Hello", "Hey")):
            greeted += 1
        # every agent turn: prose present (not bare JSON) yet parses to an action
        for t in agent_turns:
            r = parse_agent_action(t.text)
            assert r.ok, t.text
            assert not t.text.strip().startswith("{"), "expected NL prefix"
    assert greeted >= len(demos) * 0.8, "most demos should open with a greeting"


def test_cannot_fulfill_demos_explain_and_redirect():
    from shoprl.data.catalog import generate_catalog
    from shoprl.data.sft_v2 import generate_sft_v2_dialogues
    demos = generate_sft_v2_dialogues(generate_catalog(n=300, seed=0),
                                      n=150, seed=11)
    cf = [d for d in demos if d.kind == "cannot_fulfill"]
    assert cf, "family must be generated"
    for d in cf:
        # turn 1 is the odd request, turn 2 the demonstrated explain+redirect
        odd, redirect = d.turns[1], d.turns[2]
        assert odd.role == "user" and odd.injected
        assert redirect.role == "agent"
        assert '"action": "ask_user"' in redirect.text
        low = redirect.text.lower()
        assert ("laptops-only" in low or "laptops only" in low
                or "only stock laptops" in low
                or "maximum" in low or "cheapest" in low)
        # a detected price floor carries the store notice the policy sees live
        if "minimum" in odd.text.lower() or "mínimo" in odd.text.lower():
            assert "[store notice:" in odd.text
        # and the episode still ends as a legit cheapest-valid cart
        assert d.turns[-2].role == "agent" and "add_to_cart" in d.turns[-2].text
