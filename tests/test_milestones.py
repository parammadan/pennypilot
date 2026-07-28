"""Milestone funnel, friction metrics, attribution — on fixture sessions
with known semantic-event traces."""
import pytest

from shoprl.platform.attribution import attribute_session
from shoprl.platform.milestones import friction_metrics, milestone_funnel
from shoprl.platform.store import PlatformStore

GOAL = {"goal_text": "cheapest laptop", "budget_max": 900,
        "must_have_constraints": {"min_ram": 16, "max_weight": 1.6},
        "expertise": "NOVICE", "goal_source": "SIMULATED_GROUND_TRUTH"}


def _sem(store, sid, typ, ti, eid, **attrs):
    store.ingest({"kind": "semantic", "session_id": sid, "type": typ,
                  "turn_index": ti, "attributes": attrs, "event_id": eid})


def _good_session(store, sid, mv="modelA"):
    store.ingest({"kind": "episode_start", "session_id": sid, "label": mv,
                  "model_version": mv, "goal": GOAL,
                  "scenario_family": "GRADUAL_CONSTRAINT_REVEAL"})
    for ti, key in ((1, "budget"), (2, "min_ram"), (3, "max_weight")):
        _sem(store, sid, "constraint_revealed", ti, f"{sid}-rev-{key}", key=key)
    _sem(store, sid, "search_executed", 4, f"{sid}-search",
         premature_search=False, missing_required_constraints=[],
         known_constraints=["budget", "min_ram", "max_weight"], result_count=5)
    _sem(store, sid, "recommendation_shown", 5, f"{sid}-rec",
         product_id="LAP-1", satisfies_known_constraints=True,
         satisfies_full_ground_truth=True, grounded_in_catalogue=True)
    _sem(store, sid, "permission_requested", 6, f"{sid}-pr", product_id="LAP-1")
    _sem(store, sid, "permission_granted", 6, f"{sid}-pg", product_id="LAP-1")
    store.ingest({"kind": "episode_end", "session_id": sid, "cart": ["LAP-1"],
                  "outcome": {"task_satisfied": True,
                              "constraints_satisfied": True,
                              "cheapest_valid_product_selected": True,
                              "permission_obtained": True,
                              "correct_cart_action": True,
                              "safety_violation": False,
                              "goal_satisfaction": "SATISFIED"}})


def _premature_session(store, sid, mv="modelB"):
    store.ingest({"kind": "episode_start", "session_id": sid, "label": mv,
                  "model_version": mv, "goal": GOAL,
                  "scenario_family": "GRADUAL_CONSTRAINT_REVEAL"})
    _sem(store, sid, "constraint_revealed", 1, f"{sid}-rev-b", key="budget")
    _sem(store, sid, "search_executed", 2, f"{sid}-search1",
         premature_search=True,
         missing_required_constraints=["min_ram", "max_weight"],
         known_constraints=["budget"], result_count=10)
    _sem(store, sid, "recommendation_shown", 3, f"{sid}-rec1",
         product_id="LAP-9", satisfies_known_constraints=True,
         satisfies_full_ground_truth=False, grounded_in_catalogue=True)
    _sem(store, sid, "customer_correction", 4, f"{sid}-corr",
         reason="AGENT_IGNORED_WEIGHT_CONSTRAINT", violated_key="max_weight")
    _sem(store, sid, "conversation_abandoned", 6, f"{sid}-aband",
         reason="excessive_friction")
    store.ingest({"kind": "episode_end", "session_id": sid, "cart": [],
                  "outcome": {"task_satisfied": False,
                              "constraints_satisfied": False,
                              "cheapest_valid_product_selected": False,
                              "permission_obtained": False,
                              "correct_cart_action": False,
                              "safety_violation": False,
                              "goal_satisfaction": "UNSATISFIED"}})


@pytest.fixture
def store(tmp_path):
    st = PlatformStore(tmp_path)
    for k in range(3):
        _good_session(st, f"good-{k}")
    for k in range(2):
        _premature_session(st, f"bad-{k}")
    # a no-goal legacy session: must be excluded, counted
    st.ingest({"kind": "episode_start", "session_id": "legacy", "label": "x"})
    return st


def test_milestone_funnel_stages_and_reasons(store):
    f = milestone_funnel(store)
    assert f["excluded_no_goal"] == 1
    ov = {s["stage"]: s for s in f["overall"]["stages"]}
    assert ov["session_started"]["reached"] == 5
    assert ov["goal_understood"]["reached"] == 3      # bad sessions never
    assert ov["valid_search"]["reached"] == 3
    assert ov["task_satisfied"]["reached"] == 3
    assert ov["valid_search"]["conversion_from_prev"] == 1.0
    reasons = f["overall"]["failure_reasons"]["goal_understood"]
    assert any("missing=" in r for r in reasons)
    bym = f["by_model_version"]
    assert bym["modelA"]["stages"][-1]["reached"] == 3
    assert bym["modelB"]["stages"][-1]["reached"] == 0


def test_friction_metrics_provenance(store):
    m = friction_metrics(store)
    p = m["premature_search_rate"]
    assert p["value"] == 0.4 and "(2)" in p["numerator"] and "(5)" in p["denominator"]
    assert p["by_model_version"]["modelB"]["value"] == 1.0
    assert p["by_model_version"]["modelA"]["value"] == 0.0
    c = m["customer_correction_rate"]
    assert c["value"] == 0.4
    t = m["turns_to_goal_understood"]
    assert t["value"] == 3
    assert t["by_model_version"]["modelA"]["value"] == 3   # a MEDIAN TURN, not a ratio                # last required constraint at turn 3
    assert "never reached" in t["exclusions"]
    assert p["low_support"] is True       # 5 < MIN_SUPPORT, flagged not hidden


def test_attribution_fires_and_abstains(store):
    a = attribute_session(store, "bad-0")
    assert a["primary_category"] == "CONSTRAINT_EXTRACTION"
    assert a["confidence"] == 0.9
    assert "bad-0-search1" in a["evidence_event_ids"]
    assert "bad-0-corr" in a["evidence_event_ids"]
    assert a["evidence"]["constraints_never_revealed"] == ["max_weight", "min_ram"]
    good = attribute_session(store, "good-0")
    assert good["primary_category"] == "UNKNOWN" and good["confidence"] == 0.0
    legacy = attribute_session(store, "legacy")
    assert legacy["primary_category"] == "UNKNOWN"
    assert "ineligible" in legacy["reason"]


def test_attribution_v2_excessive_friction(tmp_path):
    store = PlatformStore(tmp_path)
    sid = "fric-0"
    store.ingest({"kind": "episode_start", "session_id": sid, "label": "m",
                  "model_version": "m", "goal": GOAL})
    for ti, key in ((1, "budget"), (2, "min_ram"), (3, "max_weight")):
        _sem(store, sid, "constraint_revealed", ti, f"{sid}-r{key}", key=key)
    for k in range(2):    # agent repeats known questions
        _sem(store, sid, "constraint_requested", 4 + k, f"{sid}-red{k}",
             field="budget", redundant=True)
    _sem(store, sid, "conversation_abandoned", 7, f"{sid}-ab",
         reason="EXCESSIVE_FRICTION")
    store.ingest({"kind": "episode_end", "session_id": sid, "cart": [],
                  "outcome": {"task_satisfied": False,
                              "constraints_satisfied": False,
                              "cheapest_valid_product_selected": False,
                              "permission_obtained": False,
                              "correct_cart_action": False,
                              "safety_violation": False,
                              "goal_satisfaction": "UNSATISFIED"}})
    a = attribute_session(store, sid)
    assert a["primary_category"] == "EXCESSIVE_FRICTION"
    assert a["taxonomy_version"] == "v2" and a["confidence"] == 0.85
    assert f"{sid}-red0" in a["evidence_event_ids"]
    assert f"{sid}-ab" in a["evidence_event_ids"]


def test_attribution_precedence_constraint_extraction_first(store):
    # bad-0 has premature+correction+missed -> still CONSTRAINT_EXTRACTION
    a = attribute_session(store, "bad-0")
    assert a["primary_category"] == "CONSTRAINT_EXTRACTION"
