"""Recipe layer: approval gate, outcome-validated selection, dedup, lineage."""
import json

import pytest

from shoprl.platform.recipes import Recipe, apply_recipe, load_recipe
from shoprl.platform.store import PlatformStore

GOAL = {"goal_text": "g", "budget_max": 900,
        "must_have_constraints": {"min_ram": 16},
        "goal_source": "SIMULATED_GROUND_TRUTH"}


def _ep(store, sid, mv, satisfied, scenario_idx):
    store.ingest({"kind": "episode_start", "session_id": sid, "label": mv,
                  "model_version": mv, "goal": GOAL})
    store.ingest({"kind": "turn", "session_id": sid, "i": 0,
                  "agent": '{"action": "ask_user", "question": "budget?"}',
                  "observation": "about $900"})
    store.ingest({"kind": "episode_end", "session_id": sid,
                  "cart": ["LAP-1"] if satisfied else [],
                  "outcome": {"task_satisfied": satisfied,
                              "constraints_satisfied": satisfied,
                              "cheapest_valid_product_selected": satisfied,
                              "permission_obtained": satisfied,
                              "correct_cart_action": satisfied,
                              "safety_violation": False,
                              "goal_satisfaction":
                                  "SATISFIED" if satisfied else "UNSATISFIED"}})


@pytest.fixture
def store(tmp_path):
    st = PlatformStore(tmp_path / "s")
    _ep(st, "camp-rl-100-0", "rl7b-B3", True, 0)
    _ep(st, "camp-rl-100-0b", "rl7b-B3", True, 0)   # same scenario idx 0 -> dedup
    _ep(st, "camp-rl-100-1", "rl7b-B3", False, 1)   # failed -> excluded
    _ep(st, "camp-sft-100-2", "sft7b-B3", False, 2) # targeting only
    return st


def _recipe(status="approved"):
    return Recipe(recipe_id="t-v1", target_failure="PREMATURE_SEARCH",
                  demonstration_source={"model_version": "rl7b-B3",
                                        "require": {"task_satisfied": True}},
                  targeting_source={"model_version": "sft7b-B3"},
                  approval_status=status, approved_by="param")


def test_draft_recipe_refuses(store, tmp_path):
    with pytest.raises(PermissionError):
        apply_recipe(store, _recipe(status="draft"), tmp_path / "out")


def test_apply_selects_validated_dedups_and_records_lineage(store, tmp_path):
    m = apply_recipe(store, _recipe(), tmp_path / "out")
    assert m["sequences"] == 1                    # 2 satisfied - 1 dup
    assert m["dropped_duplicates"] == 1
    assert m["dropped_outcome_invalid"] == 0      # failed one never matched
    assert m["quality_checks"]["all_outcome_validated"] is True
    assert m["targeting_sessions_informing_slices"] == 1
    rows = store.query("SELECT recipe_id, dataset_sha FROM lineage")["rows"]
    assert rows[0][0] == "t-v1" and len(rows[0][1]) == 16
    data = [json.loads(l) for l in
            (tmp_path / "out" / "t-v1.jsonl").read_text().splitlines()]
    assert data[0]["messages"][0]["role"] == "system"


def test_committed_recipe_file_is_valid_draft():
    r = load_recipe("recipes/premature-search-v1.json")
    assert r.approval_status == "draft"           # awaits the human
    assert r.labeling_policy == "outcome_validated_demonstration_v1"
