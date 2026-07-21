"""Dialogue state: corrections update state AND invalidate stale plans —
the spec's canonical correction sentences, verbatim."""
from shoprl.state import DialogueState


def _opened_state() -> DialogueState:
    s = DialogueState()
    s.observe_user_message(
        "Voy con dos niños, budget is $100, and necesito sunscreen SPF 50.")
    return s


def test_spanglish_opener_populates_state():
    s = _opened_state()
    assert s.budget_total == 100.0 and s.currency == "USD"
    assert s.number_of_children == 2
    assert s.requested_categories == ["sunscreen"]
    assert s.hard_constraints == {"spf_minimum": 50.0}
    assert s.code_switched and s.turn_number == 1
    assert "2 children" in s.normalized_english_intent


def test_budget_correction_invalidates_plan():
    s = _opened_state()
    s.record_candidates(["SPF-1", "SPF-2"])
    s.select_product("SPF-1", estimated_total=95.0)
    s.request_permission(["SPF-1"])
    s.observe_user_message("Actually, my budget is $90.")
    assert s.budget_total == 90.0
    assert s.selected_products == [] and s.estimated_total is None
    assert s.plan_stale
    assert s.permission_status == "not_requested"  # stale plan => stale approval


def test_remove_item_correction():
    s = _opened_state()
    s.select_product("waterproof pouch", estimated_total=25.0)
    s.observe_user_message("Remove the waterproof pouch.")
    assert s.selected_products == [] and s.plan_stale


def test_owned_item_excluded():
    s = DialogueState()
    s.observe_user_message("Planning a beach day: sunscreen, towels, una sombrilla.")
    assert set(s.requested_categories) == {"sunscreen", "towel", "umbrella"}
    s.observe_user_message("I already have sunscreen.")
    assert "sunscreen" in s.items_user_already_owns
    assert "sunscreen" not in s.requested_categories


def test_forbidden_spanglish_correction():
    s = DialogueState()
    s.observe_user_message("Necesito una sombrilla y toallas para la playa.")
    assert "umbrella" in s.requested_categories
    s.observe_user_message("No quiero una umbrella.")
    assert "umbrella" in s.forbidden_items
    assert "umbrella" not in s.requested_categories


def test_children_count_correction_invalidates():
    s = _opened_state()
    s.select_product("SPF-1", estimated_total=40.0)
    s.observe_user_message("Make that three children, not two.")
    assert s.number_of_children == 3 and s.plan_stale


def test_permission_hold_blocks_add():
    s = _opened_state()
    s.observe_user_message("Do not add anything yet.")
    assert s.permission_status == "hold"
    s.request_permission(["SPF-1"])            # hold is not overridden by asking
    assert s.permission_status == "hold"
    s.resolve_permission(True)
    assert s.permission_status == "hold"
    assert not s.add_to_cart("SPF-1")


def test_add_requires_explicit_grant_for_that_item():
    s = _opened_state()
    assert not s.add_to_cart("SPF-1")          # never asked
    s.request_permission(["SPF-1"])
    assert not s.add_to_cart("SPF-1")          # asked, not yet granted
    s.resolve_permission(True)
    assert not s.add_to_cart("SPF-9")          # granted, but for a different item
    assert s.add_to_cart("SPF-1")
    assert s.cart_contents == ["SPF-1"]


def test_budget_remaining():
    s = _opened_state()
    s.select_product("SPF-1", estimated_total=74.5)
    assert s.budget_remaining == 25.5
