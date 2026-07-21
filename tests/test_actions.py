"""Abstract-action schema: parse, validate, reject — the policy/env contract."""
from shoprl.actions import (AddToCart, AskUser, ParseResult,
                            RequestCartPermission, Search, action_to_json,
                            parse_agent_action)


def test_each_action_kind_parses():
    cases = {
        '{"action": "ask_user", "question": "What is your total budget?"}': AskUser,
        '{"action": "search", "query": "laptop under 800"}': Search,
        '{"action": "inspect_product", "product_id": "LAP-0007"}': None,
        '{"action": "select_product", "product_id": "LAP-0007", "reason": "cheapest valid"}': None,
        '{"action": "request_cart_permission", "items": ["LAP-0007"], "estimated_total": 74.5}': RequestCartPermission,
        '{"action": "add_to_cart", "product_id": "LAP-0007"}': AddToCart,
    }
    for text, cls in cases.items():
        r = parse_agent_action(text)
        assert r.ok, f"{text} -> {r.error}"
        if cls is not None:
            assert isinstance(r.action, cls)


def test_prose_wrapped_json_is_extracted():
    r = parse_agent_action(
        'Sure! Let me look. {"action": "search", "query": "sunscreen SPF 50"} '
        "I'll check the results.")
    assert r.ok and r.action.query == "sunscreen SPF 50"


def test_first_action_object_wins():
    r = parse_agent_action(
        '{"action": "ask_user", "question": "budget?"} '
        '{"action": "add_to_cart", "product_id": "LAP-0001"}')
    assert isinstance(r.action, AskUser)


def test_unknown_action_name_is_error_with_raw():
    r = parse_agent_action('{"action": "purchase_now", "product_id": "LAP-1"}')
    assert not r.ok and r.error and r.raw == {"action": "purchase_now",
                                              "product_id": "LAP-1"}


def test_missing_required_field_is_error():
    r = parse_agent_action('{"action": "add_to_cart"}')
    assert not r.ok and "product_id" in r.error


def test_no_json_is_error():
    assert not parse_agent_action("I think we should buy the cheapest one.").ok
    assert not parse_agent_action("").ok


def test_permission_items_typed():
    r = parse_agent_action(
        '{"action": "request_cart_permission", "items": ["A", "B"], '
        '"estimated_total": 74.50}')
    assert r.action.items == ["A", "B"] and r.action.estimated_total == 74.5


def test_roundtrip_canonical_json():
    r = parse_agent_action('{"action": "ask_user", "question": "q?"}')
    r2 = parse_agent_action(action_to_json(r.action))
    assert r2.ok and r2.action == r.action


def test_parse_result_shape():
    assert isinstance(parse_agent_action("x"), ParseResult)
