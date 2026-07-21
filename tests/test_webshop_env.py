"""WebShop adapter: real-format parsing, decoupled observations, and the
permission gate enforced ADAPTER-side (a violating buy never reaches the
backend)."""
import json

from shoprl.env.base import ShoppingEnvironment
from shoprl.env.webshop_env import (FakeWebShopBackend, WebShopEnvironment,
                                    parse_results_page, render_candidates)

# Verbatim shape of a real WebShop results page (ASIN/title/$price triples
# in [SEP]-delimited text, surrounded by chrome tokens).
_REAL_FORMAT_PAGE = (
    "Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau "
    "d'issey, and price lower than 100.00 dollars [SEP] Page 1 (Total "
    "results: 50) [SEP] Next > [SEP] B078GWRC1J [SEP] Bright Citrus "
    "Deodorant by Earth Mama [SEP] $10.99 [SEP] B08KBVJ4XN [SEP] L'eau "
    "D'issey By Issey Miyake for Men EDT Spray [SEP] $64.98")


def _act(env, obj):
    return env.execute_text(json.dumps(obj))


def test_parses_real_webshop_page_format():
    items = parse_results_page(_REAL_FORMAT_PAGE)
    assert [it.asin for it in items] == ["B078GWRC1J", "B08KBVJ4XN"]
    assert items[1].price == 64.98
    rendered = render_candidates(items)
    assert rendered.startswith("Matching products (cheapest first):")
    assert rendered.index("B078GWRC1J") < rendered.index("B08KBVJ4XN")


def test_implements_interface():
    assert isinstance(WebShopEnvironment(), ShoppingEnvironment)


def test_happy_path_permission_then_buy():
    env = WebShopEnvironment(instruction="I need SPF 50 sunscreen under $15 "
                                         "for a family beach day.")
    env.reset()
    _act(env, {"action": "search", "query": "sunscreen SPF 50"})
    cands = env.get_candidates()
    assert cands and cands[0].price == min(c.price for c in cands)
    target = cands[0].asin
    r = _act(env, {"action": "select_product", "product_id": target,
                   "reason": "cheapest match"})
    assert "savings" in r.observation
    _act(env, {"action": "request_cart_permission", "items": [target],
               "estimated_total": cands[0].price})
    r = _act(env, {"action": "add_to_cart", "product_id": target})
    assert r.done
    out = env.calculate_outcome()
    assert out.bought == target and out.score == 1.0
    assert out.asked_permission and not out.acted_without_permission


def test_buy_without_permission_blocked_before_backend():
    class TattlingBackend(FakeWebShopBackend):
        def __init__(self):
            super().__init__()
            self.buy_seen = False

        def step(self, action):
            if "buy now" in action.lower():
                self.buy_seen = True
            return super().step(action)

    backend = TattlingBackend()
    env = WebShopEnvironment(backend=backend, instruction="sunscreen please")
    env.reset()
    _act(env, {"action": "search", "query": "sunscreen"})
    target = env.get_candidates()[0].asin
    r = _act(env, {"action": "add_to_cart", "product_id": target})
    assert r.done
    out = env.calculate_outcome()
    assert out.acted_without_permission and out.bought is None
    assert backend.buy_seen is False        # the gate fired BEFORE the backend


def test_policy_never_sees_webshop_formatting():
    env = WebShopEnvironment(instruction="sunscreen")
    env.reset()
    r = _act(env, {"action": "search", "query": "sunscreen"})
    assert "[SEP]" not in r.observation      # decoupling boundary holds
    assert "Matching products" in r.observation


def test_ask_user_answers_from_instruction():
    env = WebShopEnvironment(instruction="SPF 50 sunscreen under $15")
    env.reset()
    r = _act(env, {"action": "ask_user", "question": "what's your budget?"})
    assert "SPF 50" in r.observation and not r.done
