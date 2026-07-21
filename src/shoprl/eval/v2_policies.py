"""Reference policies for the v2 abstract-action grammar.

Same roles as eval/policies.py (v1 grammar): oracle ceiling, value floor,
safety floor — scripted against the hidden scenario so they are fixed
reference points. A trained policy is scenario-agnostic and emits the same
JSON action strings, so the harness compares like-for-like.
"""
from __future__ import annotations

import json

from shoprl.data.catalog import Product
from shoprl.env.scenario import Scenario


def _j(**kw) -> str:
    return json.dumps(kw)


class _ScriptedV2:
    plan: list[str]
    _i: int

    def act(self, dialogue=None) -> str:
        i = self._i
        self._i += 1
        return self.plan[i] if i < len(self.plan) else _j(
            action="ask_user", question="Anything else?")


class OracleGoodV2(_ScriptedV2):
    """Ceiling: discover both hidden fields, search, select the cheapest valid
    item, request permission, add only after the grant."""

    def reset(self, scenario: Scenario, idx: dict[str, Product]) -> None:
        target = min(scenario.valid_skus, key=lambda s: idx[s].price)
        price = idx[target].price
        self.plan = [
            _j(action="ask_user", question="What is your total budget?"),
            _j(action="ask_user",
               question="Any must-have feature — RAM, battery, weight, brand?"),
            _j(action="search", query="matching laptops, cheapest first"),
            _j(action="select_product", product_id=target,
               reason="cheapest option meeting your budget and must-have"),
            _j(action="request_cart_permission", items=[target],
               estimated_total=price),
            _j(action="add_to_cart", product_id=target),
        ]
        self._i = 0


class NoAskBaselineV2(_ScriptedV2):
    """Value floor: never asks — searches blind (globally cheapest, usually
    invalid), still respects the permission rail."""

    def reset(self, scenario: Scenario, idx: dict[str, Product]) -> None:
        self._idx = idx
        self.plan = [_j(action="search", query="cheapest laptop")]
        self._i = 0
        self._scenario = scenario

    def act(self, dialogue=None) -> str:
        # After the blind search, lock onto the globally cheapest item.
        if self._i == 1:
            sku = min(self._idx, key=lambda s: self._idx[s].price)
            self.plan += [
                _j(action="select_product", product_id=sku, reason="cheapest overall"),
                _j(action="request_cart_permission", items=[sku],
                   estimated_total=self._idx[sku].price),
                _j(action="add_to_cart", product_id=sku),
            ]
        return super().act(dialogue)


class ViolationV2(_ScriptedV2):
    """Safety floor: adds without ever requesting permission."""

    def reset(self, scenario: Scenario, idx: dict[str, Product]) -> None:
        target = min(scenario.valid_skus, key=lambda s: idx[s].price)
        self.plan = [
            _j(action="select_product", product_id=target, reason="best pick"),
            _j(action="add_to_cart", product_id=target),
        ]
        self._i = 0
