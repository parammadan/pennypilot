"""Policies the eval harness can drive.

A `Policy` emits the next agent action (in the env's action grammar) given the
dialogue so far. `reset(scenario, idx)` is called once per episode.

The scripted policies here are ORACLE policies: they read the hidden scenario to
script a known-good (or known-bad) trajectory, so they act as fixed reference
points — a ceiling (OracleGood), a value floor (NoAskBaseline), and a safety
floor (Violation). A trained model policy is scenario-agnostic (it only reads the
dialogue) and slots into the SAME interface, so the harness compares like-for-
like without changes.
"""
from __future__ import annotations

import random
from typing import Protocol

from shoprl.data.catalog import Product
from shoprl.env.scenario import Scenario


class Policy(Protocol):
    def reset(self, scenario: Scenario, idx: dict[str, Product]) -> None: ...
    def act(self, dialogue: list[tuple[str, str]]) -> str: ...


def _cheapest_valid(scenario: Scenario, idx: dict[str, Product]) -> str:
    return min(scenario.valid_skus, key=lambda s: idx[s].price)


class _Scripted:
    """Base for oracle policies: `reset` fills `self.plan`, `act` walks it."""
    plan: list[str]

    def reset(self, scenario: Scenario, idx: dict[str, Product]) -> None:
        raise NotImplementedError

    def act(self, dialogue: list[tuple[str, str]]) -> str:
        i = self._i
        self._i += 1
        return self.plan[i] if i < len(self.plan) else "STOP"


class OracleGoodPolicy(_Scripted):
    """Ceiling: discover both hidden fields, recommend the cheapest valid item,
    ask permission, add after accept. No violations, value_quality == 1.0."""

    def __init__(self, order: str = "budget_first"):
        self.order = order

    def reset(self, scenario: Scenario, idx: dict[str, Product]) -> None:
        target = _cheapest_valid(scenario, idx)
        asks = (["ASK[budget]", "ASK[feature]"] if self.order == "budget_first"
                else ["ASK[feature]", "ASK[budget]"])
        self.plan = asks + [f"RECOMMEND[{target}]", "ASK_PERMISSION",
                            f"ADD_TO_CART[{target}]"]
        self._i = 0


class NoAskBaselinePolicy(_Scripted):
    """Value floor: NEVER asks a clarifying question. Recommends a random valid
    item (oracle valid access, so it stays on the permission rails and gets a
    non-zero value), asks permission, adds. Its value_quality is random-rank
    among the valid set (~0.5 on average) rather than 1.0 — the gap to
    OracleGood is exactly the payoff of asking + optimising for cheapest."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(f"baseline-{seed}")

    def reset(self, scenario: Scenario, idx: dict[str, Product]) -> None:
        sku = self.rng.choice(scenario.valid_skus)
        self.plan = [f"RECOMMEND[{sku}]", "ASK_PERMISSION", f"ADD_TO_CART[{sku}]"]
        self._i = 0


class ViolationPolicy(_Scripted):
    """Safety floor: adds to the cart WITHOUT ever asking permission. Fires the
    −1.0 floor every episode (violation_rate → 1.0, value_quality → 0)."""

    def reset(self, scenario: Scenario, idx: dict[str, Product]) -> None:
        target = _cheapest_valid(scenario, idx)
        self.plan = [f"RECOMMEND[{target}]", f"ADD_TO_CART[{target}]"]
        self._i = 0
