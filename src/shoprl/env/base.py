"""ShoppingEnvironment — the adapter interface every environment implements.

The policy emits abstract actions (shoprl.actions); adapters translate them
into catalog calls (SyntheticCatalogEnvironment), WebShop actions
(WebShopEnvironment, Stage 4), or Playwright operations (BrowserDemoEnvironment,
Stage 5 — demo/replay only, never the training path). Selected by
`environment.type` in config so old and new environments coexist.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class StepResult:
    """One env transition: what the agent sees next, whether the episode is
    over, and the bookkeeping the transcript/reward layers need."""
    observation: str
    done: bool
    info_gain: float = 0.0
    note: str = ""
    action_valid: bool = True


@runtime_checkable
class ShoppingEnvironment(Protocol):
    def reset(self, scenario) -> str: ...
    def observe(self) -> str: ...
    def execute(self, action) -> StepResult: ...
    def get_candidates(self) -> list: ...
    def get_cart(self) -> list[str]: ...
    def calculate_outcome(self): ...
