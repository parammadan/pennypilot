"""Environment factory — the consumer of `environment.type` (audit item 2).

The config knob existed without a consumer; this closes that gap. One entry
point maps the three declared types to their constructors, with the honest
guard: `browser_demo` is NEVER a training/eval environment and refuses to be
constructed as one (use scripts/demo_browser.py — the browser is a projection
of decisions, not a decision surface).
"""
from __future__ import annotations

from shoprl.config import Config, EnvironmentConfig
from shoprl.data.catalog import Product, catalog_index
from shoprl.env.catalog_env import SyntheticCatalogEnvironment
from shoprl.env.scenario import Scenario
from shoprl.env.webshop_env import WebShopBackend, WebShopEnvironment


def make_environment(config: Config | EnvironmentConfig, *,
                     catalog: list[Product] | None = None,
                     scenario: Scenario | None = None,
                     instruction: str = "",
                     backend: WebShopBackend | None = None):
    """Build the environment `environment.type` selects.

    synthetic_catalog -> needs catalog + scenario (the training path)
    webshop           -> needs instruction (+ optional real backend)
    browser_demo      -> refuses: demo/replay only, never train/eval
    """
    env_cfg = config.environment if isinstance(config, Config) else config
    kind = env_cfg.type

    if kind == "synthetic_catalog":
        if catalog is None or scenario is None:
            raise ValueError("synthetic_catalog requires catalog= and scenario=")
        return SyntheticCatalogEnvironment(
            catalog, scenario, idx=catalog_index(catalog),
            max_turns=env_cfg.max_turns, language=env_cfg.language)
    if kind == "webshop":
        return WebShopEnvironment(backend=backend, instruction=instruction,
                                  max_turns=env_cfg.max_turns)
    if kind == "browser_demo":
        raise ValueError(
            "browser_demo is a projection, never a training/eval environment "
            "— run it via scripts/demo_browser.py (spec: the browser renders "
            "decisions already made)")
    raise ValueError(f"unknown environment.type: {kind!r}")
