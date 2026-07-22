"""environment.type has a consumer now (audit item 2): the factory maps every
declared value, and browser_demo refuses with the spec's rationale."""
import pytest

from shoprl.config import Config, EnvironmentConfig
from shoprl.data.catalog import generate_catalog
from shoprl.env.catalog_env import SyntheticCatalogEnvironment
from shoprl.env.factory import make_environment
from shoprl.env.scenario import generate_hard_scenarios
from shoprl.env.webshop_env import WebShopEnvironment


def test_synthetic_catalog_from_full_config():
    catalog = generate_catalog(n=50, seed=0)
    scen = generate_hard_scenarios(catalog, n=1, seed=1)[0]
    cfg = Config.model_validate(
        {"environment": {"type": "synthetic_catalog", "language": "es-en",
                         "max_turns": 9}})
    env = make_environment(cfg, catalog=catalog, scenario=scen)
    assert isinstance(env, SyntheticCatalogEnvironment)
    assert env.max_turns == 9
    env.reset()
    assert env.state.code_switched          # language knob reached the env


def test_synthetic_requires_catalog_and_scenario():
    with pytest.raises(ValueError, match="requires catalog"):
        make_environment(EnvironmentConfig(type="synthetic_catalog"))


def test_webshop_type():
    env = make_environment(EnvironmentConfig(type="webshop", max_turns=7),
                           instruction="sunscreen under $15")
    assert isinstance(env, WebShopEnvironment)
    assert env.max_turns == 7
    assert env.reset() == "sunscreen under $15"


def test_browser_demo_refuses_with_rationale():
    with pytest.raises(ValueError, match="projection"):
        make_environment(EnvironmentConfig(type="browser_demo"))
