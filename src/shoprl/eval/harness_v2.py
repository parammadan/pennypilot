"""v2 eval harness: any JSON-action policy over held-out scenarios × languages.

Same philosophy as eval/harness.py (v1): held-out seeds, scripted reference
policies bounding a trained run, one `evaluate` per (policy, split). Adds the
v2 metric set the spec requires — task success, permission-violation rate
(the safety headline, must be 0), invalid-action rate, per-language breakdown.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable

from shoprl.data.catalog import Product, catalog_index
from shoprl.env.catalog_env import SyntheticCatalogEnvironment
from shoprl.env.scenario import Scenario, generate_hard_scenarios

HELDOUT_SEED_V2 = 5000   # disjoint from SFT (0), v1 eval (1000), RL (2000)


def heldout_hard_scenarios(catalog: list[Product], n: int = 128,
                           seed: int = HELDOUT_SEED_V2) -> list[Scenario]:
    """The pre-registered eval split (HYPOTHESES.md: n=128)."""
    return generate_hard_scenarios(catalog, n=n, seed=seed)


@dataclass
class EpisodeV2:
    scenario_id: str
    language: str
    success: bool            # legit cart item satisfying every hidden constraint
    value_quality: float
    violation: bool
    asked: int               # constraint reveals earned
    invalid_actions: int
    turns: int
    total: float


@dataclass
class EvalReportV2:
    policy: str
    language: str
    n: int
    task_success_rate: float
    permission_violation_rate: float   # the safety metric — must be 0
    mean_value_quality: float
    ask_rate: float                    # episodes with >=1 reveal
    invalid_action_rate: float         # invalid actions / total actions
    mean_turns: float
    mean_total: float

    def as_row(self) -> str:
        return (f"{self.policy:<14} lang={self.language:<5} n={self.n:<4} "
                f"success={self.task_success_rate:.2f} "
                f"viol={self.permission_violation_rate:.2f} "
                f"value={self.mean_value_quality:.3f} ask={self.ask_rate:.2f} "
                f"invalid={self.invalid_action_rate:.3f} "
                f"turns={self.mean_turns:.1f} total={self.mean_total:+.3f}")


def run_episode_v2(catalog: list[Product], scenario: Scenario,
                   idx: dict[str, Product], policy, language: str = "en",
                   max_turns: int = 12, max_steps: int = 15) -> EpisodeV2:
    env = SyntheticCatalogEnvironment(catalog, scenario, idx=idx,
                                      max_turns=max_turns, language=language)
    env.reset()
    policy.reset(scenario, idx)
    done = False
    steps = 0
    while not done and steps < max_steps:
        done = env.execute_text(policy.act()).done
        steps += 1
    out = env.calculate_outcome()
    return EpisodeV2(
        scenario_id=scenario.scenario_id,
        language=language,
        success=out.value_quality > 0 and not out.acted_without_permission,
        value_quality=out.value_quality,
        violation=bool(out.acted_without_permission),
        asked=sum("revealed" in t.note for t in env.turns),
        invalid_actions=sum(not t.valid for t in env.turns),
        turns=len(env.turns),
        total=out.total,
    )


def evaluate_v2(catalog: list[Product], scenarios: list[Scenario],
                policy_factory: Callable[[], object], name: str,
                language: str = "en", max_turns: int = 12) -> EvalReportV2:
    idx = catalog_index(catalog)
    eps = [run_episode_v2(catalog, s, idx, policy_factory(), language, max_turns)
           for s in scenarios]
    mean = statistics.mean
    total_actions = sum(e.turns for e in eps)
    return EvalReportV2(
        policy=name,
        language=language,
        n=len(eps),
        task_success_rate=mean(e.success for e in eps),
        permission_violation_rate=mean(e.violation for e in eps),
        mean_value_quality=mean(e.value_quality for e in eps),
        ask_rate=mean(e.asked > 0 for e in eps),
        invalid_action_rate=(sum(e.invalid_actions for e in eps) / total_actions
                             if total_actions else 0.0),
        mean_turns=mean(e.turns for e in eps),
        mean_total=mean(e.total for e in eps),
    )
