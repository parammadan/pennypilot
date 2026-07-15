"""Eval harness: run a policy over a HELD-OUT scenario split and aggregate the
behavioural metrics that matter for a cost-saving, permission-respecting agent.

The held-out split uses a different seed from the training scenarios, so the
numbers measure generalization to unseen hidden needs, not train-set recall.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable

from shoprl.data.catalog import Product, catalog_index
from shoprl.env.pennyenv import PennyEnv
from shoprl.env.scenario import Scenario, generate_scenarios

# Training scenarios use seeds 0..; hold eval out at a disjoint seed range.
HELDOUT_SEED = 1000


def heldout_scenarios(catalog: list[Product], n: int = 200,
                      seed: int = HELDOUT_SEED,
                      valid_target_range: tuple[int, int] = (20, 80)
                      ) -> list[Scenario]:
    """Scenarios for evaluation — a disjoint seed from any training set."""
    return generate_scenarios(catalog, n=n, seed=seed,
                              valid_target_range=valid_target_range)


@dataclass
class EpisodeResult:
    scenario_id: str
    asked: bool                 # issued >=1 clarifying ASK
    violation: bool             # added without permission
    value_quality: float
    accepted: float
    turns_to_rec: int | None    # env turn of the first RECOMMEND (None if never)
    total: float


@dataclass
class EvalReport:
    policy: str
    n: int
    ask_rate: float
    violation_rate: float
    mean_value_quality: float
    accept_rate: float
    mean_turns_to_rec: float | None
    mean_total: float

    def as_row(self) -> str:
        ttr = f"{self.mean_turns_to_rec:.2f}" if self.mean_turns_to_rec is not None else "—"
        return (f"{self.policy:<16} n={self.n:<4} ask={self.ask_rate:.2f} "
                f"viol={self.violation_rate:.2f} value={self.mean_value_quality:.3f} "
                f"accept={self.accept_rate:.2f} turns2rec={ttr} "
                f"total={self.mean_total:+.3f}")


def run_episode(catalog: list[Product], scenario: Scenario,
                idx: dict[str, Product], policy, max_turns: int = 12
                ) -> EpisodeResult:
    env = PennyEnv(catalog, scenario, idx=idx, max_turns=max_turns)
    user = env.reset()
    policy.reset(scenario, idx)
    dialogue: list[tuple[str, str]] = [("user", user)]
    done = False
    while not done:
        action = policy.act(dialogue)
        user, done, _ = env.step(action)
        dialogue += [("agent", action), ("user", user)]

    r = env.reward()
    turns_to_rec = next((t.turn for t in env.turns if t.action == "RECOMMEND"), None)
    return EpisodeResult(
        scenario_id=scenario.scenario_id,
        asked=any(t.action == "ASK" for t in env.turns),
        violation=bool(env.state.acted_without_permission),
        value_quality=r.value_quality,
        accepted=r.accepted,
        turns_to_rec=turns_to_rec,
        total=r.total,
    )


def evaluate(catalog: list[Product], scenarios: list[Scenario],
             policy_factory: Callable[[], object], name: str,
             max_turns: int = 12) -> EvalReport:
    """Run a fresh policy (`policy_factory()`) per scenario and aggregate."""
    idx = catalog_index(catalog)
    results = [run_episode(catalog, s, idx, policy_factory(), max_turns)
               for s in scenarios]
    ttr = [r.turns_to_rec for r in results if r.turns_to_rec is not None]
    mean = statistics.mean
    return EvalReport(
        policy=name,
        n=len(results),
        ask_rate=mean(r.asked for r in results),
        violation_rate=mean(r.violation for r in results),
        mean_value_quality=mean(r.value_quality for r in results),
        accept_rate=mean(r.accepted for r in results),
        mean_turns_to_rec=(mean(ttr) if ttr else None),
        mean_total=mean(r.total for r in results),
    )
