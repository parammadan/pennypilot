"""Evaluation for the Pennywise agent: run any policy over a held-out scenario
split and report behavioural metrics (ask-rate, permission-violation rate, mean
value_quality, accept-rate, turns-to-recommendation). Scripted oracle policies
give a floor/ceiling to compare a trained policy against."""
from shoprl.eval.harness import (EpisodeResult, EvalReport, evaluate,
                                 heldout_scenarios, run_episode)
from shoprl.eval.policies import (NoAskBaselinePolicy, OracleGoodPolicy, Policy,
                                  ViolationPolicy)

__all__ = [
    "EpisodeResult", "EvalReport", "evaluate", "heldout_scenarios", "run_episode",
    "Policy", "OracleGoodPolicy", "NoAskBaselinePolicy", "ViolationPolicy",
]
