"""Multi-turn Shopping Environment Simulator for trajectory-level RL.

A practice environment on top of the completed single-turn pipeline: the model
takes multiple actions (add/remove/filter/checkout) across turns, and the reward
is assigned at the END of the episode (delayed / trajectory-level) — the setting
where credit assignment and reward variance actually get hard.

This module is the SIMULATOR only (state, actions, transitions, episode). The
trajectory reward + credit assignment is built separately (taught step by step);
nothing here is wired into training yet.
"""
from shoprl.env.pennyenv import (PennyEnv, PennyState, PennyTurn,
                                 format_candidates, search_catalog)
from shoprl.env.reward import (PennyBreakdown, assign_credit, episode_reward,
                               pennywise_reward, value_quality)
from shoprl.env.scenario import Scenario, generate_scenarios, scenario_valid_skus
from shoprl.env.shopenv import Action, EnvState, Goal, ShopEnv, parse_action
from shoprl.env.simulator import (ConversationModel, FrozenLLMConversation,
                                  ScriptedConversation, judge_accept)

__all__ = [
    # single-goal ShopEnv (legacy substrate)
    "Action", "EnvState", "Goal", "ShopEnv", "parse_action",
    "episode_reward", "assign_credit",
    # Pennywise hidden-need + permission
    "Scenario", "generate_scenarios", "scenario_valid_skus",
    "judge_accept", "ConversationModel", "ScriptedConversation",
    "FrozenLLMConversation",
    "PennyEnv", "PennyState", "PennyTurn", "search_catalog", "format_candidates",
    "pennywise_reward", "value_quality", "PennyBreakdown",
]
