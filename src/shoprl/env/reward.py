"""Trajectory-level reward + credit assignment (no training wiring).

Two pieces:
  1. episode_reward(state, goal, idx) -> one scalar scoring the COMPLETED episode
     (valid purchase within budget/constraints + cart quality). This is the
     delayed, end-of-episode signal.
  2. assign_credit(reward, n_turns, scheme) -> per-turn advantages, i.e. how that
     one scalar is spread across the episode's turns. Two schemes implemented:
       - "uniform": every turn gets (reward - baseline)  [outcome-supervised;
         GRPO/RLOO/PPO lifted to the trajectory level]
       - "discounted": turn t gets gamma^(T-1-t) * (reward - baseline)  [later
         turns, nearer the reward, get more credit]

Kept pure + framework-free so it's unit-testable and, later, droppable into a
trajectory trainer. Nothing here touches the training loop yet.
"""
from __future__ import annotations

from dataclasses import dataclass

from shoprl.data.catalog import Product
from shoprl.data.prompts import satisfies
from shoprl.env.scenario import Scenario
from shoprl.env.shopenv import EnvState, Goal


def episode_reward(state: EnvState, goal: Goal, idx: dict[str, Product]) -> float:
    """Score a finished episode in [0, 1].

    Gates: must have CHECKED OUT with a non-empty cart (else 0 — no purchase).
    Budget is env-enforced (over-budget adds are rejected), so a checked-out cart
    is always within budget. Score then blends:
      - constraint satisfaction: fraction of cart items meeting the goal's
        constraints (the catalog's TRUE specs — verifiable, un-hackable)
      - item-count match: how close |cart| is to the target count
    """
    if not state.checked_out or not state.cart:
        return 0.0
    constraints_ok = sum(satisfies(idx[s], goal.constraints) for s in state.cart) / len(state.cart)
    if goal.target_items > 0:
        count_ok = max(0.0, 1.0 - abs(len(state.cart) - goal.target_items) / goal.target_items)
    else:
        count_ok = 1.0
    return round(0.6 * constraints_ok + 0.4 * count_ok, 4)


def assign_credit(reward: float, n_turns: int, scheme: str = "uniform",
                  gamma: float = 0.99, baseline: float = 0.0,
                  per_turn_bonus: list[float] | None = None) -> list[float]:
    """Spread the trajectory reward across `n_turns` as per-turn advantages.

    The advantage a turn receives is what the policy gradient actually uses; the
    baseline (e.g. mean reward of sibling episodes for the same goal) is what
    turns the raw reward into a low-variance, centered signal.

    `per_turn_bonus` is a genuinely per-turn signal (e.g. Pennywise's
    info-gain: a clarifying question is credited on the turn it was asked) added
    ON TOP of the spread trajectory reward. It reuses this one credit path rather
    than introducing a parallel reward: the *outcome* reward (value / accepted /
    permission) is the trajectory scalar that gets spread; the *dense* signal is
    the per-turn bonus. Its length must equal `n_turns`.
    """
    if n_turns <= 0:
        return []
    adv = reward - baseline
    if scheme == "uniform":
        credit = [adv] * n_turns
    elif scheme == "discounted":
        credit = [round(gamma ** (n_turns - 1 - t) * adv, 6) for t in range(n_turns)]
    else:
        raise ValueError(f"unknown credit scheme: {scheme!r}")
    if per_turn_bonus is not None:
        if len(per_turn_bonus) != n_turns:
            raise ValueError(
                f"per_turn_bonus length {len(per_turn_bonus)} != n_turns {n_turns}")
        credit = [round(c + b, 6) for c, b in zip(credit, per_turn_bonus)]
    return credit


# ---------------------------------------------------------------------------
# Pennywise hidden-need + permission reward (multi-turn, verifiable).
#
# R = 0.4·value_quality + 0.4·accepted + 0.2·asked_permission
#     − 1.0·acted_without_permission + Σ per_turn_info_gain
#
# The first four terms are the OUTCOME scalar (spread across turns by
# assign_credit); info-gain is the per-turn dense term (passed to assign_credit
# as per_turn_bonus). The permission penalty is a hard floor, not traded against
# value: acting without an explicit accept never yields a legit cart, so the
# violating trajectory gets no value/accepted term to offset the −1.0 — that is
# what makes the gate structural rather than a soft cost.
# ---------------------------------------------------------------------------

DEFAULT_PENNY_WEIGHTS = {"value": 0.4, "accepted": 0.4, "permission": 0.2}
PERMISSION_VIOLATION_PENALTY = 1.0


@dataclass
class PennyBreakdown:
    value_quality: float
    accepted: float
    asked_permission: float
    acted_without_permission: float
    info_gain: float
    outcome: float   # the trajectory scalar assign_credit spreads across turns
    total: float     # outcome + info_gain — the reported/logged reward


def value_quality(sku: str | None, scenario: Scenario,
                  idx: dict[str, Product]) -> float:
    """Grade a recommendation in [0, 1] by price-rank among the VALID set.

    Gated to 0 if the item violates a hard constraint (not in `valid_skus` =
    over budget or missing the must-have). Otherwise the cheapest valid item
    scores 1.0 and the priciest valid item scores 0.0, linear in rank — so the
    reward-maximizing valid pick is the cheapest one (the cost-saving behaviour).
    """
    if not sku or sku not in scenario.valid_skus:
        return 0.0
    prices = sorted(idx[s].price for s in scenario.valid_skus)
    if len(prices) == 1:
        return 1.0
    rank = prices.index(idx[sku].price)  # 0 = cheapest valid
    return round(1.0 - rank / (len(prices) - 1), 4)


def pennywise_reward(
    *,
    chosen_sku: str | None,
    accepted: bool,
    asked_permission: bool,
    acted_without_permission: bool,
    info_gains: list[float],
    scenario: Scenario,
    idx: dict[str, Product],
    weights: dict[str, float] | None = None,
) -> PennyBreakdown:
    """Score a completed Pennywise trajectory. `chosen_sku` is the item actually
    in the cart (a LEGITIMATELY added item); it is None when nothing was legally
    added (including after a permission violation), which is exactly why the
    −1.0 floor cannot be bought back with value."""
    w = weights or DEFAULT_PENNY_WEIGHTS
    vq = value_quality(chosen_sku, scenario, idx)
    acc = 1.0 if accepted else 0.0
    ask = 1.0 if asked_permission else 0.0
    viol = 1.0 if acted_without_permission else 0.0
    info = round(sum(info_gains), 6)
    outcome = round(
        w["value"] * vq + w["accepted"] * acc + w["permission"] * ask
        - PERMISSION_VIOLATION_PENALTY * viol,
        6,
    )
    return PennyBreakdown(
        value_quality=vq, accepted=acc, asked_permission=ask,
        acted_without_permission=viol, info_gain=info,
        outcome=outcome, total=round(outcome + info, 6),
    )
