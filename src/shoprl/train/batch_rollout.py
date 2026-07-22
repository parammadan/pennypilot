"""Batched multi-episode rollouts — N conversations stepped in lockstep.

The single-GPU answer to rollout dominance (SS0: 95.8% of the iteration):
instead of one conversation leaving the GPU idle during env steps, N envs
advance together — each round gathers the prompts of all UNFINISHED episodes,
generates them as ONE batch, then steps each env on CPU while the next batch
forms. One episode's env-step gap is filled by other episodes' tokens
(SS4/SS4b measure exactly this).

Engine-agnostic via `generate_batch(list_of_message_lists) -> list[str]`:
tests inject a scripted function; SS4 injects vLLM; the trainer can inject
either. Deliberately synchronous lockstep (not asyncio, not multiprocess) —
the approved design hint, and the simplest thing that fills the gaps.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LiveEpisode:
    env: object
    messages: list[dict]
    done: bool = False
    steps: int = 0
    traj: object | None = None
    info_gains: list[float] = field(default_factory=list)


def batched_rollouts(generate_batch, envs: list, system: str,
                     max_steps: int = 15) -> list:
    """Run one full episode in every env, batching generation across them.

    Returns TrajectoryV2 objects in env order (same fields as rollout_v2, so
    the trainer's reward/masking path is unchanged)."""
    from shoprl.train.algo import TrajectoryV2

    eps: list[LiveEpisode] = []
    for env in envs:
        opener = env.reset()
        eps.append(LiveEpisode(env=env, messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": opener}]))

    while True:
        live = [e for e in eps if not e.done]
        if not live:
            break
        actions = generate_batch([e.messages for e in live])
        if len(actions) != len(live):
            raise RuntimeError(f"generate_batch returned {len(actions)} "
                               f"for {len(live)} episodes")
        for e, action_text in zip(live, actions):
            step = e.env.execute_text(action_text)
            e.messages += [{"role": "assistant", "content": action_text},
                           {"role": "user", "content": step.observation}]
            e.steps += 1
            if step.done or e.steps >= max_steps:
                e.done = True

    out = []
    for e in eps:
        o = e.env.calculate_outcome()
        out.append(TrajectoryV2(
            messages=e.messages,
            reward=o.total,
            value_quality=o.value_quality,
            violation=bool(o.acted_without_permission),
            asked=sum("revealed" in t.note for t in e.env.turns),
            invalid_actions=sum(not t.valid for t in e.env.turns),
            turns=len(e.env.turns),
            per_turn_info_gain=[t.info_gain for t in e.env.turns],
        ))
    return out
