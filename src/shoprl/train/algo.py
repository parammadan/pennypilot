"""Thin algorithm interface (approved design: train_step + advantage seams
ONLY — no framework growth).

The comparison the project pre-registered (HYPOTHESES.md) isolates exactly one
thing: how a group of trajectory rewards becomes per-trajectory advantages.
Everything else (rollout, reward, masking, KL, optimizer step) is shared in
`rl_step`, so a GRPO-vs-RLOO difference is attributable to the advantage rule
alone. Advantage rules are pure Python (CPU-testable); torch enters only
inside `rl_step`.

Module layout:
  RLOO / GRPO / GRPONoStd  — advantage seams (pre-registered variants)
  rollout_v2(agent_fn,env) — one conversation vs SyntheticCatalogEnvironment
                             (agent_fn injected: scripted for tests, model on GPU)
  rl_step(...)             — the shared train step (GPU)
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field


# ---- advantage seams -------------------------------------------------------
class RLOO:
    """Leave-one-out baseline, no rescaling (unbiased; flat group -> zeros)."""
    name = "rloo"

    def advantages(self, rewards: list[float]) -> list[float]:
        k = len(rewards)
        if k <= 1:
            return [0.0] * k
        s = sum(rewards)
        return [r - (s - r) / (k - 1) for r in rewards]


class GRPO:
    """Group mean baseline with std division — the rule H1 puts on trial:
    as group variance collapses, the division amplifies noise."""
    name = "grpo"

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def advantages(self, rewards: list[float]) -> list[float]:
        if len(rewards) <= 1:
            return [0.0] * len(rewards)
        mean = statistics.mean(rewards)
        std = statistics.pstdev(rewards)
        return [(r - mean) / (std + self.eps) for r in rewards]


class GRPONoStd:
    """Pre-registered fallback: group-mean baseline WITHOUT the std division —
    isolates the hypothesized instability mechanism in one config line."""
    name = "grpo-nostd"

    def advantages(self, rewards: list[float]) -> list[float]:
        if len(rewards) <= 1:
            return [0.0] * len(rewards)
        mean = statistics.mean(rewards)
        return [r - mean for r in rewards]


ALGORITHMS = {a.name: a for a in (RLOO(), GRPO(), GRPONoStd())}


# ---- rollout ---------------------------------------------------------------
@dataclass
class TrajectoryV2:
    messages: list[dict]           # system + user/env + assistant turns
    reward: float                  # outcome.total (the trajectory scalar)
    value_quality: float
    violation: bool
    asked: int
    invalid_actions: int
    turns: int
    per_turn_info_gain: list[float] = field(default_factory=list)


def rollout_v2(agent_fn, env, system: str, max_steps: int = 15) -> TrajectoryV2:
    """One full conversation with `agent_fn(messages)->str` against a reset
    v2 environment. Injecting the generation fn keeps this CPU-testable and
    engine-agnostic (HF now, vLLM behind the same signature in SS1)."""
    opener = env.reset()
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": opener}]
    done = False
    steps = 0
    while not done and steps < max_steps:
        action_text = agent_fn(messages)
        step = env.execute_text(action_text)
        messages += [{"role": "assistant", "content": action_text},
                     {"role": "user", "content": step.observation}]
        done = step.done
        steps += 1
    out = env.calculate_outcome()
    return TrajectoryV2(
        messages=messages,
        reward=out.total,
        value_quality=out.value_quality,
        violation=bool(out.acted_without_permission),
        asked=sum("revealed" in t.note for t in env.turns),
        invalid_actions=sum(not t.valid for t in env.turns),
        turns=len(env.turns),
        per_turn_info_gain=[t.info_gain for t in env.turns],
    )


# ---- shared train step (GPU) -----------------------------------------------
@dataclass
class RLConfigV2:
    algo: str = "rloo"             # bring-up default; comparison flips this
    k: int = 8
    prompts_per_step: int = 1
    max_turns: int = 12
    max_new_tokens: int = 64
    max_len: int = 2048
    temperature: float = 1.0
    beta: float = 0.04
    lr: float = 1e-6
    max_grad_norm: float = 1.0
    language: str = "es-en"


def rl_step(policy, reference, tok, optimizer, scaler, catalog, scenarios, idx,
            cfg: RLConfigV2, system: str, annotate=None) -> dict:
    """One optimizer step: k sampled conversations per scenario (identical
    hidden state), rewards -> algo advantages -> micro-batched per-token PG +
    k3-KL to the frozen reference. fp16 recipe: autocast + GradScaler, same
    precision path for policy and reference logprobs (no phantom KL)."""
    import torch

    from shoprl.env.catalog_env import SyntheticCatalogEnvironment
    from shoprl.train.rloo import _k3_kl, _sequence_and_mask, _token_logprobs

    algo = ALGORITHMS[cfg.algo]

    @torch.no_grad()
    def agent_fn(messages):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      tokenize=True, return_dict=True,
                                      return_tensors="pt")
        ids = enc["input_ids"].to("cuda")
        out = policy.generate(ids, max_new_tokens=cfg.max_new_tokens,
                              do_sample=cfg.temperature > 0,
                              temperature=max(cfg.temperature, 1e-5), top_p=0.95,
                              pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    n = cfg.k * len(scenarios)
    optimizer.zero_grad(set_to_none=True)
    rewards_all, advs_all, kls = [], [], []
    values, viols, asks, invalids = [], [], [], []

    mark = annotate or (lambda label: None)   # Stage-6 capture seam (SS0)
    for scen in scenarios:
        policy.eval()
        policy.config.use_cache = True
        try:
            policy.gradient_checkpointing_disable()
        except Exception:
            pass
        mark("rollout:start")
        trajs = []
        for _ in range(cfg.k):
            env = SyntheticCatalogEnvironment(catalog, scen, idx=idx,
                                              max_turns=cfg.max_turns,
                                              language=cfg.language)
            trajs.append(rollout_v2(agent_fn, env, system))
        mark("rollout:end")
        rewards = [t.reward for t in trajs]
        advs = algo.advantages(rewards)
        rewards_all += rewards
        advs_all += advs
        values += [t.value_quality for t in trajs]
        viols += [float(t.violation) for t in trajs]
        asks += [float(t.asked > 0) for t in trajs]
        invalids += [t.invalid_actions for t in trajs]

        policy.train()
        policy.config.use_cache = False
        try:
            policy.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:
            pass
        mark("update:start")
        for t, A in zip(trajs, advs):
            ids, mask = _sequence_and_mask(tok, t.messages, cfg.max_len)
            asst = torch.tensor(mask[1:], device="cuda", dtype=torch.bool)
            if asst.sum() == 0:
                continue
            with torch.autocast("cuda", dtype=torch.float16):
                lp_pol = _token_logprobs(policy, ids, "cuda")
                with torch.no_grad():
                    lp_ref = _token_logprobs(reference, ids, "cuda")
            pg = -A * lp_pol[asst].mean()          # per-token normalization
            kl = _k3_kl(lp_pol[asst], lp_ref[asst])
            loss = (pg + cfg.beta * kl.mean()) / n
            scaler.scale(loss).backward()
            kls.append(float(kl.mean().item()))
        mark("update:end")

    mark("optimizer:start")
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        (p for p in policy.parameters() if p.requires_grad), cfg.max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    mark("optimizer:end")

    mean = statistics.mean
    return {
        "algo": cfg.algo,
        "reward_mean": mean(rewards_all),
        "reward_std": statistics.pstdev(rewards_all) if len(rewards_all) > 1 else 0.0,
        "adv_abs_mean": mean(abs(a) for a in advs_all),
        "kl_mean": mean(kls) if kls else 0.0,
        "kl_max": max(kls) if kls else 0.0,
        "value_mean": mean(values),
        "violation_rate": mean(viols),
        "ask_rate": mean(asks),
        "invalid_actions_mean": mean(invalids),
        "grad_norm": float(grad_norm),
        "scaler_scale": float(scaler.get_scale()),
        "group_size": cfg.k,
    }
