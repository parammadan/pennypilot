"""Multi-turn RLOO trainer.

One optimizer step:
  1. for each prompt (scenario), reset the simulator to an IDENTICAL hidden state
     and sample k full-conversation rollouts with the current policy (on-policy);
  2. score each trajectory with the verifiable reward (`pennywise_reward.total`);
  3. leave-one-out advantage: A_i = r_i − mean(r_{−i}) (no std normalization —
     the RLOO baseline, swapped in for GRPO's group-mean+std);
  4. per trajectory (micro-batched to batch=1, since full-FT can't forward all k
     long sequences at once): teacher-force forward → per-token logprobs of the
     AGENT tokens; REINFORCE loss −A·Σlogp + β·KL(policy‖frozen SFT reference),
     accumulate gradient;
  5. one optimizer.step() after the whole group.

Trajectory-level advantage applied to all agent tokens (turn-discounting is a
later lever if learning stalls). KL to the frozen SFT reference is the stability
lever and is logged every step.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

import torch

from shoprl.data.catalog import Product
from shoprl.env.pennyenv import PennyEnv
from shoprl.env.scenario import Scenario
from shoprl.train.sft import encode


# ---- rollout -------------------------------------------------------------
@torch.no_grad()
def _generate_action(policy, tok, messages, device, max_new_tokens, temperature):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  tokenize=True, return_dict=True,
                                  return_tensors="pt")
    ids = enc["input_ids"].to(device)
    out = policy.generate(ids, max_new_tokens=max_new_tokens,
                          do_sample=temperature > 0,
                          temperature=max(temperature, 1e-5), top_p=0.95,
                          pad_token_id=tok.pad_token_id)
    txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    return txt.splitlines()[0].strip() if txt else ""


def rollout(policy, tok, catalog, scenario, idx, device, *, max_turns,
            max_new_tokens, temperature) -> dict:
    """One full conversation with the current policy vs the simulator."""
    env = PennyEnv(catalog, scenario, idx=idx, max_turns=max_turns)
    opener = env.reset()
    messages = [{"role": "user", "content": opener}]
    done = False
    while not done:
        action = _generate_action(policy, tok, messages, device, max_new_tokens,
                                  temperature)
        user, done, _ = env.step(action)
        messages += [{"role": "assistant", "content": action},
                     {"role": "user", "content": user}]
    r = env.reward()
    return {
        "messages": messages,
        "reward": r.total,
        "value": r.value_quality,
        "asked": 1.0 if any(t.action == "ASK" for t in env.turns) else 0.0,
        "violation": float(r.acted_without_permission),
        "accepted": r.accepted,
    }


# ---- advantage + logprobs ------------------------------------------------
def rloo_advantages(rewards: list[float]) -> list[float]:
    k = len(rewards)
    if k <= 1:
        return [0.0] * k
    s = sum(rewards)
    return [rewards[i] - (s - rewards[i]) / (k - 1) for i in range(k)]


def _sequence_and_mask(tok, messages, max_len: int):
    """Full token ids + boolean mask (True on AGENT/assistant tokens)."""
    full = encode(tok, messages, add_generation_prompt=False)
    mask = [False] * len(full)
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        p = len(encode(tok, messages[:i], add_generation_prompt=True))
        q = len(encode(tok, messages[:i + 1], add_generation_prompt=False))
        for j in range(p, min(q, len(full))):
            mask[j] = True
    return full[:max_len], mask[:max_len]


def _token_logprobs(model, ids: list[int], device):
    """Per-token logprob of the realized next token, positions 1..T-1."""
    t = torch.tensor([ids], device=device)
    logits = model(t).logits[0]                       # [T, V]
    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp[:-1].gather(-1, t[0, 1:].unsqueeze(-1)).squeeze(-1)  # [T-1]


def _k3_kl(logp_pol, logp_ref):
    """k3 estimator (nonnegative, low variance) per token."""
    d = logp_ref - logp_pol
    return torch.exp(d) - d - 1.0


# ---- one training step ---------------------------------------------------
@dataclass
class RLOOConfig:
    k: int = 8
    prompts_per_step: int = 1
    max_turns: int = 12
    max_new_tokens: int = 32
    max_len: int = 640
    temperature: float = 1.0
    beta: float = 0.04
    lr: float = 1e-6
    max_grad_norm: float = 1.0


def rloo_step(policy, reference, tok, optimizer, catalog, scenarios, idx, device,
              cfg: RLOOConfig) -> dict:
    n = cfg.k * len(scenarios)
    optimizer.zero_grad(set_to_none=True)
    rewards_all, kls, values, asks, viols, advs_all = [], [], [], [], [], []

    for scen in scenarios:
        # 1-2. rollout k trajectories (generation config: cache on, ckpt off).
        policy.eval()
        policy.config.use_cache = True
        try:
            policy.gradient_checkpointing_disable()
        except Exception:
            pass
        trajs = [rollout(policy, tok, catalog, scen, idx, device,
                         max_turns=cfg.max_turns, max_new_tokens=cfg.max_new_tokens,
                         temperature=cfg.temperature) for _ in range(cfg.k)]
        rewards = [t["reward"] for t in trajs]
        advs = rloo_advantages(rewards)
        rewards_all += rewards
        advs_all += advs
        values += [t["value"] for t in trajs]
        asks += [t["asked"] for t in trajs]
        viols += [t["violation"] for t in trajs]

        # 3-4. micro-batched policy gradient (train config: ckpt on, cache off).
        policy.train()
        policy.config.use_cache = False
        try:
            policy.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:
            pass
        for t, A in zip(trajs, advs):
            ids, mask = _sequence_and_mask(tok, t["messages"], cfg.max_len)
            asst = torch.tensor(mask[1:], device=device, dtype=torch.bool)
            if asst.sum() == 0:
                continue
            lp_pol = _token_logprobs(policy, ids, device)
            with torch.no_grad():
                lp_ref = _token_logprobs(reference, ids, device)
            pg = -A * lp_pol[asst].sum()
            kl = _k3_kl(lp_pol[asst], lp_ref[asst])
            loss = (pg + cfg.beta * kl.sum()) / n
            loss.backward()
            kls.append(kl.mean().item())

    grad_norm = torch.nn.utils.clip_grad_norm_(
        (p for p in policy.parameters() if p.requires_grad), cfg.max_grad_norm)
    optimizer.step()

    mean = statistics.mean
    return {
        "reward_mean": mean(rewards_all),
        "reward_std": statistics.pstdev(rewards_all) if len(rewards_all) > 1 else 0.0,
        "adv_abs_mean": mean(abs(a) for a in advs_all),
        "kl_mean": mean(kls) if kls else 0.0,
        "value_mean": mean(values),
        "ask_rate": mean(asks),
        "violation_rate": mean(viols),
        "grad_norm": float(grad_norm),
        "group_size": cfg.k,
        "n_prompts": len(scenarios),
    }
