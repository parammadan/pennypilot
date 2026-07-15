# Pennywise — Architecture

A multi-turn RL post-training **pipeline** — rollout → reward → RLOO
optimization → eval → checkpoint — validated on a cost-saving shopping agent. The
**pipeline is the deliverable**; the shopping agent is the workload that exercises
it. The framing target is training-system engineering (stability, observability,
efficiency), not the agent product.

## Provenance (honest)

Pennywise **evolves the multi-turn substrate of `shoprl-fabric`** by adding the
three things that make the task teach clarification and safety:

1. **hidden needs** — the shopper's budget and must-have are not stated, so the
   agent must *ask* to discover them;
2. a **permission gate** — the agent may only add to the cart after an explicit
   user accept; and
3. an **info-gain reward** — a dense per-turn signal for a clarifying question
   that reduces uncertainty about the hidden need.

Reused from `shoprl-fabric` (single machine → this repo, vendored): the frozen
catalog generator, the verifiable rule-reward pattern, the RLOO-from-GRPO
adaptation, and the trajectory credit-assignment structure. **What this repo
reports is only what this project measures.** ShopRL's KL numbers (RLOO KL 0.015
vs PPO 6.78) are cited only as the *reason* RLOO was chosen — they are not
presented as Pennywise's results.

## 1. Three planes

```
  ┌──────────────────────────── ENVIRONMENT PLANE (Phase 1, done, CPU) ───────────────────────────┐
  │  catalog (frozen, verifiable)  ──►  scenario generator (hidden budget + must-have)             │
  │                                       │                                                        │
  │  user-simulator ── two strictly separate seams:                                                │
  │     • ConversationModel  (generation; ScriptedConversation now, FrozenLLMConversation later)   │
  │     • judge_accept       (judgment; rule-based, the ONLY accept/reject authority)              │
  │                                       │                                                        │
  │  PennyEnv (ASK / RECOMMEND / ASK_PERMISSION / ADD_TO_CART) + verifiable reward + info-gain      │
  └───────────────────────────────────────┬────────────────────────────────────────────────────┘
                                           │ trajectories + verifiable reward
  ┌──────────────────────────── TRAINING PLANE (Phase 2, designed) ──────────────────────────────┐
  │  SFT warmup (1k synthetic demos)  ──►  multi-turn RLOO loop:                                   │
  │     reset simulator to identical hidden-state → sample k=8 full conversations                  │
  │     → verifiable reward per trajectory → leave-one-out advantage → KL-controlled update        │
  │     → checkpoint                                                                               │
  └───────────────────────────────────────┬────────────────────────────────────────────────────┘
                                           │ checkpoints (S3), metrics.jsonl
  ┌──────────────────────────── INFRASTRUCTURE PLANE (Phase 3, designed) ─────────────────────────┐
  │  1× V100 32GB (fp16 train, SLURM)  ──S3──►  2× A10G (bf16 vLLM rollout, spot)                  │
  │  observability: metrics.jsonl per step (KL, reward, throughput, GPU util) → dashboard          │
  │  reliability: checkpoint/resume across SLURM preemption + spot interruption                    │
  └───────────────────────────────────────────────────────────────────────────────────────────┘
```

Only the **environment plane** is built and tested (CPU, no GPU). The training and
infrastructure planes are designed; see `PHASE2_NOTES.md` for the concrete
blockers to clear before the first GPU run.

## 2. Hidden-need environment

`env/scenario.py` samples a `Scenario` = `{category, hidden_budget, must_have}`
plus a deliberately vague opening utterance that mentions **neither**. The full
hidden constraint set is derived from the catalog, so "valid" is a verifiable
ground-truth predicate (`valid_skus`), not a judgment call.

- **Undiscoverable without asking, by construction.** The opener leaks no
  number and no feature; the simulator reveals a field only in response to a
  relevant clarifying question. So asking is the reward-maximizing strategy
  *structurally* — the information needed to pick a valid, cheapest item is not
  on the table until the agent asks for it.
- **Valid-set size is a tunable knob** (`valid_target_range`, default dozens),
  because `value_quality` grades by price-rank among valid items: a valid set of
  size ~2 collapses that into a near-binary signal, so we size it to dozens to
  keep "cheapest valid" a meaningful ranking with real gradient.

`env/pennyenv.py` (`PennyEnv`) owns the conversation state machine over the
action grammar `ASK[budget|feature] | RECOMMEND[SKU] | ASK_PERMISSION[SKU?] |
ADD_TO_CART[SKU]`, and delegates the two simulator seams (§4).

## 3. Permission gate + info-gain reward

Per completed trajectory (`env/reward.py::pennywise_reward`):

```
R = 0.4·value_quality + 0.4·accepted + 0.2·asked_permission
    − 1.0·acted_without_permission + Σ per_turn_info_gain
```

- **`value_quality`** — gated to 0 if the recommended item violates a hard
  constraint (over budget / missing must-have); else graded by price-rank among
  the valid set (cheapest valid = 1.0). This is the cost-saving objective.
- **`accepted`** — from the programmatic `judge_accept`, an objective fit check,
  never persuasion.
- **Permission gate (`−1.0`, hard floor).** An add is legitimate only after an
  explicit accept; acting without one is a floor, *not* traded against value. It
  is structural, not a soft cost: acting without permission never produces a
  legit cart, so the violating trajectory has no value/accepted term to offset
  the −1.0 (measured: good +1.14 vs add-without-permission −1.00).
- **`per_turn_info_gain`** — a clarifying question is credited on the turn it was
  asked, by how much it shrinks the set of catalog items still consistent with
  what the agent knows (bits of uncertainty removed). This is the dense signal
  that counters the "RL kills clarification" collapse.

**Credit assignment (no parallel reward).** The four *outcome* terms form the
trajectory scalar that `assign_credit` spreads across turns (uniform, or
γ-discounted if learning stalls); info-gain is passed to the *same*
`assign_credit` as a `per_turn_bonus`, added on the turn it occurred. One credit
path, reused — not a divergent second reward.

## 4. User-simulator seams (swappable generation, fixed judgment)

Two objects, strictly separated (`env/simulator.py`):

| Seam | Object | Now | Later | Rule |
|---|---|---|---|---|
| Generation | `ConversationModel` | `ScriptedConversation` | `FrozenLLMConversation` | only phrases the user turn |
| Judgment | `judge_accept` | rule-based | **unchanged** | the ONLY accept/reject authority |

The env calls both through the interface, never a concrete model directly. The
reward reads the *decision* (`judge_accept`), never the *words* — so a
conversation model cannot be persuaded into accepting a bad item (reward
hacking), and the accept signal stays verifiable. `FrozenLLMConversation` takes
an injected `generate(prompt)->str` so it is CPU-testable with a stub today and
swaps to a real frozen instruct LLM in Phase 2 without touching env/reward/judge.

## 5. RLOO from GRPO (Phase 2, designed)

Method: RLVR-dominant verifiable reward + one model-based acceptance signal,
optimized with **RLOO** (REINFORCE leave-one-out) — critic-free, chosen for
stability. RLOO reuses the group-of-samples machinery of GRPO and changes only
the baseline:

- **GRPO:** advantage for sample *i* = (r_i − mean of ALL samples) / group std.
- **RLOO:** advantage for sample *i* = r_i − mean of the *other* samples
  (leave-one-out), no std rescaling — a provably-unbiased, cleaner critic-free
  estimator at identical cost.

The rest of the loop (clipped surrogate + KL-to-reference) is held fixed so the
only measured difference from a GRPO baseline is the baseline choice. Rollouts
reset the simulator to an *identical hidden state*, then sample k=8 full
conversations per group; the trajectory advantage is applied to all tokens to
start, with γ≈0.7 turn-discounting added only if learning stalls. KL drift from
the SFT reference is the core stability lever (final + max KL logged per step).

## 6. Evaluation

`eval/harness.py` runs any policy over a **held-out** scenario split (disjoint
seed from training — a generalization measure, not recall) and reports ask-rate,
permission-violation rate, mean `value_quality`, accept-rate, and mean
turns-to-recommendation. Three scripted reference policies bound a trained run
through the same `Policy` interface: `OracleGood` (ceiling), `NoAskBaseline`
(value floor: never asks → value ~0.55 vs 1.0), `Violation` (safety floor:
violation-rate 1.0, value 0).

## 7. Real vs simulated (authoritative)

- **Real (built + tested, CPU):** the whole environment plane — catalog +
  hidden-need scenarios, `PennyEnv`, verifiable reward + info-gain, the two
  simulator seams, SFT demo generator, eval harness + reference policies. 24
  tests.
- **Designed, not yet run:** the training plane (SFT-train + multi-turn RLOO on
  V100) and the infrastructure plane (A10G vLLM rollout, S3 bridge, dashboard).
- **Cited, not this project's result:** ShopRL's RLOO-vs-PPO KL numbers (choice
  rationale only).

## 8. V100 VRAM note — **to be re-profiled in Phase 2**

> ⚠️ The scaling numbers previously recorded here were for **Qwen2.5-3B,
> single-turn, LoRA** on a V100. **That profile does NOT hold for Pennywise**,
> which is **Qwen-1.5B instruct, full fine-tune, multi-turn**. Two things change
> the memory math materially and must be measured before the first real run:
> 1. **Full fine-tune ≠ LoRA** — all parameters carry gradients + AdamW moment
>    state (not a tiny adapter), a different (and larger) optimizer-state
>    footprint.
> 2. **Multi-turn sequences are far longer than single-turn** — and Volta has
>    **no FlashAttention**, so attention memory is quadratic in sequence length.
>    The max viable sequence length must be measured first; it may cap
>    `max_turns` or force LoRA even on a 32GB card.
>
> No VRAM/throughput number is claimed for Pennywise until it is measured on the
> V100 (see `PHASE2_NOTES.md`).
