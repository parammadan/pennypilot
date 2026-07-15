# Pennywise

A multi-turn RL **post-training pipeline** — rollout → reward → RLOO
optimization → eval → checkpoint — validated on a cost-saving shopping agent that
asks clarifying questions, recommends the cheapest item that fits, and only adds
to the cart with explicit permission.

**The pipeline is the deliverable; the shopping agent is the workload.** Framing
target: training-system engineering (stability, observability, efficiency) on
constrained hardware — 1× V100 32 GB (train) + 2× A10G (rollout).

> Status: **Phase 1 complete** (environment plane, CPU, 25 tests). **Phase 2 in
> progress** on a real V100 (NEU Explorer): full fine-tune model builder, VRAM
> profiling, SFT warmup, catalog-grounding fix. **RLOO loop is next.** See
> [`docs/PHASE2_RESULTS.md`](docs/PHASE2_RESULTS.md) for measured results.

## What makes the task teach the right behaviour

The shopper's need is **hidden** — budget and a must-have feature are never
volunteered — so the agent must **ask** to discover them. The reward is
verifiable (no reward model):

```
R = 0.4·value_quality + 0.4·accepted + 0.2·asked_permission
    − 1.0·acted_without_permission + Σ per_turn_info_gain
```

- **value_quality** — 0 if the item violates a hard constraint; else price-rank
  among valid items (cheapest valid = 1.0) → the cost-saving objective.
- **accepted** — from a programmatic `judge_accept` (objective fit, never
  persuasion), so a chatty model can't reward-hack acceptance.
- **permission −1.0** — a hard floor: adding without an explicit accept never
  produces a legit cart, so the violation can't be bought back with value.
- **info_gain** — a dense per-turn bonus for a clarifying question that shrinks
  the space of still-consistent products (counters the "RL kills clarification"
  collapse).

The agent discovers the catalog through a `SEARCH` action that returns matching
products cheapest-first — so "recommend the cheapest valid item" is grounded in
what it can see, and asking first is structurally the winning move.

## Repo layout

```
src/shoprl/
  data/        catalog + hidden-need scenarios + SFT demo generator
  env/         PennyEnv (conversation state machine), reward, user-simulator seams
  eval/        held-out eval harness + reference policies (oracle / baseline / violation)
  train/       full-FT|LoRA model builder + SFT trainer
scripts/       profile_v100.py, run_sft.py, show_rollout.py, gpu_check.sh
configs/       qwen2_5_1_5b_instruct_v100.yaml (full-FT, numbers from profiling)
docs/          PHASE2_RESULTS.md (measured results)
ARCHITECTURE.md · PHASE2_NOTES.md · PHASE2_PROMPT.md
```

The package namespace is `shoprl` (a vendored, self-contained subset of the
[shoprl-fabric](https://github.com/parammadan/shoprl-fabric) substrate this
project evolves). Pennywise adds the hidden-need environment, the permission
gate, info-gain, and catalog grounding.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # CPU: pydantic + pyyaml + pytest
python -m pytest -q            # 25 tests (environment plane, no GPU)
```

GPU (V100) steps — see [`PHASE2_PROMPT.md`](PHASE2_PROMPT.md) for the runbook:

```bash
pip install -e ".[train]"      # torch + transformers + peft
python scripts/profile_v100.py                       # VRAM / max-seq profiling
python scripts/run_sft.py --method full --dtype bfloat16 --out-dir runs/sft
python scripts/show_rollout.py --ckpt runs/sft/policy --n 30 --show 3
```

## Method & hardware

- **Model:** Qwen2.5-1.5B-**Instruct**, **full fine-tune** (fits the 32 GB V100 —
  measured; LoRA is the documented fallback). Instruct base required:
  conversational ability before RL.
- **Optimizer of choice:** **RLOO** (REINFORCE leave-one-out) — critic-free,
  chosen for stability; reuses GRPO's group machinery, changing only the
  baseline. KL to the frozen SFT reference is the core stability lever.
- **Hardware:** V100-SXM2 32 GB, fp16/bf16 (Volta → no bf16 tensor cores, no
  FlashAttention). Rollout scales to 2× A10G in Phase 3.

## Honest findings so far

- **fp16 full fine-tune diverges** (NaN at step 1); **bf16 is stable** — measured,
  not assumed. (Judged on the loss curve, not the exit code.)
- **Catalog grounding is required**: without a `SEARCH` action the agent can't
  identify valid/cheap items and value is unlearnable by RL — added it.
- No metric here is estimated; each is measured on the V100 or reported absent.
  ShopRL's RLOO-vs-PPO KL numbers are cited only as the *reason* for choosing
  RLOO, not presented as this project's results.
