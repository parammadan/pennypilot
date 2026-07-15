# Phase 2 readiness notes

Concrete blockers found during Phase 1, written down so they are not
rediscovered under GPU time pressure. Phase 1 (environment plane) is built and
tested on CPU; Phase 2 is the training plane on the V100.

## Ordering rule (do not violate)

**Profile the max viable sequence length FIRST. Every config number depends on
it.** Do not set `max_new_tokens`, `num_samples`, `max_turns`, or batch size
until the measured sequence-length/VRAM curve is in hand. Setting them by guess
wastes GPU time on OOM trial-and-error.

## Blockers

### B1 — Trainer is LoRA-only; full fine-tune is the target
`src/shoprl/grpo/trainer.py::build_policy` **always** wraps the model in LoRA
(`get_peft_model`) and treats the frozen base as the reference. Pennywise's
target is **Qwen-1.5B instruct, full fine-tune** (a 32GB V100 allows it). Needs:
- a non-LoRA path in `build_policy` (train all params; a separate reference —
  either a second frozen copy or periodic reference refresh, since with full-FT
  the base is no longer a free no-op reference the way a disabled LoRA adapter
  is);
- AdamW optimizer state for **all** 1.5B params (moments in fp32) — a much
  larger, different memory footprint than a tiny adapter.
- **Measured fallback, not a failure:** if Step-3 profiling shows full-FT does
  not fit the multi-turn sequences on Volta, **switch to LoRA**. That is the
  correct measured outcome. LoRA + gradient checkpointing is the documented
  fallback (and 3B LoRA is the model fallback if 1.5B plateaus).

### B2 — Config defaults are stale (0.6B / single-turn / M1)
`src/shoprl/config.py` defaults: `model.name=Qwen/Qwen3-0.6B`,
`rollout.max_new_tokens=128`, `rollout.num_samples=4`,
`training.prompts_per_step=1`. All sized for a 0.6B single-turn model on an 8GB
M1. For Phase 2:
- model → a **Qwen-1.5B _instruct_** variant (instruct base required: the model
  needs conversational ability before RL);
- `num_samples` → RLOO group size (target k=8), set after profiling;
- `max_new_tokens` → per agent turn, set after profiling.
- The `configs/qwen25_3b_v100.yaml` in the repo is the **wrong target** (3B +
  LoRA, single-turn reward weights). Add a correct 1.5B full-FT config with the
  numbers left as profiling placeholders until Step 3.

### B3 — `max_turns=8` sizing (env)
`PennyEnv(max_turns=8)` / `ShopEnv(max_turns=8)` is a single-turn-era default. In
Pennywise it directly bounds the **conversation length → the training sequence
length → attention memory**. It must be chosen from the profiling curve, not
left at 8 by inertia. A full good trajectory needs ~5-6 turns (2 asks + recommend
+ permission + add, plus recovery); pick `max_turns` with headroom above that but
under the measured sequence ceiling.

### B4 — V100 memory chain (the one that can force LoRA)
Volta (V100, compute capability 7.0) has **no bf16 tensor cores** (fp16 only) and
**no FlashAttention**. Without FlashAttention, attention memory is **quadratic in
sequence length**. Multi-turn dialogues are far longer than the single-turn
prompts the old §12 profile measured, so:
```
no FlashAttention  +  longer multi-turn sequences
      → quadratic attention memory in seq len
      → measure max viable sequence length FIRST
      → it may cap max_turns, or force LoRA even on a 32GB card
```
The old ARCHITECTURE.md §12 numbers (3B, single-turn, LoRA) **do not transfer**.
No VRAM/throughput number is claimed for Pennywise until measured on the V100.

## Run discipline (Phase 2 execution)

GPU time is available, so the plan is **many short observed runs, not one long
unobserved batch**. Concretely:
- Steps 1-4 this round: (1) full-FT trainer path, (2) 1.5B config with profiling
  placeholders, (3) **VRAM/seq-length profiling**, (4) bounded SFT warmup on the
  ~1k demos (`data/sft.py`) — confirm it produces valid-grammar output. **STOP
  after Step 4.**
- RLOO loop is the **next** round: **50-step observed iterations**. Confirm
  reward moves and KL stays controlled before scaling to hundreds of steps. Do
  not launch a big blind training run.
- Every run: checkpoint to S3, `metrics.jsonl` per step (KL final + max, reward,
  grad-norm, entropy, GPU util), billing alarm on, EBS delete-on-terminate.

## Environment note

This repo is developed on a local Mac (CPU, Phase 1). Phase 2 runs on the NEU RC
SLURM cluster (login `madan.pa`); the GPU node has the conda env + CUDA module
(see `env.sh`). The V100 partition name and exact GRES string are NEU-specific
and must be discovered on the login node (`sinfo`, `scontrol show partition`),
not assumed.

## Phase 2b (optional, NOT built) — harder scenarios for measurable RLOO learning

Post-SFT the task is easy: the `SEARCH` list is price-sorted, so "recommend the
cheapest valid" reduces to "pick item #1", which SFT already solves (greedy value
1.0). To make RLOO show *measurable learning* rather than only stability, add
difficulty so cheapest-valid ≠ list position #1:
- multiple simultaneous must-haves (e.g. brand AND min-battery AND max-weight);
- distractor items (valid-looking but violating a hidden constraint) in the list;
- a value function where the best pick trades off price against a soft preference,
  so #1-by-price isn't automatically optimal.
This is an **optional upgrade to the RL story**, not required for the
training-infrastructure narrative (which is carried by the stable, observable,
no-regression RLOO loop already demonstrated).
