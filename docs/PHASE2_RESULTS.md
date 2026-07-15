# Phase 2 results — measured on NEU Explorer (V100)

Durable record of what was measured and decided in Phase 2, Steps 1–4. Numbers
here are measured on the cluster, not estimated. Companion to `PHASE2_NOTES.md`
(the blockers/plan) and `ARCHITECTURE.md` (the design).

_Last updated: 2026-07-15._

## Environment & access

- **Cluster:** NEU **Explorer** (`login.explorer.northeastern.edu`), SLURM, login `madan.pa`.
- **Access from the dev Mac:** key-based SSH (`~/.ssh/id_ed25519_explorer`) with a
  persistent control-master (`~/.ssh/config` Host `explorer`). Password auth is
  impossible from the sandbox (no TTY); the public key was installed once into
  `~/.ssh/authorized_keys` on Explorer. **No Duo on SSH** (password-only), so key
  auth alone connects.
- **conda env `shoprl`** (`/home/madan.pa/.conda/envs/shoprl`, Python 3.12.13):
  torch **2.5.1+cu121**, transformers **5.13.1**, peft **0.19.1**. `pip install -e .`
  adds the pennywise package (pydantic/pyyaml). All 24 Phase-1 tests pass here.
- **HF cache / storage:** `/scratch/madan.pa` (1.1 PB free); `HF_HOME` set in
  `env.sh`. Compute nodes are offline → pre-download models on the login node,
  run jobs with `HF_HUB_OFFLINE=1`.

## SLURM facts (discovered, not assumed)

| Partition | GPU GRES (V100) | Walltime |
|---|---|---|
| `gpu` | `gpu:v100-sxm2:N` (32 GB SXM2), also `v100-pcie` | 8:00:00 |
| `short` | `gpu:v100-sxm2:4` on d1027 | 2-00:00:00 |
| `gpu-short` | `gpu:v100-sxm2` | 2:00:00 |

- **Job pattern used:** `srun --partition=gpu --gres=gpu:v100-sxm2:1 --cpus-per-task=8 --mem=64G --time=... <cmd>`.
- CUDA modules available: `cuda/12.1.1 12.3.0 12.8.0 13.2.0` (`env.sh` uses 12.3.0).

## V100 confirmation

`Tesla V100-SXM2-32GB` · driver 545.23.08 · **compute capability 7.0 (Volta)** ·
32768 MiB (~34 GB decimal). Volta ⇒ **no bf16 tensor cores** (fp16 for
throughput; `torch.cuda.is_bf16_supported()` returns True but that's *emulated*,
no speedup) and **no FlashAttention** (attention memory is quadratic in seq len).

## Step 1 — full-FT model builder

`src/shoprl/train/build.py::build_model(method="full"|"lora")`, fp16. Full-FT
loads a **separate frozen copy as the KL reference** (a moved full-FT policy
can't double as its own reference the way a disabled LoRA adapter can). Decoupled
from the stale single-turn `grpo/trainer.py`.

## Step 3 — VRAM / sequence-length profiling (the gate)

`scripts/profile_v100.py` → `profiling/profile_v100.json`. Realistic training
step (ref fwd + policy fwd → loss → backward → optimizer.step), 30 GB cap:

| config | full-FT peak | LoRA peak |
|---|---|---|
| fixed footprint (weights+optim+ctx) | **18.57 GB** | 3.67 GB |
| batch 1 × seq 512 | 18.73 | 7.07 |
| batch 1 × seq 1024 | 19.40 | 11.52 |
| batch 1 × seq 1536 | 23.90 | 16.56 |
| batch 1 × seq 2048 | 29.17 | 22.39 |
| batch 4 × seq 512 | 25.02 | 18.21 |
| batch 1 × seq 3072 | **OOM** | **OOM** |

**Findings**
- **Activation memory dominates at long sequences and is identical for full-FT
  and LoRA** (same forward graph, quadratic without FlashAttention). LoRA only
  saves the ~15 GB fixed footprint, so both hit the same wall (~seq 3072 @ batch
  1). LoRA buys headroom only at lower seq/batch, not a higher ceiling.
- **Decision (measured): full fine-tune, LoRA not needed.** Demo dialogues
  tokenize to **max 221 tokens** (mean 161, p99 215) under the Qwen chat
  template — far below the OOM wall — so full-FT fits multi-turn comfortably.

## Step 2 — 1.5B config (from measurement)

`configs/qwen2_5_1_5b_instruct_v100.yaml`: `Qwen/Qwen2.5-1.5B-Instruct`,
`finetune: full`, `dtype: float16`, `max_len: 512` (2.3× the 221-token max),
`max_turns: 12`, `num_samples: 8`.
**RLOO caveat:** full-FT cannot forward all 8 group trajectories at once (would
exceed the cap) → the RLOO loop must micro-batch / gradient-accumulate over the
group of 8 (next round).

## Step 4 — bounded SFT warmup (close-out)

Config: full-FT, 1000 demos, batch 4, 1 epoch (250 steps), **lr 1e-5, grad-clip
1.0, no warmup**, `max_len 512`, `max_turns 12`.

**1. Precision — fp16 diverges, bf16 is stable (measured).**
Pure **fp16** full-FT went **NaN at step 1** (gradient overflow, no loss
scaling) and collapsed to `!!!!!` garbage — despite a clean exit code (judge on
numbers, not exit status). **bf16** (emulated on Volta) held finite. fp32-master
would be stable but needs ~30 GB fixed → doesn't fit. So **bf16 is the full-FT
precision on this hardware.**

**2. Loss trajectory (bf16) — finite + decreasing:**
`3.31 (s1) → 0.48 (s10) → 0.17 (s20) → 0.14 (s50) → ~0.09–0.13 (s120–250)`; NaN
early-stop never fired.

**3. Grammar (well-formed action rate), held-out:** pre-SFT **0.0 (0/240)** →
post-SFT **0.947 (358/378)**. Cold-start taught the grammar.

**4. Masking — verified correct:** supervised tokens per batch = 168–206 (agent
turns only); decoded supervised span = exactly the agent actions
(`ASK[...]/RECOMMEND[...]/ASK_PERMISSION/ADD_TO_CART[...]`), no user/prompt tokens.

**5. Behaviour (30 held-out, `show_rollout.py`) — ritual YES, value NO:**
`ask_rate 1.000 · violation_rate 0.000 · accept_rate 0.400 · mean value_quality
0.339 · cheapest-valid hit 0.267`. It always asks and never violates permission
(the behaviours RL must preserve), and sometimes nails the cheapest valid item
(value 1.0) — but ~60% of recommendations are invalid.

**6. Duration / bf16:** both SFT runs `sacct` Elapsed **00:10:07**. This does NOT
cleanly show the expected bf16-emulation slowdown because the comparison is
**confounded** — the fp16 run generated with the KV cache OFF (slow), the bf16
run with it ON (fast) — so the training-speed delta is masked. bf16 on Volta
(cap 7.0) has no tensor cores and runs emulated, so it *is* slower per-op; a
clean fp16-vs-bf16 fwd/bwd micro-benchmark is still TODO. bf16 did NOT OOM,
confirming it ran at 2-byte width (not a silent fp32 fallback, which wouldn't
fit).

### KEY FINDING — catalog grounding is a prerequisite for RLOO
`value_quality` is weak (0.339, *below* the random-valid baseline 0.547) because
**the agent never sees the catalog** — `PennyEnv` exposes only the conversation,
so the model cannot identify which SKU is valid (right brand, under budget), let
alone the cheapest. It learned to emit plausible SKU tokens, not grounded
selection. **RL cannot fix this**: with no catalog in context there is no
learnable path from action to "cheapest valid", so an RLOO value reward would
only invite reward-hacking on memorized SKUs.

**Prerequisite before RLOO:** add candidate grounding to the env — after the
agent discovers the constraints, present the matching catalog items (SKU +
price), or add a `SEARCH`/`FILTER` action that returns them (cf. shoprl's
retrieve→shortlist, and Phase-4's Chromium browsing). Then "recommend cheapest
valid" is groundable and RLOO can sharpen it. The ritual (ask/permission) is
already present and must be preserved through RL.

### Grounding fix — CONFIRMED (SEARCH action added)

Added a `SEARCH` action (`env.search_catalog` / `format_candidates`): once the
agent has discovered the constraints, it returns the matching products
cheapest-first as an observation; searching before asking returns the globally
cheapest (usually invalid), so asking-first is still required. SFT demos now
teach ask → discover → SEARCH → RECOMMEND[first/cheapest] → permission → add.

Grounded SFT (bf16, 1000 demos, 250 steps, `max_len 640`) then re-evaluated on
the **same 30 held-out scenarios**:

| metric | pre-grounding | grounded |
|---|---|---|
| ask_rate | 1.000 | **1.000** |
| violation_rate | 0.000 | **0.000** |
| accept_rate | 0.40 | **1.000** |
| mean value_quality | 0.339 | **1.000** |
| cheapest-valid hit | 0.267 | **1.000** |

It learned a **general procedure** (search → read the list → pick the cheapest
shown), not memorized SKUs — hence 100% on unseen hidden needs.

### Honest implication for RLOO
SFT + grounding **near-saturates this synthetic task** (value/accept/cheapest all
1.0, zero violations, always asks). So RLOO's value here is **not** squeezing a
saturated metric — it is (a) demonstrating the post-training *pipeline* (the
deliverable: multi-turn rollout → verifiable reward → leave-one-out advantage →
KL-controlled update → checkpoint, stable and observable), and (b) guarding
against regression — confirming RL does **not** destroy the clarifying/permission
behaviour (the "RL kills asking" collapse) and that KL stays controlled (the
RLOO-vs-PPO stability story). Harder/curriculum scenarios can widen the SFT gap
later if a larger reward delta is wanted.

## Bugs caught (before they cost GPU time)

1. `.gitignore` bare `env/` was ignoring the entire `src/shoprl/env/` novelty
   dir → anchored the venv patterns to repo root.
2. `apply_chat_template` returns a **`BatchEncoding` (UserDict, not a `dict`
   subclass)** → `isinstance(_, dict)` was False and the encoder read length-2,
   which would have silently broken SFT loss masking. Fixed with
   `return_dict=True` + explicit `["input_ids"]`.

## Next round — RLOO

Short **50-step observed** iterations (not a blind batch). Adapt the multi-turn
rollout vs the simulator, RLOO leave-one-out advantage, KL-to-SFT-reference;
micro-batch the group of 8. Confirm reward moves and KL stays controlled before
scaling.

## Phase 2c — RLOO loop bring-up

Multi-turn RLOO (`src/shoprl/train/rloo.py`, `scripts/run_rloo.py`): k rollouts
per prompt vs the simulator, leave-one-out advantage (A_i = r_i − mean(r_−i), no
std norm), micro-batched to batch=1 for the policy-gradient backward (full-FT
can't forward k long sequences at once), REINFORCE loss −A·logp̄ + β·k3-KL vs the
frozen SFT reference. Policy + reference both init from the grounded SFT
checkpoint.

**10-step smoke:** ran end-to-end (reward computes, KL logs, no crash), KL
~0.0013, ask 1.0 / violation 0.0 preserved. Surfaced a gradient-scale bug —
`grad_norm` up to ~668 because the PG summed logp over ~40 agent tokens, so
clipping (max_grad_norm 1.0) masked the real update. Fixed: normalize PG + KL
**per token** (mean). grad_norm dropped to O(0.1–5).

**50-step observed run** (k=8, β=0.04, lr 1e-6, bf16):

| step | reward | KL | value | ask | viol | grad_norm |
|---|---|---|---|---|---|---|
| 0  | +0.885 | 0.00000 | 0.75 | 1.0 | 0.0 | 2.70 |
| 10 | +1.104 | 0.00135 | 0.88 | 1.0 | 0.0 | 4.56 |
| 20 | +1.100 | 0.00098 | 0.88 | 1.0 | 0.0 | 2.05 |
| 30 | +1.097 | 0.00201 | 0.88 | 1.0 | 0.0 | 2.59 |
| 40 | +1.028 | 0.00343 | 0.88 | 1.0 | 0.0 | 3.41 |
| 49 | +1.153 | 0.00348 | 1.00 | 1.0 | 0.0 | 0.28 |

- **KL controlled** — a slow bounded climb 0 → ~0.0035 (the RLOO-stable regime;
  no PPO-style blowup). This is the core stability result.
- **No regression / no collapse** — ask 1.0 and violation 0.0 through all 50
  steps; post-RLOO held-out eval (30 scenarios) = value 1.0 / accept 1.0 /
  cheapest-hit 1.0 / ask 1.0 / violation 0.0, identical to SFT.
- **Near-saturation shows up as sparse signal** — most steps have adv≈0 (the 8
  rollouts are identical on easy scenarios → zero leave-one-out advantage), so
  the pipeline is demonstrated to be stable rather than shown climbing a big
  reward gap. Harder/curriculum scenarios would widen that gap.

**Bottleneck (motivates Phase 3):** each step is dominated by rollout generation
on *emulated* bf16 (~30 s/step for k=8). Phase 3's vLLM data-parallel rollout on
the 2× A10Gs is exactly the fix.

### Phase 2 status: COMPLETE
Pipeline validated end-to-end on the V100 — SFT warmup → grounded env → RLOO
(rollout → verifiable reward → leave-one-out advantage → KL-controlled update →
checkpoint), stable and observable. Next: Phase 3 (efficiency + observability;
A10G rollout throughput, dashboard, S3 checkpoint bridge, checkpoint/resume).

## Did RLOO improve value_quality, or hold it? (plain answer)

**SFT already reached value_quality ≈ 1.0; RLOO held it stable — this is a
stability result, not a learning gain.**

- **Greedy eval** (`show_rollout`, deterministic): grounded **SFT = 1.0**;
  **post-RLOO = 1.0** on the same 30 held-out scenarios → RLOO did **not**
  improve greedy behaviour (it was already solved) and, importantly, did **not
  regress** it.
- **RLOO training-step value_mean** (sampled rollouts, T=1.0): 0.750 (step 0) →
  0.875 (step 10) → 1.000 (step 25–49). This rise is the policy *sharpening*
  under sampling (its T=1.0 samples converge onto the already-correct greedy
  path), not new capability — greedy was 1.0 the whole time.

Honest headline: **stable multi-turn RLOO — KL controlled (~0.0035), clarifying
+ permission behaviour preserved, no regression.** A legitimate training-infra
result; NOT a dramatic behaviour improvement (the numbers don't show one).

## Phase 2 — honest summary

- **Full fine-tune fits** the V100 32 GB (profiled: fixed 18.57 GB, OOM wall
  ~seq 3072; short dialogues ≪ that). LoRA not needed.
- **fp16 full-FT diverges to NaN; bf16 is stable** (fp32-master doesn't fit).
  bf16 on Volta is *emulated* (no tensor cores) → slower per-op; accepted for
  stability. (Clean fp16-vs-bf16 throughput micro-benchmark still TODO — the two
  SFT runs were confounded by a KV-cache change.)
- **Catalog grounding was the real unlock**: without a `SEARCH` action the agent
  couldn't identify valid/cheap items (value 0.339, below the 0.55 random
  baseline); with it, value_quality 0.339 → **1.0** (accept + cheapest-hit → 1.0).
- **Stable RLOO**: multi-turn rollout → verifiable reward → leave-one-out
  advantage → KL-controlled update → checkpoint. KL a bounded 0 → ~0.0035,
  ask 1.0 / violation 0.0 through 50 steps, no regression.
- **Honest caveat**: post-SFT the task is easy ("read the price-sorted SEARCH
  list, pick the cheapest valid = item #1"), so RLOO's headline is **stability +
  no-regression**, not dramatic learning. See PHASE2_NOTES "Phase 2b" for the
  harder-scenario upgrade that would make RLOO show measurable learning.

## Artifacts & notes

- **RLOO rollout transcripts** (30 held-out, raw dialogues): `results/rloo50_transcripts.jsonl`.
- **Simulator note (fixed in Phase 4):** the `ScriptedConversation` generic
  fallback emits an odd trailing user line ("Could you help me find the right
  one?") after the agent already recommends/adds. Cosmetic — it does not affect
  the programmatic reward/judge — and is replaced when the frozen-LLM simulator
  lands in Phase 4.
