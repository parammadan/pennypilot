# PennyPilot (evolved from Pennywise)

> *Multi-turn, multilingual RL shopping agent — prefix-cached rollouts,
> verifiable rewards, permission-gated carts. The training stack is the
> deliverable.*

A multi-turn RL **post-training system** — rollout → verifiable reward → RLOO
optimization → eval → checkpoint — whose v2 workload is a **multilingual
(English / Spanish / Spanglish), budget-aware shopping agent** that asks
clarifying questions, keeps structured environment-side dialogue state, picks
the cheapest item satisfying every discovered constraint, and **never touches
the cart without explicit permission**.

**The training system is the deliverable; the agent is the workload.** Target:
training-infrastructure engineering (stability, observability, efficiency) on
constrained hardware (1× V100 32 GB, university SLURM).

> **Status (2026-07-21):** v1 (Pennywise) complete — SFT + grounded env +
> stable 50-step RLOO on the V100 ([docs/PHASE2_RESULTS.md](docs/PHASE2_RESULTS.md)).
> v2 (PennyPilot) in progress — Stage 1 vertical slice green (88 CPU tests),
> validation session measured, **SFT next**. Design notes + dev log live in the
> private companion docs repo (`shoprl-fabric-docs`).

## How v2 evolved from what was here (reuse, not rewrite)

| Kept from Pennywise/shoprl | v2 adds on top |
|---|---|
| Hidden-need scenarios + tunable valid-set knob | **Hard mode**: 2–3 simultaneous must-haves, one reveal per ask — 99/100 scenarios leave a *distractor* (cheapest-visible is invalid) under partial discovery |
| Permission gate (structural −1.0) + info-gain reward + `SEARCH` grounding | **Structured JSON abstract actions** (`ask_user/search/inspect/select/request_cart_permission/add_to_cart`) behind a `ShoppingEnvironment` adapter interface |
| User-simulator seams (programmatic `judge_accept`; words swappable) | **Multilingual user** (EN / ES / code-switched) + language layer: detection, bilingual constraint extraction, optional English gloss |
| SFT masking (assistant-tokens-only, prefix-difference) | **Loud loss-mask verifier** (reference-exact + leak checks) wired into training; model-free tests |
| RLOO trainer, eval-harness pattern, dashboard/alerts, configs | **Structured `DialogueState`** (corrections update state and invalidate stale plans), v2 eval harness (per-language, held-out), env-recorded SFT demos with build-time correctness guard |

## Measured so far (V100, nothing estimated)

- **v1:** full-FT fits (18.6 GB fixed, OOM ~seq 3072); grounding unlock
  (value 0.34 → 1.0); 50-step RLOO stable (KL ≤ 0.0035, no regression).
- **v2 validation session (2026-07-21):**
  - **Precision:** fp16+GradScaler **4.27×** faster than bf16-emulated
    (3455 vs 810 tok/s), both stable → **fp16 pinned** (v1's "fp16 diverges"
    was unscaled fp16).
  - **Scenario Hardness Gate: PASS** — base-model success **12.5%** (n=64,
    band 10–40%): violation rate 0, ask rate 1.0, dominant failure = invalid
    actions / max-turns, i.e. real headroom for SFT+RL.
  - Artifacts: [`profiling/gate/`](profiling/gate/).

## Safety invariants (enforced by structure, not convention)

- An `add_to_cart` without an explicit grant **for that exact SKU** never
  produces a cart item (`DialogueState.add_to_cart` refuses; reward −1.0 with
  nothing to offset it). Ambiguous replies are not approval; a user hold
  ("don't add anything yet") survives further permission requests.
- Savings are only ever claimed against a defined baseline.
- The visible-browser demo (Stage 5, Playwright) renders **already-decided**
  structured actions — the browser is never the training path and never sees
  real payments/orders.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # CPU: pydantic + pyyaml + pytest
python -m pytest -q          # 88 tests, model-free, <2s
```

GPU (Explorer V100; see the docs repo `RUN_ON_SLURM.md`):

```bash
bash scripts/validation_session.sh        # egress + precision + vLLM + hardness gate
python scripts/base_success.py --n 64     # hardness gate alone
python scripts/bench_precision.py         # fp16+GradScaler vs bf16 micro-bench
python scripts/vllm_smoke.py              # per-candidate vLLM smoke (own venv)
```

v1 paths still work: `scripts/run_sft.py`, `scripts/run_rloo.py`,
`scripts/show_rollout.py`, `scripts/profile_v100.py`.

## Roadmap (gated; each stage passes before the next)

1. ~~Env + oracle + vertical slice~~ ✅  2. **SFT warmup** (next: `run_sft_v2`,
fp16+GradScaler, LoRA) 3. RLOO bring-up 4. GRPO comparison at equal rollout
budget (pre-registered) 5. WebShop adapter eval 6. Chromium demo (live +
deterministic replay) 7. Profiling-driven optimization campaign (SS0–SS13:
no optimization without a measurement).

## Known limitations (current, honest)

- Trained-model results for v2 do not exist yet — the only v2 numbers are the
  validation-session measurements above.
- The scripted multilingual simulator emits a cosmetic trailing user line
  after `add_to_cart` (inherited v1 quirk; replaced with the frozen-LLM
  simulator later).
- Deeper correction demos (mid-flow budget/party-size changes) need evolving
  ground truth in the scenario schema — deliberately deferred, not faked.
- vLLM-on-V100 pin pending its smoke run; WebShop + browser demo not started.

The package namespace stays `shoprl` (vendored substrate of
[shoprl-fabric](https://github.com/parammadan/shoprl-fabric)); Pennywise → 
PennyPilot is the project evolving on top of it.
