# PennyPilot

> *Multi-turn, multilingual RL shopping agent — prefix-cached rollouts,
> verifiable rewards, permission-gated carts. The training stack is the
> deliverable.*

PennyPilot is a **post-training system** — rollout → verifiable reward → RL
optimization → eval → checkpoint — validated on a shopping assistant that
understands English, Spanish, and code-switched requests, asks clarifying
questions only when they pay off, builds budget-aware plans, picks the
cheapest product satisfying every discovered constraint, and **never touches
the cart without explicit permission**.

The agent is the workload; the engineering target is the **training
infrastructure**: multi-turn RL stability, safety gates that survive
optimization pressure, observability, and profiling-driven rollout efficiency
— all on constrained hardware (1× V100 32 GB on a university SLURM cluster).

## The task (designed to be hard for the right reasons)

The shopper's needs are **hidden**: a budget and 2–3 hard requirements that
are only revealed when the agent asks the right clarifying question — one
reveal per question. Until the last constraint is discovered, the cheapest
visible item is almost always a trap (**measured: 99/100 scenarios** leave a
distractor under partial discovery), so "grab the first result" loses and
disciplined clarification wins. Rewards are computed against catalog ground
truth — no reward model, nothing to hack.

Base-model difficulty is **gated, not assumed**: an un-tuned
Qwen2.5-1.5B-Instruct solves **12.5%** of held-out scenarios (n=64) — enough
signal to bootstrap, enough headroom to matter (acceptance band 10–40%,
measured before any training spend).

## Architecture

The policy emits **structured JSON actions** — never raw browser selectors:

```json
{"action": "ask_user", "question": "What is your total budget?"}
{"action": "search", "query": "family sunscreen SPF 50 under 20 dollars"}
{"action": "select_product", "product_id": "LAP-0014", "reason": "cheapest valid"}
{"action": "request_cart_permission", "items": ["LAP-0014"], "estimated_total": 796.29}
```

Environment adapters translate them: a **synthetic catalog environment**
(training), a WebShop adapter (unseen-product eval, planned), and a Playwright
Chromium mirror (demo/replay **only** — the browser renders decisions already
made and is never the training path).

Other load-bearing pieces:

- **Structured, environment-side dialogue state** — corrections ("actually,
  my budget is $90", "no quiero una umbrella", "I already have sunscreen")
  update state and invalidate stale plans; the model's context window is a
  view of the state, never the storage.
- **Language layer** — deterministic EN/ES/code-switch detection and bilingual
  constraint extraction; constraints survive whatever language they arrive in.
- **Structural permission gate** — `add_to_cart` without an explicit grant for
  that exact SKU cannot produce a cart item, so no reward term can buy back a
  violation. Ambiguous replies are not approval; "don't add anything yet"
  survives further permission requests.
- **Loud loss-mask verification** — policy loss on policy tokens only;
  a reference-exact verifier crashes the run if user/env tokens ever leak into
  the loss (the failure mode where loss improves while the agent gets worse).
- **SFT demos recorded from the environment itself** — demonstrations are env
  transcripts with a build-time correctness guard, not plausible-looking text.

## Measured on the V100 (nothing estimated)

- **Precision:** fp16 + GradScaler **3455 tok/s** vs bf16-emulated 810 tok/s
  (**4.27×**, both numerically stable, peak 7.34 GB) → fp16 pinned. Unscaled
  fp16 diverges at step 1 — measured, which is why the scaler is not optional.
- **Training loop:** end-to-end multi-turn RLOO validated on-cluster — 50
  observed steps, KL to the frozen reference bounded ≤ 0.0035, clarifying and
  permission behaviour preserved throughout (no "RL kills asking" collapse).
- **Memory:** 1.5B full fine-tune fits (18.6 GB fixed footprint; OOM wall
  ~seq 3072 without FlashAttention on Volta); LoRA is the working default for
  engine-coexistence and adapter hot-swap.
- **Task hardness:** base success 12.5% (n=64), violation rate 0, ask rate 1.0
  — the base model asks but can't drive the action loop to a valid cheapest
  pick; exactly the gap SFT+RL should close. Artifacts: [`profiling/gate/`](profiling/gate/).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # CPU: pydantic + pyyaml + pytest
python -m pytest -q          # 88 tests, model-free, <2s
```

GPU (SLURM):

```bash
bash scripts/validation_session.sh        # egress + precision bench + vLLM smoke + hardness gate
python scripts/base_success.py --n 64     # task-hardness measurement alone
python scripts/bench_precision.py         # fp16+GradScaler vs bf16 micro-benchmark
python scripts/vllm_smoke.py              # vLLM-on-Volta candidate smoke (own venv)
```

## Roadmap (gated — each stage must pass before the next)

1. Environment + deterministic oracle + vertical slice ✅
2. SFT warmup (multilingual demos, corrections, permission edges) ← **next**
3. RLOO training (bring-up algorithm)
4. GRPO comparison at equal rollout budget (pre-registered hypotheses)
5. WebShop adapter evaluation on unseen products
6. Visible-Chromium demo: live execution + deterministic replay
7. Profiling campaign: baseline trace → vLLM rollout → prefix caching →
   concurrency → memory → packing → (Triton kernel only if the re-profile
   shows the hotspot). Rule: **no optimization without a measurement.**

## Current limitations (honest)

- No trained-model results yet — the numbers above are platform and
  task-design measurements; SFT/RL results land as they are produced.
- Scripted user simulator has cosmetic phrasing quirks (a frozen-LLM user
  model slots in behind the same seam later).
- Mid-flow corrections that change ground truth (budget revisions) are
  deferred pending scenario-schema support — not faked.
- WebShop adapter and browser demo not started; vLLM version pin in progress.

Safety by scope: no real purchases, no real-site scraping, no CAPTCHA/auth
bypass — the storefront is synthetic and the browser is a projection.
