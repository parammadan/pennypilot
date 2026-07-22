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

**The learning story** — every arm violation-free, every number held-out:

| Stage | Task success (es-en hard split) |
|---|---|
| Base model (hardness gate) | 12.5% |
| SFT (env-recorded multilingual demos) | 48.4% on the RL-difficulty split (98–100% on the standard split) |
| RLOO, 50 steps | 92.2% |
| **GRPO, equal 400-trajectory budget** | **100.0%** |

The pre-registered result (see the companion docs repo): GRPO's notorious
instability is **regime-dependent** — measured KL 0.013 here (~40× under
threshold) vs 0.58 drift on a saturated task with the same stack. Predictions
were registered before the runs; the one mechanism surprise is documented,
not hidden. Zero-shot WebShop transfer: safety carried perfectly (0
violations), competence partially (60%) — an honest domain-shift datapoint.

**The systems story** — the full SS0–SS13 campaign, each row a
predicted-vs-measured artifact ([`benchmarks/artifacts/`](benchmarks/artifacts/)):

- Rollout generation = **95.8%** of an RL iteration (3 consistent profiles).
- **Waterfall: 93.2 s → 13.4 s per iteration (6.96×)** — batched lockstep
  rollouts (4.98× tokens, 6.4× episodes/min), merge-at-sync vLLM serving
  (2.3× alone; also the fix for Volta's broken LoRA kernels), ckpt-off
  updates (1.48× at batch ≤ 2).
- **Measured-out, with mechanisms:** engine swap alone (1.06×), prefix
  caching (Volta-gated *and* ≤7%/turn at our lengths), packing (8.2% waste),
  prefetch (0.2%), and a Triton kernel deliberately **not** written (the
  profiled ceiling was 0.84% of the iteration).
- Precision: fp16+GradScaler **4.27×** over bf16-emulated, both stable —
  unscaled fp16 diverges at step 1, which is why the scaler isn't optional.

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
2. SFT warmup (multilingual demos, corrections, permission edges) ✅
3. RLOO training (bring-up algorithm) ✅
4. GRPO comparison at equal rollout budget (pre-registered hypotheses) ✅
5. WebShop adapter evaluation on unseen products ✅ · Chromium demo: replay ✅
   live mode ✅ (live *session* with the trained policy: scheduled on demand)
6. Profiling campaign SS0–SS13 ✅ — **14/14 rows measured**, rule held:
   no optimization without a measurement.

Watch it: `python scripts/demo_browser.py --transcript demos/trained_es-en.json --headed`

## Current limitations (honest)

- Scripted user simulator has cosmetic phrasing quirks (a frozen-LLM user
  model slots in behind the same seam later).
- Mid-flow corrections that change ground truth (budget revisions) are
  deferred pending scenario-schema support — not faked.
- "WebShop" evaluation uses the dialect-faithful in-repo backend; the real
  princeton-nlp instance remains a documented backend plug-in.
- vLLM on Volta: LoRA serving and prefix caching are hardware-gated (Triton
  kernels need Ampere) — worked around via merge-at-sync, documented with
  the asserts.
- H2's registered mechanism (multi-epoch reuse) is untested — our GRPO won
  via advantage scaling; the reuse arm is a documented follow-up.

Safety by scope: no real purchases, no real-site scraping, no CAPTCHA/auth
bypass — the storefront is synthetic and the browser is a projection.
