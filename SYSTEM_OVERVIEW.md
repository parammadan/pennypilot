# PennyPilot + PennyData — complete system overview

> Self-contained description of everything built in this project. Written so
> that a person or an LLM with no prior context can understand the system,
> its architecture, its measured results, and where every piece lives.

## 0. One-paragraph summary

A single engineer built, on a MacBook (M1, 8GB) plus free university GPU
allocation (single NVIDIA V100s on a Slurm cluster), a **complete miniature
of Amazon's "shopping AI + data platform" stack**: (1) a multilingual,
multi-turn, permission-gated shopping agent trained with a full RL
post-training pipeline (SFT → RLOO), (2) a browser storefront ("PennyMart")
where humans converse with the live agent, (3) an event-streaming data
platform ("PennyData") that captures every conversation turn and every UI
tick through Apache Kafka into analytics + S3 cold storage, and (4) a
behavioral-intelligence layer (defined metrics, funnels, cohorts, ranked
slices, anomaly detection, alerting, automated analyst reports) that turns
that traffic into findings and back into validated training datasets — the
full "learn from customer behavior" flywheel: store → platform → training
data → better model → store. Every scientific claim was pre-registered
before its run; every failure was root-caused and documented.

## 1. Repos

- **`pennywise-v100-infra`** (private, GitHub `parammadan/pennywise-v100-infra`)
  — all code: agent, training, evals, storefront, platform. Package namespace
  `shoprl`, `src/` layout. ~152 pytest tests, all CPU-runnable.
- **`shoprl-fabric-docs`** — the lab notebook: HYPOTHESES.md (pre-registered
  predictions + verdicts), PROGRESS.md (dated, append-only), RESULTS.md,
  notes/CHALLENGES.md (33 root-caused failures), DECISIONS, STAGE6_BENCHMARKS.

## 2. The agent (the workload)

**Task:** a cost-saving shopping assistant for a synthetic laptop store.
Multi-turn dialogue in English/Spanish/Spanglish. The shopper has hidden
needs (budget + must-have constraints) revealed only when asked. The agent
must: ask to discover constraints → SEARCH the catalog → recommend the
CHEAPEST valid item → request explicit permission → add to cart. A
**structural permission gate** makes un-permitted cart adds impossible to
reward (−1.0, unrecoverable): the safety invariant, held at 0.000 violations
across every eval ever run.

**Actions** are single-line JSON (`ask_user`, `search`, `select_product`,
`request_cart_permission`, `add_to_cart`, `inspect_product`), optionally
preceded by natural-language prose ("chat + actions" format); the parser
extracts the JSON, so the gate is format-independent.

**Training pipeline** (`src/shoprl/train/`, `scripts/run_sft_v2.py`,
`scripts/run_rl_v2.py`):
- SFT: LoRA, fp16 + GradScaler (measured 4.27× faster than emulated bf16 on
  Volta), demos are RECORDED from the real environment with build-time
  correctness asserts, loss masked to agent turns (mask verified
  reference-exact).
- RL: RLOO (critic-free, leave-one-out advantage, k=8 rollouts/prompt,
  k3-KL to a frozen SFT reference). `--shared-ref` puts policy+reference as
  two LoRA adapters on ONE frozen base so 7B RL fits a 32GB V100.
- Thin algorithm interface (`train/algo.py`): RLOO/GRPO/GRPO-no-std differ
  only in the advantage rule.

**Models:** Qwen2.5-1.5B-Instruct (the original deliverable line) and
Qwen2.5-7B-Instruct (the "7B chapter"). All checkpoints on cluster scratch
(`/scratch/madan.pa/pennypilot/`).

## 3. Headline scientific results (all pre-registered, docs repo)

- **H1** (GRPO stability is regime-dependent): CONFIRMED — GRPO KL max 0.013
  in a healthy-variance regime vs 0.582–0.988 blowups in the saturated v1
  regime.
- **H2** (sample efficiency at equal rollout budget): SFT 0.484 → RLOO 0.922
  → GRPO 1.000 task success (n=128, equal 400-trajectory budgets);
  mechanism caveat documented (advantage scaling, not reuse).
- **H3** (catastrophic forgetting is rehearsal-mitigable, 7B):
  CONFIRMED on every clause. Specialist arm: retention 12.5%→10.0%, shopping
  0.820. **Rehearsal arm (25% self-distilled general chat in SFT): retention
  97.5%, shopping 0.859** — rehearsal cost NOTHING (beat the specialist) and
  the SFT-stage tax (0.31 vs 0.62) was erased by identical RL.
- **The B-loop** (live user found 2 bugs → data-recipe fix): B2 failed
  (demos trained an unreachable double-user-block context — position/format
  fidelity lesson), B3 fixed it (**12/12 redirect rate**, retention 97.5%,
  shopping 0.906, violations 0), B4 ablation showed explicit notice-training
  was unnecessary and mildly harmful vs B3's generalization of one crisp
  rule ("[store notice:] → explain + redirect"). ~12 GPU-h bug→ship; 2-min
  SFT probes fail-fasted the dead ends.
- **Stage 6 systems campaign** (14 benchmarks, predictions before every
  measurement): rollout = 95.8% of RL iteration; engine swap alone ≈ nothing
  (1.06×); **batched lockstep rollouts = 5–6.4×**; merged-weights vLLM =
  2.3× (and the Volta LoRA workaround); prefix caching irrelevant at these
  lengths; waterfall 93.2s→13.4s = 6.96×. Deliberately NOT built: a custom
  kernel (profiled at 0.84% — the profile is the justification).
- **WebShop zero-shot transfer:** purchase 0.60, violations 0.00 — safety
  transfers fully, competence partially.

**Deployed demo model:** `rl7b_B3` = 7B + rehearsal SFT + 50-step shared-ref
RLOO + the deterministic notice layer. Chats like a general assistant AND
shops AND refuses out-of-scope requests correctly.

## 4. PennyMart (the storefront, `src/shoprl/env/browser_demo.py` +
`scripts/demo_human.py`)

Amazon-styled single-page store rendered from the synthetic catalog:
search (functional, filters the grid), product cards (deterministic stylized
SVG art, ratings, price typography), product-detail modal, **cart drawer**
with subtotal, add-to-cart toast, permission modal whose Approve/"Not yet"
buttons are the human's real authority, demo-hint ribbon (the shopper's
secret brief). Chat panel: assistant named **Penny** with the serving model
tag beside her name, avatars, timestamps, entrance animations, **typing
indicator** while the model generates, 👍/👎 on every agent reply.
`demo_human.py` drives it via Playwright against a GPU policy server
(`scripts/serve_policy.py`, stdlib HTTP, measures TTFT/ITL/MFU;
`serve_duo.sh` serves 1.5B + 7B on one V100 for A/B). Every episode
self-captures (per-turn screenshots + transcript.json) and can stream events
via `--platform-url` (HTTP) or `--kafka`.

## 5. PennyData (the platform, `src/shoprl/platform/`)

**Event model** (`events.py`): `episode_start` (label = model/cohort, brief,
policy), `turn` (agent text, observation, note, latency_ms), `feedback`
(👍/👎 per turn), `ui` (click/hover/input/modal/impression ticks with
browser event-time), `episode_end` (verdict, violation, cart). Pydantic-
validated.

**Transport:** Apache Kafka (local KRaft), topic `pennymart.events`,
4 partitions, events KEYED BY session_id (per-conversation ordering),
acks=all; consumer commits after apply; the store is idempotent per
(session, kind, i) → effectively-once views. Consumer groups scale
horizontally (`--consumers N` splits partitions). **Replay** = fresh group
id (offsets belong to the group; the log is the truth). HTTP ingest also
supported.

**Storage:** `store.py` — events.jsonl (append-only source of truth) +
derived SQLite (episodes/turns/ui_events/events), rebuildable by replay.
**Cold path:** `s3_sink.py` — an independent consumer group archives to
S3 as Hive `date=`-partitioned JSONL parts (commit-after-durable-put);
99k+ events archived. **Analyst workbench:** `scripts/lake.py` — DuckDB SQL
directly over the S3 lake (views: raw/episodes/turns/ui; canned funnel/
cohort/slice reports). AWS: account 771965334314, bucket
`pennydata-771965334314-us-east-2`, scoped IAM user, zero-spend budget
guard, $100 credits.

**Behavioral intelligence** (`behavior.py`) — the discipline is the point:
- `session_features`: sessionization (explicit IDs), reformulation
  (Jaccard≥0.6 on consecutive user msgs), repeat-question (exact normalized
  dupe), turn buckets, per-session clickstream counts, latencies.
- `metrics()`: every rate ships with NUMERATOR + DENOMINATOR + eligible
  population (abandonment; cart-rate-after-search; reformulation; repeat;
  invalid-action; violation; **recommendation CTR** [impression-eligible
  denominator]; hover→click; median session duration; agent latency p50).
- `slice_report()`: 1- and 2-dim slices, min_support=30 (low-n suppressed,
  counted), tautology exclusion (carted can't "explain" abandonment),
  universal-value pruning, ranked |deviation|×share, output labeled
  "hypotheses, not causes".
- `funnel_by()`: engaged→searched→selected→permission→carted with
  stage-to-stage conversion, per cohort.
- `timeseries()` + `detect_anomalies()`: median/MAD robust shift detection;
  low-n buckets never flagged.
- `alerts()`: CRITICAL (permission violation, data-quality flags) /
  WARNING (invalid-rate spike, abandonment shift, latency) — the emergency
  channel.
- `data_quality()`: missing labels, never-ended episodes, orphan turns,
  duplicates — "validate the data before blaming the model" gates every
  report.
- `report.py` / `scripts/analyze.py`: the **automated analyst** — runs the
  whole investigation (quality gate → metrics → funnel → cohorts → ranked
  adverse slices with evidence sessions → hypotheses with caveats) and
  writes markdown+JSON. Proven: discovered a deliberately planted model-v14
  regression in synthetic cohorts.
- `export.py`: filtered episodes → SFT chat-format sequences + validation
  report; violations excluded by default; thumbs-down turns surfaced as
  hard-example candidates. This closes the flywheel — the export format is
  exactly what the SFT trainer consumes.

**Dashboard** (`console.py`, served by `scripts/run_platform.py` at
localhost:8770): 7 views — Overview (KPI tiles incl. CX row, two time-series
charts with tooltips, live feed), Funnel, Slices (ranked table), Cohorts,
Sessions (explorer with full-transcript drill-down + feedback marks),
Data Quality, SQL (SELECT-only self-service + one-click dataset export).
Global cohort filter, emergency alert banner, data-quality badge. Palette
validated with the dataviz checker, light+dark.

**Traffic simulation** (`scripts/synth_traffic.py`): four behavioral
personas (expert/impulsive/browser/confused) with realistic funnel
drop-offs; batch (`--procs` distributed producers), paced-live (`--rate`),
cohort stamping (`--label`, `--weights`) for planted regressions; emits
impressions + clickstream. Measured: ~170 eps/s single producer, 5k+
episodes/55k+ events through the broker; known bottleneck = single SQLite
writer (the "why real platforms shard sinks" talking point).

## 6. The flywheel, end to end (demonstrated live 2026-07-28)

Human shops in PennyMart (real clicks/hovers recorded with event-time) +
synthetic traffic streams alongside → Kafka → dashboard updates in seconds +
S3 parts accumulate → analyst surfaces slices/anomalies → session explorer
shows raw transcripts → SQL/export produces validated training sequences →
(previously) the B-loop retrained the model from exactly such findings →
new model serves the store. A live A/B ran: `1.5b-specialist` (pure action
policy, no chat face) vs `7b-B3` (chats + shops + redirects) with the human
as the shopper and the cohort table filling from their own behavior.

## 7. Compute + cost

~54 GPU-hours total across the entire program (all free single V100s;
≈$165 at AWS on-demand rates, spent $0). Mac runs everything non-GPU:
Kafka, platform, dashboard, storefront, DuckDB, tests.

## 8. Honest limitations (documented, not hidden)

Raw unknown-noun off-catalog requests still under-trigger the redirect
(lexicon coverage = config fix); base-7B shopping eval measures format
compliance; H2's reuse arm not implemented; Spark/Flink/MSK described as the
deployment path, deliberately not faked at this scale; agent latency is
5–15s/turn on the free V100 (measured levers to fix: batching 5–6.4×,
merged-weights vLLM 2.3×, streaming).

## 9. Where things live

- `DEMO_RUNBOOK.md` — the full live-show command sequence.
- `profiling/` — every eval artifact (h3_7b/, b_loop/, compare/, webshop/).
- `benchmarks/artifacts/` — SS0–SS13 measurements + manifests.
- `demos/live_sessions/` (gitignored) — self-captured human episodes.
- Docs repo notes/CHALLENGES.md — 33 root-caused failures worth reading.
