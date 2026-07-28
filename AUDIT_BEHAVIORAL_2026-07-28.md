# Behavioral-intelligence audit + implementation plan — 2026-07-28

Scope: audit of PennyPilot + PennyData against the "credible miniature of a
production shopping-AI continuous-improvement platform" spec. No code was
modified for this audit. Every pointer names the exact file/function as it
exists at commit HEAD today.

---

## PART A — AUDIT

### A1. Implemented well (preserve; do not rebuild)

| Capability | Where | Why it holds up |
|---|---|---|
| Event transport semantics | `platform/streaming.py` | Keyed-by-session ordering, acks=all, commit-after-apply, replay-needs-fresh-group — all demonstrated live, correct reasoning documented |
| Source-of-truth storage | `platform/store.py` (events.jsonl + rebuildable SQLite) | Replay-rebuild proven; episodes/turns idempotent on (session,i) |
| S3 cold path + lake | `platform/s3_sink.py`, `scripts/lake.py` | Independent consumer group (fan-out), Hive `date=` parts, commit-after-durable-put; DuckDB queries the lake directly |
| Safety invariant | `state/dialogue.py::add_to_cart`, `env/catalog_env.py::_add` | Structural gate; 0.000 violations across every eval ever run; the one claim with airtight evidence |
| Metric definition discipline | `platform/behavior.py::metrics` | Numerator + denominator + eligible population attached to every rate |
| Slice discipline | `behavior.py::slice_report` | min-support, tautology exclusion, universal-value pruning, "hypotheses not causes" labeling |
| Data-quality gating *concept* | `behavior.py::data_quality`, analyst gate in `report.py`, dashboard badge | Right shape; check list is thin (see A3) |
| Training + eval pipeline | `run_sft_v2.py`, `run_rl_v2.py`, `eval_v2.py`, `eval_cannot.py`, `train/sft.py::verify_mask` | Recorded demos w/ build-time asserts, mask verification, registered protocols, behavioral eval w/ fixed probes |
| Pre-registration culture | docs repo `HYPOTHESES.md` | Predictions before runs, failed predictions kept (B2, B4) |
| The B-loop itself | commits `fea5dee`→B3 artifacts | A REAL behavior→data-recipe→retrain→measure loop already ran once — but manually, outside the platform (see A2.9) |
| Demo capture | `demo_human.py` (`--save-dir`, feedback, `--kafka`) | Transcripts/screenshots/votes/clickstream from real humans |

### A2. Implemented but shallow

1. **Customer intent is a string, not a schema.** `events.py::EpisodeStart.brief`
   is `"budget $2210; brand = Asus; min_ram = 32.0"`; synth traffic sends
   literally `brief="synthetic"` (`synth_traffic.py::run_episode`). The env's
   real ground truth (`env/scenario.py`: `hidden_budget`, `all_must_haves`,
   `valid_skus`) never reaches the platform in structured form. No
   expertise/urgency/preferences, no wanted-vs-said-vs-inferred separation.
2. **Reformulation/repeat detection is noise on synthetic data.**
   `behavior.py::session_features` (exact-dupe + Jaccard≥0.6) runs on
   observation strings; synthetic user turns are scripted boilerplate that
   repeats *by construction*, so synth repeat-rates measure the script, not
   behavior. No refinement-vs-correction-vs-frustration taxonomy, no labeled
   fixture set.
3. **The funnel is action-presence, not milestones.** `behavior.py::funnel_by`
   counts sessions whose turns contain an action string. "engaged" = the
   *agent* emitted ask_user — says nothing about the customer. No
   goal_understood, no grounded_recommendation, no reason codes between
   stages. Meanwhile the env already tracks the truth needed
   (`catalog_env.py::_discovered`) — it just never emits it as events.
4. **Failure attribution = 4 regexes.** `quality.py::tag_turn`
   (invalid_action / ungrounded_sku / price_floor_request /
   cannot_fulfill_redirect). No taxonomy, no confidence, no evidence event
   IDs, no abstention. Alerts (`behavior.py::alerts`) name symptoms, not
   causes.
5. **Synthetic traffic scripts BOTH sides.** `synth_traffic.py::run_episode`
   emits pre-planned action sequences — the "agent" in synthetic episodes is
   not a model at all, and the "customer" never reacts to anything. Persona
   outcomes are coded, not emergent (browser never carts because line 86
   returns before carting).
6. **CX metrics measure the generator.** Synth hovers/clicks are emitted
   unconditionally after search (`run_episode`), so `recommendation_ctr` and
   `hover_to_click` on synthetic cohorts quantify script branching. Only the
   handful of human sessions carry meaning. `agent_latency_ms_p50` exists
   only for human sessions but the dashboard tile implies platform-wide.
7. **The automated analyst stops at "slice is elevated."**
   `report.py::analyze` produces template hypotheses + evidence session IDs.
   No alternative explanations, no counterexample comparison, no
   deployment/timeline correlation, no next-analysis suggestions.
8. **Cohorts are labels, not cohorts.** No time-based tracking, no
   before/after view, no retention-style analysis (`quality.py::stats`
   groups by label only).
9. **The learning loop closes OUTSIDE the platform.** The successful B-loop
   (cannot_fulfill recipe → B3) lived in `data/sft_v2.py` + shell commands +
   docs. `platform/export.py::build_dataset` filters by
   label/feedback/violations — no versioned recipe, no mixtures, no
   dedup/labeling policy, no lineage from finding → dataset → run → model →
   eval.
10. **Session explorer shows transcripts, not investigations.** No linked
    evidence events, no attribution, no goal-vs-outcome panel
    (`console.py` Sessions view).

### A3. Missing capabilities

1. **Structured `CustomerGoal`** on episodes + an **outcome record**
   distinct from proxies (task_satisfied, constraints_satisfied,
   cheapest_valid_chosen, permission_obtained — all computable
   deterministically from `env.calculate_outcome()` + `scenario`, currently
   discarded at the platform boundary).
2. **Event envelope**: no `event_id` (⇒ `ui_events` INSERT at
   `store.py:83` is NOT idempotent — Kafka redelivery would double-count
   clickstream; this is a real at-least-once bug today), no
   `model_version`/`prompt_version`/`request_id`/`source` per event, no
   ingest_time-vs-event_time.
3. **Semantic events**: constraint_requested/constraint_revealed (env knows
   both exactly), search_executed with known-constraints-at-search (⇒
   premature-search is deterministically computable), recommendation_shown/
   accepted/rejected, permission_denied as first-class, conversation
   abandonment as an explicit terminal event.
4. **Conversation-aware funnel + reason codes** (needs #3).
5. **Friction & recovery metrics**: turns-to-goal-understood,
   premature-search rate, customer-correction rate, excess turns vs the
   scenario's minimal path (computable: `len(all_must_haves)+4`),
   recovery-after-denial/tool-failure, unnecessary-clarification rate.
6. **Failure-attribution pipeline**: taxonomy + deterministic rules over
   traces + confidence + evidence IDs + abstention; LLM judge only as an
   optional, evidence-citing layer.
7. **Adaptive customer simulator**: reacts to the agent's actual behavior,
   logs its reason per action, seeded, versioned — and crucially, runs
   against the REAL model (the pieces exist: `scripts/batch_rollout.py`
   from SS4 + serve/vLLM recipes; behavioral campaigns of real-policy
   episodes on one V100 are affordable at ~50 eps/GPU-h batched).
8. **A non-planted discovery experiment**: two real checkpoints (e.g.
   `sft7b_B3` vs `rl7b_B3`, or 1.5B vs 7B) on gradual-reveal scenario
   families; the platform must find whichever friction gap actually exists.
9. **Versioned data recipes + lineage tables** (recipe → dataset hash →
   training run → model version → eval results), approval gate before
   export.
10. **Behavioral before/after replay evaluation** (same seeds, same
    simulator version, two models) as a first-class platform artifact.
11. **Data-quality checks that catch cross-event inconsistencies**:
    click-without-impression, timestamp inversion, event/ingest lag,
    invalid funnel transitions, cart-outcome conflicts, missing goal
    metadata on synthetic sessions, simulator-reason vs emitted-action
    consistency.
12. **Uncertainty handling for real humans**: inferred-intent confidence,
    "ambiguous" as an admissible label.

### A4. Misleading or scientifically weak claims (as currently presented)

1. **"The platform discovered a planted regression"** — the v14 regression
   *was* a persona-mix keyed to the label; detection is circular. Valid as
   a wiring test, not as discovery. (`synth_traffic --label/--weights` +
   `report.py` test.)
2. **Persona findings are tautologies** — "synth-browser has 100%
   abandonment" restates its code. Same for the "confounder demo"
   (7+ turns ⇒ low abandonment): a good *illustration*, not a finding.
3. **Synthetic repeat/reformulation/CTR/hover metrics** presented as CX
   insight measure the generator (A2.2, A2.6).
4. **"Effectively-once views"** is true for episodes/turns, FALSE for
   ui_events (no dedup key — A3.2).
5. **Funnel stage names** ("engaged") imply customer behavior but encode
   agent actions.
6. **Latency p50 tile** implies platform-wide; it is human-sessions-only.
7. **"Behavioral intelligence implemented"** overall: the *discipline*
   (denominators, min-support, caveats, gates) is real and good; the
   *evidence base* under it is mostly scripted. The story must say so until
   real-policy adaptive-sim campaigns replace scripted traffic.

---

## PART B — PHASED IMPLEMENTATION PLAN

Ordering principle: semantics first (otherwise every downstream number stays
hollow), then behavior generation, then the loop, then presentation.

### Phase 1 — Strengthen semantics (CPU only; ~1 session)

**1.1 Event envelope + CustomerGoal + Outcome** — `platform/events.py`
- Add to every event: `event_id` (uuid, default), `source`
  ("AGENT"|"CUSTOMER"|"STORE_UI"|"SIMULATOR"|"SYSTEM"), `model_version`,
  `prompt_version`, `request_id` (optional), `ingest_ts` (set by store).
- New models: `CustomerGoal` (goal text, budget_max, must_have_constraints,
  preferences, expertise, urgency, goal_source ∈ SIMULATED_GROUND_TRUTH |
  HUMAN_BRIEF | INFERRED{confidence} | AMBIGUOUS) attached to
  `episode_start`; `Outcome` block on `episode_end` (task_satisfied,
  constraints_satisfied, cheapest_valid, permission_obtained, cart_correct,
  violation, goal_satisfaction ∈ {SATISFIED, PARTIAL, UNSATISFIED, UNKNOWN}).
- Migration: all new fields optional-with-defaults; old JSONL replays intact
  (test: replay an old log fixture).
- **Fix the ui_events idempotency bug**: dedup on `event_id`
  (`store.py::_apply` → INSERT OR IGNORE with event_id PK; migration adds
  column).

**1.2 Semantic events emitted from truth, not inference** —
`env/catalog_env.py` + `demo_human.py` + simulator driver
- Env exposes a per-step `events` list (constraint_requested,
  constraint_revealed{key}, search_executed{known_constraints, n_results,
  premature: bool}, recommendation_shown{sku, grounded: bool},
  permission_requested/denied/granted, cart_added). Drivers forward them.
- Who produces / authoritative: env = authoritative for conversation-level
  facts (server-side); storefront JS remains authoritative for raw UI ticks
  (client-side, non-authoritative for outcomes). Documented per event in
  the module docstring, incl. dedup + ordering + privacy note.

**1.3 Conversation-aware funnel + reason codes** — `platform/behavior.py`
- New `milestone_funnel()`: session_started → goal_understood (all
  must-haves revealed) → valid_search (search with full constraints) →
  eligible_found → grounded_recommendation → customer_engaged (accept OR
  explore) → permission_requested → permission_granted → correct_cart →
  task_satisfied. Reach + stage-to-stage conversion + reason codes
  (premature_search, missing_budget, ungrounded_rec, denial_no_recovery…)
  derived from semantic events, never regex-on-text.

**1.4 Outcome-vs-proxy metric hierarchy** — `behavior.py::metrics` returns
three groups (outcomes / proxies / diagnostics); every metric gains
`window`, `min_support`, `freshness`; dashboard displays group headers and
per-metric provenance. Add the "proxy may invert" note to proxy metrics.

**1.5 Attribution taxonomy v1 (deterministic only)** — new
`platform/attribution.py`: the spec's taxonomy; rules over semantic events
(e.g. premature_search + correction + abandon ⇒ CONSTRAINT_EXTRACTION
primary, evidence = event_ids, confidence from rule specificity;
else UNKNOWN/abstain). LLM-judge layer explicitly deferred to Phase 2+
(interface stub only).

Tests: schema round-trip incl. old-log replay; env-event emission against
scripted episodes with known truth; funnel milestones on hand-built
fixtures; attribution rules on fixture traces (one per taxonomy label +
abstention). Acceptance: all 152 existing tests still pass; new ~25 tests;
`analyze()` report shows outcome metrics ≠ proxy metrics.
**Not building**: multi-device identity, inactivity sessionization (explicit
IDs exist), privacy tooling beyond a documented note (synthetic data).

### Phase 2 — Real behavior (CPU + ~4–6 GPU-h; ~2 sessions)

**2.1 Scenario families** — new `env/scenario_families.py`: the spec's 10
families (constraint-discovery, conflicting-prefs, ambiguous, comparison,
correction, tool-failure, inventory-change, permission-denial,
out-of-catalog, code-switched) as data objects: hidden truth, allowed/
prohibited behaviors, minimal path length, deterministic outcome checks,
likely failure categories. Reuses `generate_hard_scenarios` machinery.

**2.2 Adaptive customer simulator** — new `env/customer_sim.py`
(NOT the agent's policy): observes the last agent action + its own goal,
reacts per family rules (correct-the-agent, reveal-on-request, frustrate on
repeats, wait-once on tool failure, abandon on thresholds), seeded,
`simulator_policy_version`, logs `{customer_action, reason,
trigger_event_ids}` — enabling the "does inferred friction match hidden
cause" validation loop.

**2.3 Real-model behavioral campaigns** — extend `scripts/batch_rollout.py`
(SS4) into `scripts/behavior_campaign.py`: N adaptive-sim episodes against
a REAL checkpoint on one V100 (merged-weights vLLM, ~2.3× recipe), events
streamed/exported to the platform with model_version stamped. Replaces
scripted traffic as the analytical evidence base (personas retire to
load-testing only).

**2.4 Layered reformulation classification** — `platform/reformulation.py`:
L1 exact dupe, L2 token overlap, L3 constraint-delta from
`lang/extract.py` (refinement adds a constraint; correction re-asserts a
violated one — deterministic, no embeddings needed), L4 = simulator reason
when available. Labeled fixture set (≥30 examples across the 6 categories)
+ tests. Jaccard-only path retired.

**2.5 Friction/recovery metrics** (A3.5 list) — `platform/friction.py`,
joined into `metrics()` diagnostics group + slices dimensions
(expertise, scenario_family, language, constraint_complexity).

**2.6 The discovery experiment (not planted)** — pre-register in docs:
run `sft7b_B3` vs `rl7b_B3` (same base, SFT-only vs post-RL) on
constraint-discovery + correction families with novice-profile simulation;
hypothesis: a friction gap exists (direction stated before the run); the
platform must localize it via funnel reason codes + friction slices +
transcript evidence, with the simulator's hidden reasons as validation.
Whatever is found (including "no gap") is the reported result.
Acceptance: analyst report localizes the top friction slice to the correct
hidden cause in ≥70% of attribution-eligible sessions (measured against
simulator reasons); zero hard-coded conclusions in alert text.
**Not building**: embedding/LLM semantic similarity (L3 is
constraint-delta), real WebShop, multi-day retention cohorts.

### Phase 3 — Close the loop inside the platform (CPU + ~4 GPU-h)

**3.1 Versioned data recipes** — `platform/recipes.py`: the spec's recipe
schema (sources incl. failed+counterexample+rehearsal mixtures, filters,
dedup policy, labeling policy = corrected-trajectory (simulator corrections
give the "what should have happened" continuation), eval slices, expected
effect, regression risks, approval_status). Registry = versioned JSON in
repo + lineage tables in store (`recipes`, `training_runs`, `evaluations`).
`export.py` becomes recipe-driven; direct export stays for ad-hoc.

**3.2 Candidate scoring + validation** — recipe apply produces a dataset
manifest (counts per source, dedup drops, quality checks incl. the
existing gate: no violations, mask-verifiable format) — never publishes
unvalidated agent output (Rule 11).

**3.3 Controlled experiment harness** — `scripts/experiment.py`:
pre-registered config (hypothesis/primary/guardrails/sample
size/acceptance/rollback), drives: recipe → SFT(+RL) run on cluster →
registered offline evals → behavioral REPLAY evaluation (same seeds, same
simulator version, both models) → before/after report with
association-vs-experimental-evidence language. Reuses the B-loop's proven
job-chain pattern + fail-fast probes.

Acceptance: one full loop executed end-to-end with lineage visible:
finding → recipe vX → dataset hash → run → model version → eval + replay
deltas. **Not building**: automated approval (human approves recipes),
online A/B routing.

### Phase 4 — Presentation (the React rebuild lands here)

- **React + Vite + TS dashboard** (user-chosen stack) organized around the
  spec's views: Executive / Journey (milestone funnel + reason codes) /
  Friction / Model-quality slices / **Investigation** (metric change →
  slices → timeline → deployments → attributed causes → representative
  sessions → alternative explanations → suggested next analysis) /
  **Dataset proposal** (recipe view w/ sample transcripts + approval) /
  Lineage. Legacy console stays until parity.
- Data-quality checks from A3.11 wired so CRITICAL failures visibly degrade
  every report ("validate → analyze → hypothesize" enforced in UI).
- `DEMO_RUNBOOK` v2 = the spec's 12-step demo; `SYSTEM_OVERVIEW` updated;
  the 5-minute interview narrative written from the *measured* Phase 2–3
  results.
- Local-vs-production architecture doc (Kafka+SQLite/DuckDB+S3+Python
  workers now; managed Kafka/Kinesis + Flink/Spark + lakehouse +
  orchestration as the documented evolution — never claimed as built).

### Estimated totals
CPU work dominates; GPU ≈ 8–12 h (campaigns + one retrain + evals) — within
the free V100 budget. Riskiest item: adaptive simulator quality (mitigate
with small labeled fixture sets + simulator-reason validation loop).
