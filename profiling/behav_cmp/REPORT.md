# Behavioral comparison — sft7b-B3 vs rl7b-B3
2026-07-28 · N=50 episodes/model · identical scenario seeds (100) · customer-sim-v1
(GRADUAL_CONSTRAINT_REVEAL, novice profile) · identical catalogue + generation
config · single V100, 38 min total · job 8806702

## Headline
**Outcome evaluation cannot tell these models apart. Behavior can.**

| metric | sft7b-B3 | rl7b-B3 |
|---|---|---|
| task satisfied (deterministic outcome) | **39/50** | **39/50** |
| safety violations | 0 | 0 |
| abandoned (simulator: EXCESSIVE_FRICTION) | 10 | 11 |
| **premature-search rate** | **52%** | **36%** |
| **customer-correction rate** | **48%** | **34%** |
| **median turns to goal understood** | **6** | **4** |
| avg agent latency / turn | 1972 ms | 1823 ms |
| avg turns / episode | 7.8 | 7.8 |

At matched task success, the RLOO stage's contribution is PROCESS quality:
it asks before it searches (36% vs 52% premature), gets corrected ~30%
(relative) less often, and reaches full goal understanding two turns sooner.
A leaderboard-style outcome eval would have called these checkpoints
equivalent.

## Funnel (reach /50)
Identical at every outcome-adjacent stage (both 38 valid_search, 50 grounded
recommendations, 39 task-satisfied). Reason codes localize the loss: 10–11
sessions per model reached recommendations only via premature searches
(`only_premature_searches`) and died to `no_permission_request` after the
customer abandoned. One sft session carted a valid but not-cheapest item.

## Attribution — honest abstention
The v1 CONSTRAINT_EXTRACTION rule fired on **0/22** failed sessions, each
abstaining with identical evidence booleans: premature=True, correction=True,
missed_or_bad_rec=False. Reading: corrections successfully revealed every
constraint, so "still-missing constraint evidence" never held. The dominant
real failure mode is **abandonment after accumulated friction with eventual
full disclosure** — a pattern the v1 taxonomy cannot name. This is a
measured coverage gap, not a bug: the rule was specified before the data and
is not being loosened after seeing it.

## Validation against hidden causes
All 21 abandonments carry simulator reason EXCESSIVE_FRICTION, with per-turn
triggers (AGENT_REPEATED_KNOWN_QUESTION, AGENT_SEARCHED_PREMATURELY,
AGENT_IGNORED_<KEY>) logged with trigger event IDs in
`campaign_*_simlog.jsonl` — the analytics-derived friction signals point at
the same sessions the simulator privately penalized.

## Limitations
- N=50/model: the friction gaps (16 pts, 14 pts) are ~1.5–2 binomial SEs —
  strongly suggestive, not conclusive; a confirmation run needs N≈150+.
- One scenario family, one customer profile, template correction phrasings.
- Simulator constants (frustration_limit=3) are versioned choices; results
  are conditional on customer-sim-v1.
- Controlled comparison of two checkpoints under identical conditions ⇒ the
  difference is attributable to the training stage (SFT-only vs +RLOO), but
  the MECHANISM (why RLOO reduces premature search) is untested.

## Recommended next vertical slice
1. Attribution v2: add an EXCESSIVE_FRICTION/REDUNDANT_QUESTIONING category
   (deterministic, from redundant-ask + abandonment events), validated
   against simulator reasons — the data showed exactly where v1 abstains.
2. Confirmation run at N=150 with a second scenario family (CORRECTION).
3. Close the loop: a data recipe targeting premature search in the SFT arm
   (failed sessions + rl7b-B3's successful counterexamples), per the
   already-planned Phase 3.
