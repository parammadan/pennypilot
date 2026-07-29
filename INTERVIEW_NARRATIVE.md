# The five-minute story
*(Every number below is measured; artifacts in `profiling/behav_cmp/`, full
trail in the docs repo. Rehearse at ~140 wpm; sections are ~40s each.)*

## 1. Problem
I built a multilingual shopping agent — multi-turn, permission-gated,
trained with SFT and RLOO on a single free V100 — and two checkpoints of it:
one SFT-only, one with the RL stage. I wanted to know which to deploy, and
whether my evaluation could even tell them apart.

## 2. Why ordinary evaluation was insufficient
On my registered offline eval — held-out scenarios, n=128, deterministic
task-success scoring — and then on live behavioral replay, the two
checkpoints were **identical: 39/50 task satisfaction each in the first
campaign, then 123/150 each at N=150.** Zero safety violations in both.
A leaderboard would call them the same model. Twice.

## 3. Behavioral signals collected
So I built the platform to look at *how* they succeed, not just whether.
Every conversation streams through Kafka into an event store: authoritative
semantic events emitted by the environment itself — which constraints were
requested and revealed and when, whether each search happened before or
after the requirements were known, whether each recommendation satisfied
them — plus an adaptive simulated customer who reacts to the agent (reveals
constraints when asked well, corrects it when ignored, abandons under
friction) and privately logs *why* it did each thing, so my analytics can be
validated against hidden ground truth.

## 4. The model-quality gap
The friction metrics separated the checkpoints immediately: the SFT model
**searched before knowing the requirements in 51% of sessions vs 29%** for
the RL model, and got **corrected by the customer 47% vs 28%** — confirmed
at N=150 across three customer-patience profiles, four to five standard
errors. Same outcomes, very different customer experience.

## 5. Evidence and root cause
The conversation-level funnel localized it — sessions failing the
"goal understood before searching" milestone, with reason codes like
`searched_before_discovery(missing=[budget])`. Deterministic attribution
rules labeled the failures with cited evidence events, and agreed with the
simulator's hidden abandonment reasons **100% of the time** (n=53). Two
findings I didn't plan: a proxy metric *inverted* — the SFT model reached
"goal understood" in fewer turns because the *customer* was doing the
discovery work through corrections — and customer patience masked friction
entirely: with patient customers the SFT model looked near-perfect, 49/50.
Outcome metrics inherit the customer's tolerance. Friction metrics don't.

## 6. Data recipe
The finding became a versioned, human-approved data recipe: outcome-validated
demonstration sessions from the well-behaved checkpoint — failures inform
targeting but never become labels — deduplicated, hashed, lineage-tracked,
mixed with generated demonstrations and general-chat rehearsal.

## 7. Post-training experiment
Pre-registered: hypothesis, primary metric, guardrails, acceptance and
rollback criteria — then SFT on the recipe dataset and a behavioral replay
against the baseline under identical seeds, simulator version, and serving
config. **My first attempt failed catastrophically** — the trained model was
fluent and never emitted a single action. Root cause was in my data
pipeline: the recipe writer used a constant opener placeholder, so 162
near-identical sequences overfit; the dataset passed every row-level quality
check and failed as a *population*. The guardrail evals caught it before
anything shipped, and the pre-registered rollback kicked in.

## 8. Measured result
The revised recipe worked: **premature search dropped from 55% to 24%,
customer corrections from 54% to 23%, task satisfaction went UP nine
points, retention held at 97.5%, zero violations** — behavioral evidence
converted into a targeted data intervention with a measured behavioral
improvement. And then the system held it back anyway: an untargeted skill —
refusing out-of-catalog requests — regressed below its guardrail, so the
candidate was **not shipped**. Two loop iterations, two different guardrail
catches, zero bad deployments. That's the platform working.

## 9. Limitations and production evolution
Honest limits: one scenario family, results conditional on the simulator's
versioned reaction rules, N=150 arithmetic rather than formal inference,
and the mechanism of why RL reduces premature search is untested. The local
stack is Kafka, SQLite/DuckDB, S3, and Python workers on a laptop plus one
V100 — architecturally the same boundaries as production, where the broker
becomes managed Kafka or Kinesis, the workers become Flink or Spark, the
store becomes a lakehouse, and orchestration replaces my Slurm dependency
chains. I measured what I ran and documented the evolution — I didn't fake
the scale.

---
*One-line version: "My evaluation said two models were identical twice;
behavioral instrumentation showed one made customers do the agent's job,
and turning that evidence into a guarded data recipe halved the friction —
then the same guardrails stopped me from shipping the improvement, because
it had quietly broken something else. That loop — evidence to data to
training to a defensible ship/no-ship decision — is what I built."*
