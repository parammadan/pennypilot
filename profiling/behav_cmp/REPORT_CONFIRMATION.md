# Confirmation run verdict — vs pre-registration (docs 2026-07-28)
N=150/model · seeds 200–202 · frustration profiles 2/3/5 (50 eps each) ·
customer-sim-v1 · job 8808277 (~1h V100)

## Predictions vs measurements

**P1 — friction gaps persist (pooled): 2 of 3 CONFIRMED, 1 REVERSED with
mechanism found.**
- premature-search: 51.3% (sft) vs 29.3% (rl) — 22-pt gap, ≈4–5 binomial SE.
  CONFIRMED, now conclusive (run 1: 52/36).
- customer-correction: 46.7% vs 28.0% — 18.7-pt gap. CONFIRMED (run 1: 48/34).
- turns-to-goal-understood: median 3 (sft) vs 4 (rl) — **REVERSED** vs run 1.
  Decomposition explains it: sft's constraints were revealed via CUSTOMER
  CORRECTIONS 70 times vs rl's 42 — when the agent skips discovery, the
  customer does the work, and the "turns to goal" clock runs FASTER while
  friction runs higher. The proxy inverted exactly as proxy-metric theory
  warns (fewer turns ≠ better). Metric marked flawed-as-defined; v2 must
  count only agent-elicited reveals or report reveal-source composition.

**P2 — direction holds per profile: CONFIRMED (6/6).** Premature and
correction gaps favor rl at every frustration limit (fl2: 44/26 & 36/26;
fl3: 58/30 & 58/30; fl5: 52/32 & 46/28).

**P3 — task satisfaction indistinguishable: CONFIRMED,** startlingly:
123/150 vs 123/150, identical again (run 1: 39/50 both).

**P4 — violations: CONFIRMED.** 0 in 300 sessions (gate holds under every
profile).

**P5 — attribution agreement ≥80%: CONFIRMED at 1.00** (n=53 abandoned).
On this data the taxonomy also separated: EXCESSIVE_FRICTION 51,
CONSTRAINT_EXTRACTION 2 (genuine still-missing-constraint cases), 1 correct
abstention.

## The profile finding (unregistered, reported as observed association)
Customer patience masks friction: with patient customers (fl5) sft satisfies
49/50 — nearly perfect outcomes riding on customer-performed discovery; with
impatient customers (fl2) it drops to 33/50. Task success is a function of
CUSTOMER TOLERANCE; the friction metrics are the stable signal underneath.
This is the strongest one-line justification for behavioral evaluation:
outcome metrics inherit the customer's patience, friction metrics do not.

## Limitations
Same family + simulator version as run 1 (gaps are conditional on
customer-sim-v1's reaction rules); turns-to-goal metric retired as defined;
no significance tests beyond SE arithmetic; mechanism for WHY RLOO reduces
premature search remains untested.
