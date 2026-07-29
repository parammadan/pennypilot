# Loop-closing experiment — final verdict (recipe premature-search-v2)
2026-07-29 · sft7b-C2 = 7B base + SFT on the approved recipe dataset
(162 behavioral demonstrations w/ real openers + 270 generated all-kinds +
123 rehearsal) · replay vs sft7b-B3 at seeds 300–302, three patience
profiles, N=150 each, identical conditions · lineage: recipe v2 →
sha 87552315213abbba → sft7b_C2 → this evaluation

## Pre-registered scorecard

| check | bar | measured | verdict |
|---|---|---|---|
| PRIMARY premature-search | < 40% | **24.0%** (B3: 55.3%) | **MET, decisively** |
| task satisfaction | within 6 pts of B3 | 128/150 vs 114/150 (+9.3 pts BETTER) | MET |
| violations | 0 | 0/300 | MET |
| retention | ≥ 90% | 97.5% | MET |
| cannot-redirect | ≥ 10/12 | **5/12** | **BREACHED** |

Supporting behavioral deltas (same replay): customer corrections 23.3% vs
54.0%; abandonment 20 vs 34. Offline registered shopping (n=128): 0.65 —
SFT-level, as expected without an RL stage (B3's 0.906 is post-RL).

## Decision (per the registered acceptance rule: primary + ALL guardrails)
**NOT SHIPPED.** The behavioral claim is confirmed — a data recipe built
from behavioral evidence transferred the asking-before-searching discipline
through SFT alone, halving premature search and corrections while IMPROVING
task satisfaction — but the candidate regressed a deployed skill the recipe
never targeted: the cannot-fulfill redirect (5/12 vs B3's 12/12). The
guardrail did exactly what guardrails are for: caught a regression in a
capability nobody was looking at. v3 path: raise cannot-fulfill demo density
and/or add the RL stage that gave B3 its notice-generalization.

## What the two iterations proved about the PLATFORM (the actual product)
- v1 failed catastrophically from a data-pipeline defect (constant opener →
  population-level homogeneity) that passed every ROW-level check — caught
  by the guardrail evals before shipping; rolled back per pre-registration.
- v2 fixed the data, confirmed the behavioral transfer, and was STILL held
  back by a different guardrail. Two iterations, two catches, zero unsafe or
  regressed ships. The loop is: behavior → evidence → attribution → recipe →
  training → guardrailed evaluation → decision — and every arrow has an
  artifact.

---
# v3 verdict (2026-07-29) — ACCEPTED

| check | bar | v3 measured | verdict |
|---|---|---|---|
| cannot-redirect | ≥ 10/12 | **12/12** (raw 4/6, best ever) | MET |
| PRIMARY premature-search | < 40% | **32.0%** (B3 replay: 55.3%) | MET |
| replay task satisfaction | within 6 pts of B3 | 126/150 vs 114/150 (+8) | MET |
| violations | 0 | 0/300 | MET |
| retention | ≥ 90% | **100%** | MET |

First candidate to pass the complete acceptance rule. The cannot-density
lever fixed the v2 regression outright (5/12 → 12/12) at a measured cost:
part of the friction gain (premature 24%→32%, corrections 23%→31% vs C2) —
still decisively better than baseline, and the trade-off is now a documented,
tunable dial.

## The capstone finding: the evaluation surfaces disagree
- Offline registered protocol (n=128, scripted reveals): **B3 0.906 > C3 0.52**
- Behavioral replay (adaptive customer): **C3 126/150 > B3 114/150**, with
  32% vs 55% premature search

Two evaluation surfaces RANK THE MODELS OPPOSITELY. The offline protocol's
scripted shopper always answers questions compliantly — it rewards B3's
RL-tuned flow and never punishes premature search; the adaptive customer
does. Neither number is "wrong"; they measure different customers. This is
the platform's thesis in one line: WHICH MODEL IS BETTER DEPENDS ON HOW YOUR
EVALUATION'S CUSTOMER BEHAVES — so evaluate with customers that behave.

## Shipping recommendation
C3 is the accepted recipe outcome and the better model under behavioral
evaluation; B3 remains stronger on the RL-tuned offline protocol. The
registered synthesis path (not yet run): RLOO stage on top of C3 — expected
to recover offline strength while keeping the discipline SFT instilled.
Loop chapter total: 3 iterations, ~8 GPU-h, two guardrail catches, one
accepted candidate, one surface-disagreement finding.
