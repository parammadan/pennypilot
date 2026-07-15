"""Hidden-need scenario generator — the core of the Pennywise task.

The single-turn setup (`data/prompts.py`) writes the constraints INTO the prompt
("a laptop under $1200 with 32GB RAM"), so the model never has to ask — it just
reads and filters. Pennywise deletes that shortcut. A `Scenario` carries a
*hidden* need the shopper does not volunteer:

  - a hidden **budget** (the real max price), and
  - a single hard **must-have** feature (e.g. >=32GB RAM),

plus a deliberately vague `opening_utterance` ("I need a new laptop") that
mentions NEITHER. The full hidden constraint set (`hidden_constraints`) is what
the reward layer scores against — it is ground truth derived from the catalog,
so the reward stays verifiable with no reward model.

Because the constraints are absent from the opening utterance and revealed by the
user-simulator ONLY in response to a relevant clarifying question, the
reward-maximizing policy is *structurally* forced to ask before it recommends:
the information needed to pick a valid, cheapest item is not on the table until
the model asks for it. That is what makes "ask a good clarifying question" a
learned behaviour rather than a hard-coded ritual.
"""
from __future__ import annotations

import random

from pydantic import BaseModel

from shoprl.data.catalog import Product
from shoprl.data.prompts import satisfies

# The must-have is exactly one hard feature constraint. Budget (`max_price`) is
# handled separately as its own hidden dimension, so it is not in this list.
_MUST_HAVE_KEYS = ("min_ram", "max_weight", "min_battery", "brand")

_OPENINGS = [
    "I'm looking for a new laptop.",
    "I need to buy a laptop, can you help?",
    "Hi, I want to get a laptop.",
    "Looking to pick up a new laptop soon.",
    "I'm in the market for a laptop.",
]


class Scenario(BaseModel):
    """One shopping episode's hidden ground truth.

    `opening_utterance` is all the model sees at reset. `hidden_budget` and
    `must_have_*` are revealed by the simulator only when asked the matching
    clarifying question. `valid_skus` is the derived answer set the reward uses.
    """
    scenario_id: str
    category: str
    opening_utterance: str
    hidden_budget: float
    must_have_key: str
    must_have_value: float | str
    valid_skus: list[str]

    @property
    def hidden_constraints(self) -> dict[str, float | str]:
        """The full hard-constraint set = budget + must-have. `satisfies()` (for
        numeric keys) and an explicit brand check together define 'valid'."""
        return {"max_price": self.hidden_budget, self.must_have_key: self.must_have_value}


def _brands(catalog: list[Product]) -> list[str]:
    return sorted({p.brand for p in catalog})


def scenario_valid_skus(catalog: list[Product], budget: float,
                        must_have_key: str, must_have_value: float | str) -> list[str]:
    """SKUs meeting BOTH the budget and the must-have — the verifiable answer set.

    Brand is a categorical must-have so it is checked here rather than through
    `satisfies()` (which only knows the numeric spec constraints)."""
    out = []
    for p in catalog:
        if p.price > budget:
            continue
        if must_have_key == "brand":
            if p.brand != must_have_value:
                continue
        elif not satisfies(p, {must_have_key: float(must_have_value)}):
            continue
        out.append(p.sku)
    return out


def _sample_must_have(rng: random.Random, catalog: list[Product]) -> tuple[str, float | str]:
    key = rng.choice(_MUST_HAVE_KEYS)
    if key == "min_ram":
        return key, float(rng.choice([16, 32, 64]))
    if key == "max_weight":
        return key, float(rng.choice([3.0, 3.5, 4.0]))
    if key == "min_battery":
        return key, float(rng.choice([10, 12, 15]))
    # brand
    return key, rng.choice(_brands(catalog))


def generate_scenarios(
    catalog: list[Product],
    n: int = 200,
    seed: int = 0,
    category: str = "laptop",
    valid_target_range: tuple[int, int] = (20, 80),
    max_tries: int = 80,
) -> list[Scenario]:
    """Deterministically generate `n` hidden-need scenarios.

    `valid_target_range` makes valid-set size a TUNABLE KNOB, not a constant: each
    scenario targets a variable number of valid items (default dozens). This
    matters for the reward — `value_quality` grades the recommendation by its
    price-rank among the valid set, so a valid set of size ~2 collapses that into
    a near-binary signal and starves the ranking gradient. A valid set of dozens
    makes "cheapest valid" a meaningful ordering.

    Each scenario is resampled until its (budget, must-have) pair admits a
    non-empty valid set. The budget is pinned to the k-th cheapest
    must-have-qualifying price (k drawn from `valid_target_range`), so ~k items
    are valid while pricier must-have-qualifying items stay over budget — the
    budget stays binding (asking about it pays off) and satisfiable at once.
    """
    rng = random.Random(f"scenario-{seed}")
    scenarios: list[Scenario] = []
    for i in range(1, n + 1):
        for _ in range(max_tries):
            must_key, must_val = _sample_must_have(rng, catalog)
            # Prices among items meeting the must-have; the budget indexes into
            # this sorted pool to hit the target valid count.
            if must_key == "brand":
                pool = [p.price for p in catalog if p.brand == must_val]
            else:
                pool = [p.price for p in catalog
                        if satisfies(p, {must_key: float(must_val)})]
            if not pool:
                continue
            pool.sort()
            target = rng.randint(*valid_target_range)
            # Cap below the full pool so at least one qualifying item stays over
            # budget -> the budget is always a binding, discoverable constraint.
            k = min(target, len(pool) - 1) if len(pool) > 1 else 1
            if k < 1:
                continue
            budget = round(pool[k - 1] + 0.01, 2)
            valid = scenario_valid_skus(catalog, budget, must_key, must_val)
            if valid:
                break
        scenarios.append(
            Scenario(
                scenario_id=f"S-{i:04d}",
                category=category,
                opening_utterance=rng.choice(_OPENINGS),
                hidden_budget=budget,
                must_have_key=must_key,
                must_have_value=must_val,
                valid_skus=valid,
            )
        )
    return scenarios
