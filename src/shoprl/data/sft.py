"""SFT warmup dialogue generator (pure CPU).

Synthetic expert demonstrations of the Pennywise ritual:

    user opens vaguely  ->  agent ASKS to discover the hidden budget + must-have
    ->  agent RECOMMENDS the genuinely cheapest valid item  ->  agent ASKS
    PERMISSION  ->  agent ADDS only after the user accepts.

RL sharpens an existing behaviour; it does not invent the asking ritual from
nothing (a policy that never asks gets a flat, uninformative reward and never
discovers that asking pays off). SFT on these demos gives the 1.5B instruct
model the ritual first, so RL starts on-distribution.

Correctness is guaranteed by construction, not by looking plausible:
  - the discovery turns reveal the scenario's TRUE hidden values
    (`hidden_budget`, `must_have_*`), so the clarifying questions map to the
    actual hidden constraints; and
  - the recommendation is the cheapest item in the scenario's verifiable
    `valid_skus` set.
`tests/test_sft.py` re-checks both by replaying every demo's agent actions
through `PennyEnv` and asserting the cheapest valid item is legitimately added.

Agent turns are the structured action grammar (ASK[...]/RECOMMEND[SKU]/
ASK_PERMISSION/ADD_TO_CART[SKU]) so they parse unambiguously and teach the
skeleton exactly; the user turns carry the natural-language variety.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from shoprl.data.catalog import Product, catalog_index
from shoprl.data.prompts import satisfies
from shoprl.env.scenario import Scenario, generate_scenarios

# --- user-turn phrasing pools (agent turns stay structured grammar tokens) ---
_BUDGET_REVEAL = [
    "My budget is about ${b:.0f}.",
    "I'd like to stay under ${b:.0f}.",
    "I can spend up to around ${b:.0f}.",
    "Around ${b:.0f} would be ideal.",
]
_BACKCHANNEL = ["Okay.", "Sounds good.", "Sure.", "Got it.", "Makes sense."]
_ACCEPT = [
    "Yes, please add it.",
    "Perfect, add that one.",
    "Great, let's go with that.",
    "Yes, that works — add it.",
]
_CLOSING = ["Thanks!", "Great, thank you.", "Appreciate the help.", "Perfect, thanks."]
_NONCOMMITTAL = [
    "Let me think about it for a bit.",
    "I'm not quite ready to decide yet.",
    "Can I sit on that for a moment?",
]
_REFRAIN = [
    "No problem — I won't add anything to your cart until you give me the go-ahead.",
    "Of course, take your time. I'll hold off until you're sure.",
    "Understood, I won't add it without your okay.",
]


def _feature_reveal(scen: Scenario, rng: random.Random) -> str:
    k, v = scen.must_have_key, scen.must_have_value
    if k == "min_ram":
        return rng.choice([
            f"It needs at least {int(v)}GB of RAM.",
            f"I really need {int(v)}GB of memory minimum.",
            f"At least {int(v)} gigs of RAM is a must.",
        ])
    if k == "max_weight":
        return rng.choice([
            f"It can't weigh more than {v} lbs.",
            f"It has to stay under {v} pounds — I travel a lot.",
            f"Lightweight matters; {v} lbs max.",
        ])
    if k == "min_battery":
        return rng.choice([
            f"I need at least {int(v)} hours of battery.",
            f"Battery life is key — {int(v)}+ hours.",
            f"It should last at least {int(v)} hours on a charge.",
        ])
    # brand
    return rng.choice([
        f"It has to be a {v}.",
        f"I only want a {v}.",
        f"Brand matters — {v} specifically.",
    ])


def _reject_line(price: float, rng: random.Random) -> str:
    return rng.choice([
        f"Hmm, ${price:.0f} is over my budget.",
        f"That one's ${price:.0f} — too expensive for me.",
        "That's a bit more than I want to spend.",
    ])


@dataclass
class DemoTurn:
    role: str            # "agent" | "user"
    text: str            # structured action (agent) or natural language (user)
    action: str | None = None  # parsed grammar action for agent turns, else None


@dataclass
class Demo:
    scenario_id: str
    kind: str            # "positive" | "rejection_recovery" | "permission_edge"
    order: str           # "budget_first" | "feature_first"
    n_clarify: int       # number of clarifying (ASK) agent turns
    has_rejection: bool
    target_sku: str      # the cheapest valid item (the demonstrated recommendation)
    turns: list[DemoTurn] = field(default_factory=list)


def cheapest_valid(scen: Scenario, idx: dict[str, Product]) -> str:
    """The genuinely cheapest item in the verifiable valid set (the target)."""
    return min(scen.valid_skus, key=lambda s: idx[s].price)


def _bad_sku(scen: Scenario, catalog: list[Product],
             rng: random.Random) -> str | None:
    """An item to be rejected in a recovery demo. Prefer one that meets the
    must-have but is OVER budget (a realistic 'nice but too pricey' reject);
    fall back to any non-valid item."""
    def feat_ok(p: Product) -> bool:
        if scen.must_have_key == "brand":
            return p.brand == scen.must_have_value
        return satisfies(p, {scen.must_have_key: float(scen.must_have_value)})

    over = [p for p in catalog if p.price > scen.hidden_budget and feat_ok(p)]
    pool = over or [p for p in catalog if p.sku not in scen.valid_skus]
    return rng.choice(pool).sku if pool else None


def _agent(action: str) -> DemoTurn:
    return DemoTurn("agent", action, action)


def _discovery_turns(scen: Scenario, order: str, confirm: bool,
                     rng: random.Random) -> tuple[list[DemoTurn], int]:
    budget = (_agent("ASK[budget]"), _BUDGET_REVEAL[rng.randrange(len(_BUDGET_REVEAL))]
              .format(b=scen.hidden_budget))
    feature = (_agent("ASK[feature]"), _feature_reveal(scen, rng))
    seq = [budget, feature] if order == "budget_first" else [feature, budget]

    if confirm:
        # A benign extra clarifying turn: re-confirm the first constraint asked.
        # Reveals nothing new (info-gain 0) but is a natural, correct question —
        # this is how we vary the number of clarifying turns without teaching a
        # wrong behaviour.
        act = seq[0][0].action
        reveal = (_BUDGET_REVEAL[rng.randrange(len(_BUDGET_REVEAL))].format(b=scen.hidden_budget)
                  if act == "ASK[budget]" else _feature_reveal(scen, rng))
        seq.append((_agent(act), f"Right — {reveal[0].lower() + reveal[1:]}"))

    turns: list[DemoTurn] = []
    for agent_turn, reveal in seq:
        turns.append(agent_turn)
        turns.append(DemoTurn("user", reveal))
    return turns, len(seq)


def _build_demo(scen: Scenario, catalog: list[Product], idx: dict[str, Product],
                kind: str, order: str, confirm: bool, rng: random.Random) -> Demo:
    target = cheapest_valid(scen, idx)
    turns: list[DemoTurn] = [DemoTurn("user", scen.opening_utterance)]
    disc, n_clarify = _discovery_turns(scen, order, confirm, rng)
    turns += disc
    has_rejection = False

    if kind == "rejection_recovery":
        bad = _bad_sku(scen, catalog, rng)
        if bad is not None:
            has_rejection = True
            turns.append(_agent(f"RECOMMEND[{bad}]"))
            turns.append(DemoTurn("user", _reject_line(idx[bad].price, rng)))

    if kind == "permission_edge":
        # Correctly DECLINE to add without an explicit accept: recommend, the
        # user is non-committal, the agent refrains from ADD_TO_CART.
        turns.append(_agent(f"RECOMMEND[{target}]"))
        turns.append(DemoTurn("user", rng.choice(_NONCOMMITTAL)))
        turns.append(DemoTurn("agent", rng.choice(_REFRAIN), None))  # NL, no action
        return Demo(scen.scenario_id, kind, order, n_clarify, has_rejection,
                    target, turns)

    # Positive close (also the tail of a rejection_recovery demo).
    turns.append(_agent(f"RECOMMEND[{target}]"))
    turns.append(DemoTurn("user", rng.choice(_BACKCHANNEL)))
    turns.append(_agent("ASK_PERMISSION"))
    turns.append(DemoTurn("user", rng.choice(_ACCEPT)))
    turns.append(_agent(f"ADD_TO_CART[{target}]"))
    turns.append(DemoTurn("user", rng.choice(_CLOSING)))
    return Demo(scen.scenario_id, kind, order, n_clarify, has_rejection, target, turns)


def generate_sft_dialogues(
    catalog: list[Product],
    n: int = 1000,
    seed: int = 0,
    rejection_frac: float = 0.15,
    edge_frac: float = 0.03,
) -> list[Demo]:
    """Generate `n` internally-correct demonstration dialogues.

    The bulk are correct positive demos; `rejection_frac` add a rejected bad
    recommendation followed by recovery to the cheapest valid item; `edge_frac`
    are permission-edge demos that correctly decline to add without an accept.
    Each demo uses a distinct hidden-need scenario, so needs are diverse.
    """
    rng = random.Random(f"sft-{seed}")
    idx = catalog_index(catalog)
    scenarios = generate_scenarios(catalog, n=n, seed=seed)
    demos: list[Demo] = []
    for scen in scenarios:
        roll = rng.random()
        if roll < edge_frac:
            kind = "permission_edge"
        elif roll < edge_frac + rejection_frac:
            kind = "rejection_recovery"
        else:
            kind = "positive"
        order = rng.choice(["budget_first", "feature_first"])
        confirm = kind != "permission_edge" and rng.random() < 0.2
        demos.append(_build_demo(scen, catalog, idx, kind, order, confirm, rng))
    return demos


def demo_stats(demos: list[Demo]) -> dict:
    """Structural-variety summary (for the report + variety test)."""
    return {
        "count": len(demos),
        "by_kind": dict(Counter(d.kind for d in demos)),
        "by_order": dict(Counter(d.order for d in demos)),
        "by_n_clarify": dict(Counter(d.n_clarify for d in demos)),
        "with_rejection": sum(d.has_rejection for d in demos),
        "turn_len_min": min(len(d.turns) for d in demos),
        "turn_len_max": max(len(d.turns) for d in demos),
    }


def write_jsonl(demos: list[Demo], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for d in demos:
            f.write(json.dumps(asdict(d)) + "\n")
    return path
