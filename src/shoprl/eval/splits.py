"""Contamination-checked eval splits (paper protocol).

Distinct scenario SEEDS do not imply distinct scenario CONTENT: SFT, RL and
eval all draw hidden needs over the same catalog (seed 0), so an eval scenario
can repeat a training scenario's exact hidden needs or valid-answer set. The
2026-10-12 audit measured, for the registered eval split (seed 5001, n=128)
vs the SFT+RL pools: 39 exact hidden-need duplicates, 52 shared answer sets.

Two splits:
  * `paper_eval_split` (PRIMARY) — same generator and difficulty knobs on a
    HELD-OUT catalog (seed 1): every product spec is unseen, no hidden-need
    duplicates vs training. SKU ids repeat across catalogs with different
    specs, so a policy that recalls ids instead of reading search results is
    penalized (strict grounding test — documented in the paper).
  * `clean_eval_scenarios` (secondary) — same catalog, overlapping candidates
    dropped. Small answer sets collide most, so this split drifts easier
    (measured mean valid-set 5.12 vs 3.93 registered); report, don't headline.
"""
from __future__ import annotations

from dataclasses import dataclass

from shoprl.data.catalog import Product, generate_catalog
from shoprl.env.scenario import Scenario, generate_hard_scenarios

# Training pools exactly as the trainers build them (run_sft_v2 /
# generate_sft_v2_dialogues: seed 0, n=1000, default difficulty; run_rl_v2:
# seed 3000, pool max(64, steps), RL difficulty).
SFT_POOL = {"n": 1000, "seed": 0, "n_must_haves": (2, 3),
            "valid_target_range": (5, 20)}
RL_POOL = {"n": 64, "seed": 3000, "n_must_haves": (3, 4),
           "valid_target_range": (3, 10)}
TRAIN_CATALOG_SEED = 0
PAPER_CATALOG_SEED = 1
EVAL_SPLIT = {"seed": 5001, "n_must_haves": (3, 4),
              "valid_target_range": (3, 10)}


def need_key(s: Scenario) -> tuple:
    """Exact hidden needs: budget + every must-have."""
    return (s.hidden_budget, tuple(sorted(s.all_must_haves.items())))


def answer_key(s: Scenario) -> frozenset:
    """The verifiable answer set (what a memorizing policy would exploit)."""
    return frozenset(s.valid_skus)


@dataclass
class OverlapReport:
    n_eval: int
    need_dups: int
    answer_dups: int

    def clean(self) -> bool:
        return self.need_dups == 0 and self.answer_dups == 0


def training_pools(catalog: list[Product]) -> list[Scenario]:
    return (generate_hard_scenarios(catalog, **SFT_POOL)
            + generate_hard_scenarios(catalog, **RL_POOL))


def overlap(eval_scenarios: list[Scenario],
            train: list[Scenario]) -> OverlapReport:
    needs = {need_key(s) for s in train}
    answers = {answer_key(s) for s in train}
    return OverlapReport(
        n_eval=len(eval_scenarios),
        need_dups=sum(need_key(s) in needs for s in eval_scenarios),
        answer_dups=sum(answer_key(s) in answers for s in eval_scenarios))


def clean_eval_scenarios(catalog: list[Product], n: int = 128,
                         oversample: int = 4) -> list[Scenario]:
    """`n` eval scenarios (registered seed/difficulty) with no need- or
    answer-set overlap against any training pool, nor within the split."""
    train = training_pools(catalog)
    needs = {need_key(s) for s in train}
    answers = {answer_key(s) for s in train}
    out: list[Scenario] = []
    for s in generate_hard_scenarios(catalog, n=n * oversample, **EVAL_SPLIT):
        if need_key(s) in needs or answer_key(s) in answers:
            continue
        needs.add(need_key(s))
        answers.add(answer_key(s))
        out.append(s)
        if len(out) == n:
            break
    if len(out) < n:
        raise ValueError(f"only {len(out)}/{n} clean scenarios; raise oversample")
    assert overlap(out, train).clean()
    return out


def paper_eval_split(n: int = 128, catalog_size: int = 300,
                     catalog_seed: int = PAPER_CATALOG_SEED
                     ) -> tuple[list[Product], list[Scenario]]:
    """(held-out catalog, n scenarios) — the primary paper eval split."""
    if catalog_seed == TRAIN_CATALOG_SEED:
        raise ValueError("paper split must use a catalog unseen in training")
    train = training_pools(generate_catalog(n=catalog_size,
                                            seed=TRAIN_CATALOG_SEED))
    needs = {need_key(s) for s in train}
    catalog = generate_catalog(n=catalog_size, seed=catalog_seed)
    out = [s for s in generate_hard_scenarios(catalog, n=n * 2, **EVAL_SPLIT)
           if need_key(s) not in needs][:n]
    if len(out) < n:
        raise ValueError(f"only {len(out)}/{n} scenarios after need-dedup")
    return catalog, out
