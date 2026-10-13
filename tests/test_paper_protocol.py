"""Paper protocol: contamination-free splits, shuffled search, $ metrics, seeds."""
import json
import random
import re

import pytest

from shoprl.data.catalog import catalog_index, generate_catalog
from shoprl.env.catalog_env import SyntheticCatalogEnvironment
from shoprl.env.scenario import generate_hard_scenarios
from shoprl.eval.harness_v2 import evaluate_v2
from shoprl.eval.splits import (EVAL_SPLIT, clean_eval_scenarios, need_key,
                                overlap, paper_eval_split, training_pools)
from shoprl.eval.v2_policies import OracleGoodV2
from shoprl.repro import seed_everything, write_manifest


@pytest.fixture(scope="module")
def catalog():
    return generate_catalog(n=300, seed=0)


def test_registered_split_is_contaminated(catalog):
    # the audit finding this module exists for — if it ever reads 0, the
    # generators changed and the paper's contamination numbers must be redone
    reg = generate_hard_scenarios(catalog, n=128, **EVAL_SPLIT)
    rep = overlap(reg, training_pools(catalog))
    assert (rep.need_dups, rep.answer_dups) == (39, 52)


def test_clean_split_disjoint(catalog):
    clean = clean_eval_scenarios(catalog, n=128)
    assert len(clean) == 128
    assert overlap(clean, training_pools(catalog)).clean()
    assert len({need_key(s) for s in clean}) == 128      # no internal dups


def test_paper_split_heldout_catalog(catalog):
    cat1, scen = paper_eval_split(n=128)
    assert len(scen) == 128
    train_needs = {need_key(s) for s in training_pools(catalog)}
    assert not any(need_key(s) in train_needs for s in scen)
    assert all(p != q for p, q in zip(cat1, catalog))    # every spec unseen
    with pytest.raises(ValueError):
        paper_eval_split(n=8, catalog_seed=0)


def _search_skus(env):
    obs = env.execute_text(json.dumps({"action": "search", "query": "laptop"}))
    return re.findall(r"[A-Z]{3}-\d{4}", obs.observation)


def test_shuffle_search_same_set_different_order(catalog):
    idx = catalog_index(catalog)
    changed = 0
    for s in generate_hard_scenarios(catalog, n=20, **EVAL_SPLIT):
        runs = []
        for shuffle in (False, True):
            env = SyntheticCatalogEnvironment(catalog, s, idx=idx,
                                              shuffle_search=shuffle)
            env.reset()
            runs.append(_search_skus(env))
        plain, shuf = runs
        assert sorted(plain) == sorted(shuf)
        assert [idx[k].price for k in plain] == sorted(idx[k].price for k in plain)
        changed += plain != shuf
    assert changed > 0


def test_dollar_metrics_oracle(catalog):
    scen = generate_hard_scenarios(catalog, n=20, **EVAL_SPLIT)
    r = evaluate_v2(catalog, scen, OracleGoodV2, "oracle")
    assert r.optimal_rate == 1.0
    assert r.mean_regret_usd == 0.0
    assert r.mean_savings_vs_median_usd >= 0.0


def test_seed_and_manifest(tmp_path):
    seed_everything(7)
    a = random.random()
    seed_everything(7)
    assert random.random() == a

    import argparse
    path = write_manifest(str(tmp_path), argparse.Namespace(lr=1e-6), note="x")
    man = json.load(open(path))
    assert man["args"] == {"lr": 1e-6}
    assert man["note"] == "x" and "git_sha" in man and "versions" in man
