"""Results table is generated from eval files only; missing values say TODO."""
import importlib.util
import json
import os

spec = importlib.util.spec_from_file_location(
    "make_table", os.path.join(os.path.dirname(__file__), "..", "scripts",
                               "paper", "make_table.py"))
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)


def _shop(d, success):
    os.makedirs(d)
    json.dump({"reports": [{"n": 128, "task_success_rate": success,
                            "optimal_rate": success, "mean_regret_usd": 0.0,
                            "permission_violation_rate": 0.0,
                            "invalid_action_rate": 0.0}]},
              open(os.path.join(d, "shop.json"), "w"))


def test_aggregate_seeds_and_todo(tmp_path):
    _shop(tmp_path / "armA__s0__paper", 0.5)
    _shop(tmp_path / "armA__s1__paper", 0.7)
    _shop(tmp_path / "base__s0__paper", 0.1)
    rows = mt.load_runs(str(tmp_path))
    aggs = {a["arm"]: a for a in mt.aggregate(rows)}
    m, s = aggs["armA"]["task_success_rate"]
    assert abs(m - 0.6) < 1e-9 and s is not None
    assert aggs["base"]["task_success_rate"][1] is None   # n=1: no fake std
    tex = mt.to_tex(mt.aggregate(rows))
    assert "TODO" in tex                                 # no MMLU yet
    assert "60.0 $\\pm$" in tex


def test_wilson():
    lo, hi = mt.wilson(118 / 128, 128)
    assert 0.85 < lo < 0.87 and 0.95 < hi < 0.97
