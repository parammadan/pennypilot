"""Seed-level tests between arms, from results.csv only (no hand-typed numbers).

Welch t-test and an exact two-sided permutation test (3 vs 3 seeds => 20
relabelings, so the smallest attainable p is 0.1) on per-seed metrics.

    python scripts/paper/stats.py paper/results/results.csv --out paper/results/stats.json
"""
import argparse
import csv
import itertools
import json
import statistics

from scipy import stats

METRICS = ["task_success_rate", "retention_40q", "ifeval_prompt_strict", "mmlu_acc"]


def perm_p(a: list[float], b: list[float]) -> float:
    obs = abs(statistics.mean(a) - statistics.mean(b))
    pool, n = a + b, len(a)
    hits = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        x = [pool[i] for i in idx]
        y = [pool[i] for i in range(len(pool)) if i not in idx]
        hits += abs(statistics.mean(x) - statistics.mean(y)) >= obs - 1e-12
        tot += 1
    return hits / tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="paper/results/stats.json")
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(args.csv)) if r["condition"] == "paper"]
    out = {}
    for stage in ("sft", "rl"):
        for m in METRICS:
            a = [float(r[m]) for r in rows if r["arm"] == f"specialist-{stage}"]
            b = [float(r[m]) for r in rows if r["arm"] == f"rehearsal25-{stage}"]
            t = stats.ttest_ind(a, b, equal_var=False)
            out[f"{stage}/{m}"] = {
                "specialist": a, "rehearsal25": b,
                "diff_mean": statistics.mean(b) - statistics.mean(a),
                "welch_t": float(t.statistic), "welch_p": float(t.pvalue),
                "perm_p_exact": perm_p(a, b)}
    json.dump(out, open(args.out, "w"), indent=2)
    for k, v in out.items():
        print(f"{k:28s} diff={100*v['diff_mean']:+6.1f}  welch_p={v['welch_p']:.4f}  perm_p={v['perm_p_exact']:.2f}")


if __name__ == "__main__":
    main()
