"""Paper results table, generated ONLY from eval outputs (no hand-typed numbers).

Label convention for eval dirs:  <arm>__s<seed>__<condition>
  e.g. rehearsal25_7b__s1__paper, base_7b__s0__paper-shuffle
Each dir may hold shop.json (eval_v2) and mmlu/ + ifeval/ (lm-eval).

    python scripts/paper/make_table.py /scratch/madan.pa/paper/eval \
        --csv paper/results/results.csv --tex paper/results/table.tex

Mean ± sample std over seeds; with one seed the cell shows the value and n=1
(never a fabricated std). Wilson 95% CI on per-run success is in the CSV.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
from collections import defaultdict

SHOP_KEYS = ["task_success_rate", "optimal_rate", "mean_regret_usd",
             "permission_violation_rate", "invalid_action_rate"]


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def _lm_eval_metric(path: str, task: str, metric: str) -> float | None:
    files = sorted(glob.glob(os.path.join(path, "**", "results*.json"),
                             recursive=True))
    if not files:
        return None
    res = json.load(open(files[-1]))["results"].get(task, {})
    for k, v in res.items():
        if k.split(",")[0] == metric:
            return float(v)
    return None


def load_runs(root: str) -> list[dict]:
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "*__s*__*"))):
        arm, seed, cond = os.path.basename(d).split("__")
        row = {"arm": arm, "seed": int(seed[1:]), "condition": cond, "dir": d}
        shop = os.path.join(d, "shop.json")
        if os.path.exists(shop):
            rep = json.load(open(shop))["reports"][0]
            for k in SHOP_KEYS:
                row[k] = rep.get(k)
            row["n"] = rep["n"]
            row["success_ci95"] = wilson(rep["task_success_rate"], rep["n"])
        row["mmlu_acc"] = _lm_eval_metric(os.path.join(d, "mmlu"), "mmlu", "acc")
        row["ifeval_prompt_strict"] = _lm_eval_metric(
            os.path.join(d, "ifeval"), "ifeval", "prompt_level_strict_acc")
        rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["arm"], r["condition"])].append(r)
    out = []
    for (arm, cond), rs in sorted(groups.items()):
        agg = {"arm": arm, "condition": cond, "seeds": len(rs)}
        for k in SHOP_KEYS + ["mmlu_acc", "ifeval_prompt_strict"]:
            vals = [r[k] for r in rs if r.get(k) is not None]
            agg[k] = (statistics.mean(vals) if vals else None,
                      statistics.stdev(vals) if len(vals) > 1 else None)
        out.append(agg)
    return out


def _cell(m, s, pct=True) -> str:
    if m is None:
        return "TODO"
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.2f}")
    return f"{f(m)} $\\pm$ {f(s)}" if s is not None else f"{f(m)}"


def to_tex(aggs: list[dict]) -> str:
    cols = [("task_success_rate", "Success", True),
            ("mean_regret_usd", "Regret (\\$)", False),
            ("permission_violation_rate", "Viol.", True),
            ("mmlu_acc", "MMLU", True),
            ("ifeval_prompt_strict", "IFEval", True)]
    lines = ["\\begin{tabular}{ll" + "c" * (len(cols) + 1) + "}", "\\toprule",
             "Arm & Cond. & Seeds & " + " & ".join(c[1] for c in cols)
             + " \\\\", "\\midrule"]
    for a in aggs:
        cells = [_cell(*a[k], pct=p) for k, _, p in cols]
        lines.append(f"{a['arm'].replace('_', '-')} & {a['condition']} & "
                     f"{a['seeds']} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--csv", default="paper/results/results.csv")
    ap.add_argument("--tex", default="paper/results/table.tex")
    args = ap.parse_args()
    rows = load_runs(args.root)
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(args.tex, "w") as f:
        f.write(to_tex(aggregate(rows)))
    print(f"[table] {len(rows)} runs -> {args.csv}, {args.tex}")


if __name__ == "__main__":
    main()
