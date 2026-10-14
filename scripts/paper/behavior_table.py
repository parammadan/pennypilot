"""Aggregate behavior.json (eval_behavior.py) over seeds -> paper/results/behavior.json.

    python scripts/paper/behavior_table.py paper/results/raw --out paper/results/behavior.json

Only the off-task / correctness / LoRA-sweep measurements are aggregated. The
on-task tool rate is NOT reported: all 128 on-task openers are the same string
(env.reset() opener), so it is one prompt repeated, not 128 measurements.
Mean and sample std over seeds; Welch t-test specialist vs rehearsal25 per stage.
"""
import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

from scipy import stats as st


def metrics(r: dict) -> dict:
    p, c = r["prompt_probes"], r["correctness"]
    m = {f"offtask_tool_{k}": v["tool_rate_all"] for k, v in p.items()}
    for cond in ("deploy", "none"):
        m[f"triviaqa_{cond}"] = c[cond]["triviaqa"]["accuracy"]
        m[f"mgsm_{cond}"] = c[cond]["mgsm_all"]["accuracy"]
        m[f"qa_tool_{cond}"] = ((c[cond]["triviaqa"]["tool"] + sum(
            v["tool"] for k, v in c[cond].items() if k.startswith("mgsm_") and k != "mgsm_all"))
            / (c[cond]["triviaqa"]["n"] + sum(
                v["n"] for k, v in c[cond].items() if k.startswith("mgsm_") and k != "mgsm_all")))
    for a, v in r["lora_sweep_tool_rate_gen40"].items():
        m[f"sweep_{a}"] = v
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="paper/results/behavior.json")
    args = ap.parse_args()
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.root, "*__s*__*", "behavior", "behavior.json"))):
        arm, seed, _ = os.path.basename(os.path.dirname(os.path.dirname(f))).split("__")
        groups[arm].append((int(seed[1:]), metrics(json.load(open(f)))))
    out = {"groups": {}, "welch": {}}
    for arm, rs in sorted(groups.items()):
        keys = rs[0][1].keys()
        out["groups"][arm] = {"seeds": [s for s, _ in rs], **{
            k: {"mean": statistics.mean(m[k] for _, m in rs),
                "std": statistics.stdev(m[k] for _, m in rs) if len(rs) > 1 else None,
                "per_seed": [m[k] for _, m in rs]} for k in keys}}
    for stage in ("sft", "rl"):
        a, b = out["groups"].get(f"specialist-{stage}"), out["groups"].get(f"rehearsal25-{stage}")
        if a and b:
            out["welch"][stage] = {k: st.ttest_ind(a[k]["per_seed"], b[k]["per_seed"],
                                                   equal_var=False).pvalue
                                   for k in ("offtask_tool_deploy", "triviaqa_deploy", "mgsm_deploy")}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    for arm, g in out["groups"].items():
        print(arm, g["seeds"], {k: round(100 * v["mean"], 1) for k, v in g.items() if k != "seeds"})
    print("welch", out["welch"])


if __name__ == "__main__":
    main()
