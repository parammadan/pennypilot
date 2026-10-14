"""Matched human-written control for the self-distilled rehearsal set.

Same prompts, same order as rehearsal_dolly.jsonl (so --rehearsal-n N picks the
SAME N prompts in both arms); only the answer changes: Dolly-15k's human
response instead of the base model's own. Isolates "self-distilled" from
"any general data".

    python scripts/paper/make_human_rehearsal.py \
        --distilled $SCR/rehearsal_dolly.jsonl \
        --prompts src/shoprl/data/rehearsal_prompts_dolly900.jsonl \
        --out $SCR/rehearsal_dolly_human.jsonl
"""
import argparse
import json

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distilled", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    idx = {}
    for l in open(args.prompts):
        p = json.loads(l)
        idx.setdefault(p["q"], p["dolly_idx"])
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    n = 0
    with open(args.out, "w") as f:
        for l in open(args.distilled):
            r = json.loads(l)
            row = ds[idx[r["q"]]]
            assert row["instruction"].strip() == r["q"], r["q"]
            f.write(json.dumps({"q": r["q"], "a": row["response"].strip(),
                                "lang": r.get("lang", "en"),
                                "dolly_idx": idx[r["q"]]}) + "\n")
            n += 1
    print(f"[human-reh] {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
