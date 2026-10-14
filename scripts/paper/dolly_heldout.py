"""200 held-out off-task Dolly prompts for the tool-decision eval.

Same eligibility filter and shuffle as dolly_select.py; takes positions
900..1099 of the shuffled pool, so it is disjoint from the 900 rehearsal
prompts by construction (asserted below).
"""
import json
import random
import re
import sys

from datasets import load_dataset

ds = load_dataset("databricks/databricks-dolly-15k", split="train")
SHOP = re.compile(r"\b(laptop|laptops|buy|buying|purchase|price|prices|cheap|cheapest|shop|shopping|cart|store|budget|deal|discount|cost)\b", re.I)
keep_cats = {"open_qa", "general_qa", "brainstorming", "creative_writing"}
seen, uniq = set(), []
for i, r in enumerate(ds):
    q = r["instruction"].strip()
    if r["category"] not in keep_cats or r["context"].strip():
        continue
    if not (15 <= len(q) <= 300) or SHOP.search(q) or q.lower() in seen:
        continue
    seen.add(q.lower())
    uniq.append({"q": q, "lang": "en", "dolly_idx": i, "category": r["category"]})
random.Random("dolly-rehearsal-0").shuffle(uniq)
reh = {json.loads(l)["dolly_idx"] for l in open(sys.argv[1])}
sel = uniq[900:1100]
assert {p["dolly_idx"] for p in uniq[:900]} == reh, "shuffle does not reproduce rehearsal set"
assert not reh & {p["dolly_idx"] for p in sel}
with open(sys.argv[2], "w") as f:
    for p in sel:
        f.write(json.dumps(p) + "\n")
print("held-out", len(sel), "disjoint from", len(reh))
