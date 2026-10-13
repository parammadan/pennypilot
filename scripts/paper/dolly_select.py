import json, random, re
from collections import Counter
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
sel = uniq[:900]
print("total", len(ds), "eligible", len(uniq), "selected", len(sel), dict(Counter(p["category"] for p in sel)))
with open("/scratch/madan.pa/paper/rehearsal_prompts_dolly900.jsonl", "w") as f:
    for p in sel:
        f.write(json.dumps(p) + "\n")
