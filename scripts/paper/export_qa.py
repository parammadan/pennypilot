"""Export the answer-correctness sets used by eval_behavior.py (CPU, one-off).

  TriviaQA rc.nocontext validation: 300 questions (seeded sample), all gold
      aliases kept -> scored as "any normalized alias appears in the reply".
  MGSM (6 languages x 50): grade-school math in es, fr, de, ru, zh, ja ->
      scored as "last number in the reply equals the gold answer".

    python scripts/paper/export_qa.py src/shoprl/data/qa_correctness.jsonl
"""
import json
import random
import sys

from datasets import load_dataset

rows = []
tq = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
for i in random.Random("triviaqa-0").sample(range(len(tq)), 300):
    r = tq[i]
    rows.append({"set": "triviaqa", "lang": "en", "q": r["question"],
                 "gold": sorted(set(r["answer"]["normalized_aliases"]))})
for lang in ["es", "fr", "de", "ru", "zh", "ja"]:
    m = load_dataset("juletxara/mgsm", lang, split="test")
    for r in list(m)[:50]:
        rows.append({"set": "mgsm", "lang": lang, "q": r["question"],
                     "gold": [str(r["answer_number"])]})
with open(sys.argv[1], "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), "rows")
