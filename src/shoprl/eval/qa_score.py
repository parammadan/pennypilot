"""Answer-correctness scoring for free-form chat replies (paper eval A).

TriviaQA: any normalized gold alias occurs as a whole-word span of the
normalized reply (SQuAD-style normalization) -- tolerant of chatty answers.
MGSM: the last number in the reply equals the gold integer.
"""
from __future__ import annotations

import re
import string


def _norm(t: str) -> str:
    t = t.lower()
    t = "".join(c for c in t if c not in set(string.punctuation))
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    return " ".join(t.split())


def correct(row: dict, reply: str) -> bool:
    if row["set"] == "mgsm":
        nums = re.findall(r"-?\d[\d,]*\.?\d*", reply)
        if not nums:
            return False
        try:
            return float(nums[-1].replace(",", "").rstrip(".")) == float(row["gold"][0])
        except ValueError:
            return False
    r = f" {_norm(reply)} "
    return any(f" {g} " in r for g in (_norm(x) for x in row["gold"]) if g)
