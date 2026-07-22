"""General-chat RETENTION eval (H3 metric).

Asks the fixed held-out GENERAL_EVAL questions under the chat system prompt and
scores the fraction answered as real conversation (prose, no shopping-action
JSON). Run on the base 7B and on each trained arm/stage:

    python scripts/eval_retention.py --label base
    python scripts/eval_retention.py --label sftA --adapter /scratch/.../sft7b_A/policy
"""
import argparse
import json

import torch

from shoprl.data.general_chat import GENERAL_EVAL, answered_generally
from shoprl.data.prompts_v2 import SYSTEM_PROMPT_CHAT_MIN
from shoprl.profiling.bench_common import load_hf_policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA dir; omit for base")
    ap.add_argument("--label", required=True, help="base | sftA | sftB | rlA | rlB")
    ap.add_argument("--max-new-tokens", type=int, default=90)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, tok = load_hf_policy(args.model, args.adapter)

    @torch.no_grad()
    def ask(q: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT_CHAT_MIN},
                {"role": "user", "content": q}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt")["input_ids"].to("cuda")
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    rows, by_lang = [], {}
    for item in GENERAL_EVAL:
        a = ask(item["q"])
        ok = answered_generally(a)
        rows.append({"q": item["q"], "lang": item["lang"], "retained": ok, "a": a})
        d = by_lang.setdefault(item["lang"], [0, 0]); d[0] += int(ok); d[1] += 1

    n = len(rows); kept = sum(r["retained"] for r in rows)
    rec = {
        "label": args.label, "model": args.model, "adapter": args.adapter,
        "retention_pct": round(100 * kept / n, 1), "n": n, "retained": kept,
        "by_lang": {k: round(100 * v[0] / v[1], 1) for k, v in by_lang.items()},
        "forgot_examples": [{"q": r["q"], "a": r["a"][:120]}
                            for r in rows if not r["retained"]][:6],
        "kept_examples": [{"q": r["q"], "a": r["a"][:120]}
                          for r in rows if r["retained"]][:4],
    }
    out = args.out or f"/scratch/madan.pa/pennypilot/retention_{args.label}.json"
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[retention:{args.label}] {rec['retention_pct']}% "
          f"({kept}/{n}) by_lang={rec['by_lang']} -> {out}")
    if rec["forgot_examples"]:
        print("  FORGOT (answered a general question with a shopping action):")
        for e in rec["forgot_examples"][:3]:
            print(f"    Q: {e['q']}\n     A: {e['a']}")


if __name__ == "__main__":
    main()
