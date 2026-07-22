"""Self-distill general-chat rehearsal data from the BASE 7B (H3 Arm B).

Answers the templated general questions with the untrained base model (plain
helpful-assistant prompt -> guaranteed clean prose), keeps only real general
answers (no JSON), and writes {q, a, lang} JSONL to mix into Arm B's SFT.

    python scripts/make_rehearsal.py --n 180 --out /scratch/.../rehearsal.jsonl
"""
import argparse
import json

import torch

from shoprl.data.general_chat import answered_generally, rehearsal_questions
from shoprl.profiling.bench_common import load_hf_policy

PLAIN = "You are a helpful, friendly assistant. Answer concisely and warmly."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=180)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=110)
    ap.add_argument("--out", default="/scratch/madan.pa/pennypilot/rehearsal.jsonl")
    args = ap.parse_args()

    model, tok = load_hf_policy(args.model, None)
    qs = rehearsal_questions(n=args.n, seed=args.seed)

    @torch.no_grad()
    def answer(q: str) -> str:
        msgs = [{"role": "system", "content": PLAIN}, {"role": "user", "content": q}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt")["input_ids"].to("cuda")
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    kept, dropped = [], 0
    for i, item in enumerate(qs):
        a = answer(item["q"])
        if answered_generally(a):
            kept.append({"q": item["q"], "a": a, "lang": item["lang"]})
        else:
            dropped += 1
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(qs)} done (kept {len(kept)}, dropped {dropped})")

    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"[rehearsal] wrote {len(kept)} exemplars -> {args.out} "
          f"(dropped {dropped} that weren't clean prose)")
    for r in kept[:3]:
        print(f"  e.g. Q: {r['q']}\n       A: {r['a'][:90]}")


if __name__ == "__main__":
    main()
