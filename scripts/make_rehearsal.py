"""Self-distill general-chat rehearsal data from the BASE 7B (H3 Arm B).

Answers the templated general questions with the untrained base model (plain
helpful-assistant prompt -> guaranteed clean prose), keeps only real general
answers (no JSON), and writes {q, a, lang} JSONL to mix into Arm B's SFT.

    python scripts/make_rehearsal.py --n 180 --out /scratch/.../rehearsal.jsonl

Paper recipe: --prompts-file src/shoprl/data/rehearsal_prompts_dolly900.jsonl
(Dolly-15k open prompts, CC BY-SA 3.0, selected by scripts/paper/dolly_select.py;
disjoint in template from the GENERAL_EVAL retention set). Answers that hit
--max-new-tokens without EOS are DROPPED: a truncated target would teach the
model to stop mid-sentence.
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
    ap.add_argument("--prompts-file", default=None,
                    help="JSONL of {q, lang}; replaces the templated pool")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--out", default="/scratch/madan.pa/pennypilot/rehearsal.jsonl")
    args = ap.parse_args()

    model, tok = load_hf_policy(args.model, None)
    if args.prompts_file:
        qs = [json.loads(l) for l in open(args.prompts_file) if l.strip()][:args.n]
    else:
        qs = rehearsal_questions(n=args.n, seed=args.seed)
    tok.padding_side = "left"                 # batched decoder-only generation
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}

    @torch.no_grad()
    def answer(batch: list[str]) -> list[tuple[str, bool]]:
        texts = [tok.apply_chat_template(
                    [{"role": "system", "content": PLAIN},
                     {"role": "user", "content": q}],
                    add_generation_prompt=True, tokenize=False) for q in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        res = []
        for row in out[:, enc["input_ids"].shape[1]:]:
            finished = any(int(t) in eos for t in row)
            res.append((tok.decode(row, skip_special_tokens=True).strip(), finished))
        return res

    kept, dropped, truncated = [], 0, 0
    for i in range(0, len(qs), args.batch_size):
        chunk = qs[i:i + args.batch_size]
        for item, (a, finished) in zip(chunk, answer([x["q"] for x in chunk])):
            if not finished:
                truncated += 1
            elif answered_generally(a):
                kept.append({"q": item["q"], "a": a, "lang": item.get("lang", "en")})
            else:
                dropped += 1
        print(f"  {min(i + args.batch_size, len(qs))}/{len(qs)} done (kept "
              f"{len(kept)}, dropped {dropped}, truncated {truncated})")

    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"[rehearsal] wrote {len(kept)} exemplars -> {args.out} "
          f"(dropped {dropped} not clean prose, {truncated} truncated)")
    for r in kept[:3]:
        print(f"  e.g. Q: {r['q']}\n       A: {r['a'][:90]}")


if __name__ == "__main__":
    main()
