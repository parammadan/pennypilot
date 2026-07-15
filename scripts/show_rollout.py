"""Load a trained policy and show full held-out rollouts + behaviour metrics.

Grammar-valid actions are necessary but not sufficient; this checks the BEHAVIOUR:
does the model ask, discover the hidden need, recommend the *cheapest valid*
item, and ask permission before adding? Prints full dialogues and reports the
mean value_quality / permission-violation / accept over held-out scenarios.

    python scripts/show_rollout.py --ckpt /scratch/madan.pa/pennywise/sft_bf16/policy \
        --n 30 --show 3 --max-turns 12
"""
from __future__ import annotations

import argparse
import re
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shoprl.data.catalog import catalog_index, generate_catalog
from shoprl.env.pennyenv import PennyEnv
from shoprl.env.reward import value_quality
from shoprl.env.scenario import generate_scenarios


@torch.no_grad()
def gen_action(model, tok, messages, device, max_new_tokens=24) -> str:
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  tokenize=True, return_dict=True,
                                  return_tensors="pt")
    ids = enc["input_ids"].to(device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    return txt.splitlines()[0].strip() if txt else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--catalog-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, dtype=torch.bfloat16).to("cuda")
    model.eval()
    model.config.use_cache = True

    cat = generate_catalog(n=args.catalog_size, seed=0)
    idx = catalog_index(cat)
    scen = generate_scenarios(cat, n=args.n, seed=args.seed)  # held-out seed

    vqs, viols, accepts, asked, cheapest_hits = [], [], [], [], []
    for k, s in enumerate(scen):
        env = PennyEnv(cat, s, idx=idx, max_turns=args.max_turns)
        opener = env.reset()
        messages = [{"role": "user", "content": opener}]
        transcript = [("user", opener)]
        done = False
        while not done:
            action = gen_action(model, tok, messages, "cuda")
            user, done, _ = env.step(action)
            transcript += [("agent", action), ("user", user)]
            messages += [{"role": "assistant", "content": action},
                         {"role": "user", "content": user}]
        r = env.reward()
        cheapest = min(s.valid_skus, key=lambda x: idx[x].price)
        added = env.state.cart[-1] if env.state.cart else None
        vqs.append(r.value_quality)
        viols.append(r.acted_without_permission)
        accepts.append(r.accepted)
        asked.append(1.0 if any(t.action == "ASK" for t in env.turns) else 0.0)
        cheapest_hits.append(1.0 if added == cheapest else 0.0)

        if k < args.show:
            print(f"\n===== held-out scenario {s.scenario_id} =====")
            print(f"HIDDEN budget=${s.hidden_budget:.0f} must_have {s.must_have_key}="
                  f"{s.must_have_value} | |valid|={len(s.valid_skus)} "
                  f"cheapest_valid={cheapest}@${idx[cheapest].price:.0f}")
            for role, text in transcript:
                print(f"  {'AGENT' if role=='agent' else 'user '}: {text}")
            print(f"  -> added={added} value_quality={r.value_quality} "
                  f"accepted={r.accepted} violation={r.acted_without_permission} "
                  f"total={r.total:+.3f}")

    m = statistics.mean
    print(f"\n===== BEHAVIOUR over {len(scen)} held-out scenarios =====")
    print(f"ask_rate            {m(asked):.3f}")
    print(f"violation_rate      {m(viols):.3f}")
    print(f"accept_rate         {m(accepts):.3f}")
    print(f"mean value_quality  {m(vqs):.3f}")
    print(f"cheapest-valid hit  {m(cheapest_hits):.3f}  (added the single cheapest valid item)")


if __name__ == "__main__":
    main()
