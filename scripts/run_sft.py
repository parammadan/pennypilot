"""Step 4: bounded SFT warmup + grammar check.

Generates the demo set (data/sft.py), SFTs the 1.5B (loss masked to agent turns),
then measures the Step-4 success signal: the fraction of the model's generated
actions that are WELL-FORMED grammar on held-out scenarios (not reward).

    python scripts/run_sft.py --model Qwen/Qwen2.5-1.5B-Instruct --method full \
        --n-demos 1000 --max-len 1024 --max-turns 10 --batch-size 4 \
        --max-steps 200 --out-dir /scratch/madan.pa/pennywise/sft1
"""
from __future__ import annotations

import argparse
import json
import os
import re

import torch

from shoprl.data.catalog import catalog_index, generate_catalog
from shoprl.data.sft import generate_sft_dialogues
from shoprl.env.pennyenv import PennyEnv
from shoprl.env.scenario import generate_scenarios
from shoprl.train.build import build_model
from shoprl.train.sft import train_sft

_GRAMMAR = re.compile(
    r"^(ASK\[(budget|feature)\]|SEARCH|RECOMMEND\[LAP-\d{4}\]|ASK_PERMISSION|ADD_TO_CART\[LAP-\d{4}\])$")


@torch.no_grad()
def _gen_action(model, tok, messages, device, max_new_tokens=24) -> str:
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  tokenize=True, return_dict=True,
                                  return_tensors="pt")
    ids = enc["input_ids"].to(device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def grammar_check(built, catalog, idx, n=40, seed=1000, max_turns=10) -> dict:
    """Drive held-out episodes with the model; report well-formed-action rate.

    Training leaves use_cache=False + gradient checkpointing on, which makes
    generation pathologically slow (no KV cache). Toggle both for the duration
    of the check, then restore the training config."""
    model, tok = built.policy, built.tokenizer
    model.eval()
    model.config.use_cache = True
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    skus = set(idx)
    scen = generate_scenarios(catalog, n=n, seed=seed)
    total, wellformed, samples = 0, 0, []
    for s in scen:
        env = PennyEnv(catalog, s, idx=idx, max_turns=max_turns)
        opener = env.reset()
        messages = [{"role": "user", "content": opener}]
        done = False
        while not done:
            raw = _gen_action(model, tok, messages, built.device)
            action = raw.splitlines()[0].strip() if raw else ""
            total += 1
            ok = bool(_GRAMMAR.match(action))
            if ok:
                m = re.search(r"\[(LAP-\d{4})\]", action)
                ok = (m is None) or (m.group(1) in skus)
            wellformed += int(ok)
            if len(samples) < 8:
                samples.append({"action": action, "ok": ok})
            user, done, _ = env.step(action)
            messages += [{"role": "assistant", "content": action},
                         {"role": "user", "content": user}]
    # Restore training config (gradient checkpointing on, cache off).
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        pass
    model.train()
    return {"actions": total, "wellformed": wellformed,
            "wellformed_rate": round(wellformed / max(total, 1), 4),
            "samples": samples}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--method", default="full", choices=["full", "lora"])
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--n-demos", type=int, default=1000)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--catalog-size", type=int, default=500)
    ap.add_argument("--out-dir", default="runs/sft")
    args = ap.parse_args()

    catalog = generate_catalog(n=args.catalog_size, seed=0)
    idx = catalog_index(catalog)
    demos = generate_sft_dialogues(catalog, n=args.n_demos, seed=0)
    print(f"[sft] {len(demos)} demos | model={args.model} method={args.method}")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    built = build_model(args.model, method=args.method, dtype=dtype,
                        device="cuda", grad_checkpointing=True)
    print(f"[sft] dtype={args.dtype} trainable={built.trainable_params():,}"
          f"/{built.total_params():,}")

    # Pre-SFT grammar baseline (instruct model, no warmup yet).
    pre = grammar_check(built, catalog, idx, n=20, max_turns=args.max_turns)
    print(f"[sft] pre-SFT well-formed rate: {pre['wellformed_rate']} "
          f"({pre['wellformed']}/{pre['actions']})")

    import math
    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "sft_metrics.jsonl")
    diverged = False
    with open(metrics_path, "w") as f:
        for m in train_sft(built, demos, max_len=args.max_len,
                           batch_size=args.batch_size, epochs=args.epochs,
                           max_steps=args.max_steps, lr=args.lr):
            f.write(json.dumps(m) + "\n")
            if m["step"] % 10 == 0 or m["step"] == 1:
                print(f"  step {m['step']:>4} | loss {m['loss']:.4f} "
                      f"| sup_tokens {m['supervised_tokens']}")
            if not math.isfinite(m["loss"]):
                print(f"[sft] DIVERGED: non-finite loss at step {m['step']} "
                      f"(dtype={args.dtype}). Stopping — not a stable config.")
                diverged = True
                break
    if diverged:
        print("[sft] skipping post-SFT grammar check + save (diverged run).")
        return

    post = grammar_check(built, catalog, idx, n=40, max_turns=args.max_turns)
    print(f"[sft] post-SFT well-formed rate: {post['wellformed_rate']} "
          f"({post['wellformed']}/{post['actions']})")
    for s in post["samples"]:
        print(f"    {'OK ' if s['ok'] else 'BAD'} {s['action']!r}")

    ckpt = os.path.join(args.out_dir, "policy")
    built.policy.save_pretrained(ckpt)
    built.tokenizer.save_pretrained(ckpt)
    with open(os.path.join(args.out_dir, "sft_results.json"), "w") as f:
        json.dump({"model": args.model, "method": args.method,
                   "n_demos": len(demos), "max_len": args.max_len,
                   "max_turns": args.max_turns, "pre": pre, "post": post}, f, indent=2)
    print(f"[sft] saved -> {ckpt}")


if __name__ == "__main__":
    main()
