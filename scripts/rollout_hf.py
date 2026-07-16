"""HF-batched rollout throughput benchmark (guaranteed fallback for Phase 3).

vLLM's continuous batching is the preferred engine, but on a bleeding-edge build
it can fail to serve (flashinfer JIT). This is the reliable fallback: plain
transformers batched generation, no vLLM. Runs B multi-turn PennyEnv trajectories
in lockstep — every turn, all active dialogues are batch-generated together (left
-padded) — and measures trajectories/sec + agent-generations/sec on one GPU.

For 2-GPU (Step 3), run one process per GPU (CUDA_VISIBLE_DEVICES) over disjoint
scenario shards and sum the throughputs — that's the data-parallel number.

    CUDA_VISIBLE_DEVICES=0 python3 scripts/rollout_hf.py --ckpt ~/pennywise/ckpt \
        --n 128 --batch 32
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shoprl.data.catalog import catalog_index, generate_catalog
from shoprl.env.pennyenv import PennyEnv
from shoprl.env.scenario import generate_scenarios


@torch.no_grad()
def run_batch(model, tok, scens, catalog, idx, device, *, max_turns, max_new_tokens):
    envs = [PennyEnv(catalog, s, idx=idx, max_turns=max_turns) for s in scens]
    msgs = [[{"role": "user", "content": e.reset()}] for e in envs]
    done = [False] * len(envs)
    gens = 0
    while not all(done):
        active = [i for i in range(len(envs)) if not done[i]]
        prompts = [tok.apply_chat_template(msgs[i], add_generation_prompt=True,
                                           tokenize=False) for i in active]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=1.0, top_p=0.95, pad_token_id=tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        for j, i in enumerate(active):
            txt = tok.decode(new[j], skip_special_tokens=True).strip()
            action = txt.splitlines()[0].strip() if txt else ""
            gens += 1
            user, d, _ = envs[i].step(action)
            msgs[i] += [{"role": "assistant", "content": action},
                        {"role": "user", "content": user}]
            done[i] = d
    return envs, gens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--catalog-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=3000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for batched decoder-only generation
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, dtype=torch.bfloat16).to("cuda")
    model.eval()

    cat = generate_catalog(n=args.catalog_size, seed=0)
    idx = catalog_index(cat)
    scen = generate_scenarios(cat, n=args.n, seed=args.seed)

    # warmup (compile/allocate) — not timed
    run_batch(model, tok, scen[:min(4, args.n)], cat, idx, "cuda",
              max_turns=args.max_turns, max_new_tokens=args.max_new_tokens)

    vals, total_gens = [], 0
    t0 = time.time()
    for i in range(0, len(scen), args.batch):
        envs, g = run_batch(model, tok, scen[i:i + args.batch], cat, idx, "cuda",
                            max_turns=args.max_turns, max_new_tokens=args.max_new_tokens)
        total_gens += g
        vals += [e.reward().value_quality for e in envs]
    dt = time.time() - t0

    dev = torch.cuda.get_device_name(0)
    print(f"[hf-rollout] {dev} | n={len(scen)} batch={args.batch}")
    print(f"  wall={dt:.1f}s  trajectories/sec={len(scen)/dt:.3f}  "
          f"agent-gens/sec={total_gens/dt:.2f}")
    print(f"  mean_value={statistics.mean(vals):.3f}  (behaviour sanity via HF)")


if __name__ == "__main__":
    main()
