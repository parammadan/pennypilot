"""Multi-turn RLOO run: load the SFT policy, optimize with leave-one-out RLOO
against the simulator, log per-step metrics.

    # 10-step smoke (does it run? reward compute? KL log?)
    python scripts/run_rloo.py --ckpt /scratch/madan.pa/pennywise/sft_ground/policy \
        --steps 10 --k 8 --out-dir /scratch/madan.pa/pennywise/rloo_smoke

    # 50-step observed
    python scripts/run_rloo.py --ckpt .../policy --steps 50 --k 8 \
        --out-dir /scratch/madan.pa/pennywise/rloo50
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from shoprl.data.catalog import catalog_index, generate_catalog
from shoprl.env.scenario import generate_scenarios
from shoprl.train.build import build_model
from shoprl.train.rloo import RLOOConfig, rloo_step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="SFT checkpoint (policy + ref init)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--catalog-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2000, help="RLOO prompt seed (disjoint from SFT=0, eval=1000)")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--out-dir", default="runs/rloo")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    built = build_model(args.ckpt, method="full", dtype=dtype, device="cuda",
                        grad_checkpointing=True)
    policy, reference, tok = built.policy, built.reference, built.tokenizer
    print(f"[rloo] policy+ref from {args.ckpt} | dtype={args.dtype} "
          f"k={args.k} steps={args.steps}")

    catalog = generate_catalog(n=args.catalog_size, seed=0)
    idx = catalog_index(catalog)
    # A pool of RLOO training scenarios (disjoint seed from SFT + eval).
    pool = generate_scenarios(catalog, n=max(64, args.steps * args.prompts_per_step),
                              seed=args.seed)

    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad), lr=args.lr)
    cfg = RLOOConfig(k=args.k, prompts_per_step=args.prompts_per_step,
                     max_turns=args.max_turns, max_len=args.max_len,
                     temperature=args.temperature, beta=args.beta, lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    mpath = os.path.join(args.out_dir, "metrics.jsonl")
    open(mpath, "w").close()

    for step in range(args.steps):
        picks = [pool[(step * args.prompts_per_step + j) % len(pool)]
                 for j in range(args.prompts_per_step)]
        m = rloo_step(policy, reference, tok, optimizer, catalog, picks, idx,
                      "cuda", cfg)
        m["step"] = step
        with open(mpath, "a") as f:
            f.write(json.dumps(m) + "\n")
        print(f"step {step:>3} | reward {m['reward_mean']:+.3f}±{m['reward_std']:.3f} "
              f"| adv|.| {m['adv_abs_mean']:.3f} | KL {m['kl_mean']:.5f} "
              f"| value {m['value_mean']:.3f} | ask {m['ask_rate']:.2f} "
              f"| viol {m['violation_rate']:.2f} | gnorm {m['grad_norm']:.3f}")
        if (step + 1) % args.save_every == 0 and (step + 1) < args.steps:
            built.policy.save_pretrained(os.path.join(args.out_dir, f"step-{step+1}"))

    built.policy.save_pretrained(os.path.join(args.out_dir, "policy"))
    built.tokenizer.save_pretrained(os.path.join(args.out_dir, "policy"))
    print(f"[rloo] done -> {args.out_dir}/policy")


if __name__ == "__main__":
    main()
