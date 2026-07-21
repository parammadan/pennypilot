"""PennyPilot RL (GPU) — RLOO bring-up (algo swappable behind the seam).

    # tiny-config gate (always first):
    python scripts/run_rl_v2.py --sft-adapter /scratch/madan.pa/pennypilot/sft_v2/policy \
        --steps 10 --k 8 --out-dir /scratch/madan.pa/pennypilot/rl_tiny

Policy AND KL-reference both initialize from the SFT adapter (the reference is
a second frozen copy — with a LoRA policy trained FROM an SFT adapter,
adapter-off would give the BASE model, which is the wrong reference). fp16 +
GradScaler per the pinned recipe; same precision path for both logprob sides.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--algo", default="rloo", choices=["rloo", "grpo", "grpo-nostd"])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--language", default="es-en", choices=["en", "es", "es-en"])
    ap.add_argument("--catalog-size", type=int, default=300)
    ap.add_argument("--scenario-seed", type=int, default=3000,
                    help="disjoint from SFT(0)/gate(11)/eval(5000)")
    ap.add_argument("--n-must-haves", type=int, nargs=2, default=[3, 4])
    ap.add_argument("--valid-range", type=int, nargs=2, default=[3, 10])
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--out-dir", default="runs/rl_v2")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from shoprl.data.catalog import catalog_index, generate_catalog
    from shoprl.data.prompts_v2 import SYSTEM_PROMPT_V2
    from shoprl.env.scenario import generate_hard_scenarios
    from shoprl.train.algo import RLConfigV2, rl_step

    tok = AutoTokenizer.from_pretrained(args.sft_adapter)

    def load(trainable: bool):
        base = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16,
            attn_implementation="sdpa").to("cuda")
        m = PeftModel.from_pretrained(base, args.sft_adapter,
                                      is_trainable=trainable)
        if not trainable:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        return m

    policy = load(trainable=True)
    reference = load(trainable=False)
    for p in policy.parameters():           # fp32 master on trainables (recipe)
        if p.requires_grad:
            p.data = p.data.float()
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[rl-v2] algo={args.algo} k={args.k} steps={args.steps} "
          f"trainable={n_trainable/1e6:.1f}M lang={args.language}")

    catalog = generate_catalog(n=args.catalog_size, seed=0)
    idx = catalog_index(catalog)
    pool = generate_hard_scenarios(
        catalog, n=max(64, args.steps * args.prompts_per_step),
        seed=args.scenario_seed, n_must_haves=tuple(args.n_must_haves),
        valid_target_range=tuple(args.valid_range))
    print(f"[rl-v2] scenario pool: {len(pool)} "
          f"(must-haves {args.n_must_haves}, valid {args.valid_range})")

    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    cfg = RLConfigV2(algo=args.algo, k=args.k,
                     prompts_per_step=args.prompts_per_step,
                     max_turns=args.max_turns,
                     max_new_tokens=args.max_new_tokens, max_len=args.max_len,
                     temperature=args.temperature, beta=args.beta, lr=args.lr,
                     language=args.language)

    os.makedirs(args.out_dir, exist_ok=True)
    mpath = os.path.join(args.out_dir, "metrics.jsonl")
    stop = {"now": False}
    signal.signal(signal.SIGTERM,
                  lambda *_: (print("[rl-v2] SIGTERM -> save+exit"),
                              stop.__setitem__("now", True)))

    def save(tag: str) -> None:
        d = os.path.join(args.out_dir, tag)
        policy.save_pretrained(d)
        tok.save_pretrained(d)
        print(f"[rl-v2] saved {d}")

    for step in range(args.steps):
        picks = [pool[(step * args.prompts_per_step + j) % len(pool)]
                 for j in range(args.prompts_per_step)]
        t0 = time.time()
        m = rl_step(policy, reference, tok, optimizer, scaler, catalog, picks,
                    idx, cfg, SYSTEM_PROMPT_V2)
        m["step"] = step
        m["step_seconds"] = round(time.time() - t0, 1)
        with open(mpath, "a") as f:
            f.write(json.dumps(m) + "\n")
        print(f"step {step:>3} | R {m['reward_mean']:+.3f}±{m['reward_std']:.3f} "
              f"| adv|.| {m['adv_abs_mean']:.3f} | KL {m['kl_mean']:.5f} "
              f"| value {m['value_mean']:.2f} | viol {m['violation_rate']:.2f} "
              f"| ask {m['ask_rate']:.2f} | gnorm {m['grad_norm']:.2f} "
              f"| {m['step_seconds']}s")
        if m["violation_rate"] > 0:
            print("[rl-v2] WARNING: permission violation in rollouts — "
                  "watch this metric; RL must never trade it away")
        if (step + 1) % args.save_every == 0 and (step + 1) < args.steps:
            save(f"step-{step+1}")
        if stop["now"]:
            save(f"step-{step+1}-sigterm")
            raise SystemExit(0)
    save("policy")
    print(f"[rl-v2] done -> {args.out_dir}/policy")


if __name__ == "__main__":
    main()
