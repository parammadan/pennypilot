"""PennyPilot SFT (GPU) — LoRA + fp16/GradScaler on env-recorded demos.

Recipe pinned by measurement (profiling/gate/precision.json): fp16 autocast +
GradScaler with fp32 trainable params — 4.27x faster than bf16-emulated on the
V100, equally stable. The loss mask is verified LOUDLY before step 1.

Tiny-config gate first (Part-B rule: no full run before a <=1h tiny run):
    python scripts/run_sft_v2.py --demos 64 --max-steps 20 \
        --out-dir /scratch/madan.pa/pennypilot/sft_tiny

Full run (fits a 3h request; sbatch template in docs repo RUN_ON_SLURM.md):
    python scripts/run_sft_v2.py --demos 1000 --epochs 1 \
        --out-dir /scratch/madan.pa/pennypilot/sft_v2

Lifecycle: checkpoints (adapter + optimizer + scaler + step + RNG) every
--save-every steps; SIGTERM -> finish step, save, exit 0 (sbatch --requeue
resumes); --resume latest picks up the newest checkpoint in --out-dir.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time


def latest_checkpoint(out_dir: str) -> str | None:
    if not os.path.isdir(out_dir):
        return None
    steps = [(int(d.split("-")[1]), d) for d in os.listdir(out_dir)
             if d.startswith("step-") and d.split("-")[1].isdigit()]
    return os.path.join(out_dir, max(steps)[1]) if steps else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--method", default="lora", choices=["lora", "full"])
    ap.add_argument("--system", default="v2", choices=["v2", "chat"],
                    help="v2 = actions-only SYSTEM_PROMPT_V2 (the 1.5B recipe); "
                         "chat = SYSTEM_PROMPT_CHAT_MIN (H3 chat-capable arms)")
    ap.add_argument("--rehearsal", default=None,
                    help="JSONL of {q,a} general-chat exemplars to MIX IN "
                         "(H3 Arm B rehearsal); omit for the specialist arm")
    ap.add_argument("--demos", type=int, default=1000)
    ap.add_argument("--cannot-frac", type=float, default=None,
                    help="override cannot_fulfill share in --mix-generated")
    ap.add_argument("--mix-generated", type=int, default=0,
                    help="with --dataset-file: also mix N generated demos "
                         "(family/kind diversity guards the action grammar)")
    ap.add_argument("--dataset-file", default=None,
                    help="train on a recipe-exported dataset (messages JSONL)"
                         " INSTEAD of generated demos; --rehearsal still mixes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--catalog-size", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1024,
                    help="v2 demos carry search observations; re-check the "
                         "token-length histogram this script prints")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--resume", default=None, help='"latest" or a step-N dir')
    ap.add_argument("--out-dir", default="runs/sft_v2")
    args = ap.parse_args()

    import torch

    from shoprl.data.catalog import generate_catalog
    from shoprl.data.prompts_v2 import SYSTEM_PROMPT_CHAT_MIN, SYSTEM_PROMPT_V2
    from shoprl.data.sft_v2 import (DemoTurnV2, DemoV2, demo_v2_stats,
                                    generate_sft_v2_dialogues)
    from shoprl.train.build import build_model
    from shoprl.train.sft import build_example, collate, verify_mask

    SYS = SYSTEM_PROMPT_CHAT_MIN if args.system == "chat" else SYSTEM_PROMPT_V2

    os.makedirs(args.out_dir, exist_ok=True)
    catalog = generate_catalog(n=args.catalog_size, seed=0)
    if args.dataset_file:
        from shoprl.data.sft_v2 import demos_from_messages_jsonl
        demos = demos_from_messages_jsonl(args.dataset_file)
        print(f"[sft-v2] recipe dataset: {len(demos)} sequences "
              f"from {args.dataset_file}")
        if args.mix_generated:
            kw = ({"cannot_frac": args.cannot_frac}
                  if args.cannot_frac is not None else {})
            gen = generate_sft_v2_dialogues(catalog, n=args.mix_generated,
                                            seed=args.seed, **kw)
            demos = demos + gen
            print(f"[sft-v2] +{len(gen)} generated demos mixed "
                  f"(action-grammar diversity)")
    else:
        demos = generate_sft_v2_dialogues(catalog, n=args.demos, seed=args.seed)
        print(f"[sft-v2] shopping demos: {json.dumps(demo_v2_stats(demos))}")
    if args.rehearsal:                        # H3 Arm B: mix in general chat
        reh = [json.loads(l) for l in open(args.rehearsal) if l.strip()]
        reh_demos = [DemoV2(scenario_id=f"chat-{i}", kind="rehearsal",
                            language=r.get("lang", "en"), target_sku="", n_asks=0,
                            turns=[DemoTurnV2("user", r["q"]),
                                   DemoTurnV2("agent", r["a"])])
                     for i, r in enumerate(reh)]
        demos = demos + reh_demos
        import random as _rnd
        _rnd.Random(args.seed).shuffle(demos)
        print(f"[sft-v2] +{len(reh_demos)} rehearsal exemplars "
              f"({len(reh_demos)/len(demos)*100:.0f}% of mix), shuffled")

    built = build_model(args.model, method=args.method,
                        dtype=torch.float16, device="cuda",
                        grad_checkpointing=True)
    model, tok = built.policy, built.tokenizer
    # fp16 recipe: fp32 master for everything that gets a gradient.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    examples = [build_example(tok, d, args.max_len, system=SYS)
                for d in demos]
    lens = sorted(len(e["input_ids"]) for e in examples)
    print(f"[sft-v2] token lens: max {lens[-1]} p99 {lens[int(len(lens)*0.99)-1]} "
          f"mean {sum(lens)//len(lens)} (max_len {args.max_len})")
    verify_mask(tok, demos[0], args.max_len, example=examples[0], system=SYS)
    print(f"[sft-v2] loss mask VERIFIED (system={args.system}, "
          "reference-exact + leak checks)")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    start_step = 0
    resume_dir = (latest_checkpoint(args.out_dir) if args.resume == "latest"
                  else args.resume)
    if resume_dir:
        state = torch.load(os.path.join(resume_dir, "train_state.pt"),
                           weights_only=False)
        model.load_state_dict(state["model"], strict=False)   # adapter weights
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["rng_cpu"])
        torch.cuda.set_rng_state_all(state["rng_cuda"])
        start_step = state["step"]
        print(f"[sft-v2] resumed from {resume_dir} at step {start_step}")

    stop = {"now": False}
    signal.signal(signal.SIGTERM,
                  lambda *_: (print("[sft-v2] SIGTERM -> save+exit"),
                              stop.__setitem__("now", True)))

    def save(step: int, final: bool = False) -> None:
        d = os.path.join(args.out_dir, "policy" if final else f"step-{step}")
        os.makedirs(d, exist_ok=True)
        model.save_pretrained(d)
        tok.save_pretrained(d)
        torch.save({
            "step": step,
            "model": {k: v for k, v in model.state_dict().items()
                      if "lora" in k.lower()} if args.method == "lora"
                     else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all(),
        }, os.path.join(d, "train_state.pt"))
        print(f"[sft-v2] saved {d}")

    perf = {"tok": 0, "steps": 0}

    def perf_summary() -> None:
        wall = time.time() - t0
        gb = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        rec = {"model": args.model, "method": args.method, "system": args.system,
               "batch_size": args.batch_size, "rehearsal": bool(args.rehearsal),
               "steps": perf["steps"], "wall_s": round(wall, 1),
               "ms_per_step": round(wall / max(perf["steps"], 1) * 1000, 1),
               "tokens_per_s": round(perf["tok"] / wall, 1) if wall else 0,
               "peak_mem_gb": gb}
        with open(os.path.join(args.out_dir, "perf.json"), "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[sft-v2] PERF baseline: {json.dumps(rec)}")

    mpath = os.path.join(args.out_dir, "metrics.jsonl")
    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for i in range(0, len(examples), args.batch_size):
            step += 1
            if step <= start_step:
                continue                       # fast-forward on resume
            batch = collate(examples[i:i + args.batch_size],
                            tok.pad_token_id, "cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(**batch)
            loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            m = {"step": step, "epoch": epoch, "loss": round(float(loss.item()), 5),
                 "grad_norm": round(float(gnorm), 4),
                 "scaler_scale": float(scaler.get_scale()),
                 "supervised_tokens": int((batch["labels"] != -100).sum().item()),
                 "batch_tokens": int(batch["attention_mask"].sum().item()),
                 "wall_s": round(time.time() - t0, 1)}
            perf["tok"] += m["batch_tokens"]; perf["steps"] += 1
            with open(mpath, "a") as f:
                f.write(json.dumps(m) + "\n")
            if step % 10 == 0 or step == 1:
                print(f"step {step:>4} | loss {m['loss']:.4f} | "
                      f"gnorm {m['grad_norm']:.2f} | scale {m['scaler_scale']:.0f}")
            if not torch.isfinite(loss):
                save(step); raise SystemExit("[sft-v2] NON-FINITE LOSS — stopped")
            if step % args.save_every == 0 or stop["now"]:
                save(step)
            if stop["now"]:
                raise SystemExit(0)
            if args.max_steps and step - start_step >= args.max_steps:
                save(step, final=True)
                perf_summary()
                print(f"[sft-v2] max-steps reached -> {args.out_dir}/policy")
                return
    save(step, final=True)
    perf_summary()
    print(f"[sft-v2] done ({step} steps, {time.time()-t0:.0f}s) -> {args.out_dir}/policy")


if __name__ == "__main__":
    main()
