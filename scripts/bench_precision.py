"""PRECISION DECISION (GPU) — fp16+GradScaler vs bf16-emulated on the V100.

Decision criterion (approved 2026-07-21): stability AND tokens/sec — bf16 on
Volta is emulated (no tensor cores), so its throughput cost must be
quantified, not assumed. Context: Phase 2 measured PURE fp16 (no loss scaling)
NaN at step 1; proper fp16 mixed precision (fp32-master via autocast +
GradScaler) was never tried. Whichever wins here becomes v2's pinned training
precision, and policy+reference logprob paths must share it (no phantom KL).

    python scripts/bench_precision.py --steps 30 --method lora \
        --out /scratch/madan.pa/pennypilot/gate/precision.json

Runs the same N SFT steps (identical demos, identical seed) under each arm:
  bf16   — model+adapters in bf16, plain steps (the v1 recipe)
  fp16   — base fp16, trainable params fp32, autocast(fp16) + GradScaler
Reports: non-finite count, final loss, tokens/sec, peak memory.
"""
from __future__ import annotations

import argparse
import json
import time


def run_arm(arm: str, args) -> dict:
    import torch

    from shoprl.data.catalog import generate_catalog
    from shoprl.data.sft import generate_sft_dialogues
    from shoprl.train.build import build_model
    from shoprl.train.sft import build_example, collate

    torch.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    dtype = torch.bfloat16 if arm == "bf16" else torch.float16
    built = build_model(args.model, method=args.method, dtype=dtype,
                        device="cuda", grad_checkpointing=True)
    model, tok = built.policy, built.tokenizer
    if arm == "fp16":
        # fp32 master for everything that gets a gradient; base stays fp16.
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    scaler = torch.amp.GradScaler("cuda", enabled=(arm == "fp16"))

    catalog = generate_catalog(n=args.catalog_size, seed=0)
    demos = generate_sft_dialogues(catalog, n=args.steps * args.batch_size, seed=0)
    examples = [build_example(tok, d, args.max_len) for d in demos]
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-5)
    model.train()

    nonfinite = 0
    losses: list[float] = []
    tokens = 0
    torch.cuda.synchronize()
    t0 = time.time()
    for step in range(args.steps):
        batch = collate(examples[step * args.batch_size:(step + 1) * args.batch_size],
                        tok.pad_token_id, "cuda")
        with torch.autocast("cuda", dtype=torch.float16, enabled=(arm == "fp16")):
            out = model(**batch)
        loss = out.loss
        if not torch.isfinite(loss):
            nonfinite += 1
        losses.append(float(loss.item()))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        tokens += int(batch["attention_mask"].sum().item())
    torch.cuda.synchronize()
    wall = time.time() - t0

    result = {
        "arm": arm, "steps": args.steps, "nonfinite_losses": nonfinite,
        "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
        "tokens_per_sec": round(tokens / wall, 1),
        "wall_seconds": round(wall, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "scaler_scale_final": float(scaler.get_scale()) if arm == "fp16" else None,
    }
    del model, built, optimizer
    torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--method", default="lora", choices=["lora", "full"])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--catalog-size", type=int, default=300)
    ap.add_argument("--out", default="runs/gate/precision.json")
    args = ap.parse_args()

    arms = [run_arm("bf16", args), run_arm("fp16", args)]
    stable = [a for a in arms if a["nonfinite_losses"] == 0]
    if not stable:
        verdict = "NEITHER STABLE — investigate before any training"
    elif len(stable) == 1:
        verdict = f"{stable[0]['arm']} (only stable arm)"
    else:
        fastest = max(stable, key=lambda a: a["tokens_per_sec"])
        other = min(stable, key=lambda a: a["tokens_per_sec"])
        speedup = fastest["tokens_per_sec"] / max(other["tokens_per_sec"], 1e-9)
        verdict = (f"{fastest['arm']} (both stable; {speedup:.2f}x faster: "
                   f"{fastest['tokens_per_sec']} vs {other['tokens_per_sec']} tok/s)")

    summary = {"model": args.model, "method": args.method, "arms": arms,
               "verdict": verdict,
               "note": ("pin this precision for policy AND reference logprob "
                        "paths (no phantom KL); record in docs repo")}
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\n=== PRECISION VERDICT: {verdict}")


if __name__ == "__main__":
    main()
