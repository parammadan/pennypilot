"""SS7 — GRADIENT-CHECKPOINTING TOGGLE: peak memory + step time (GPU).

Both effects predicted BEFORE measuring (campaign rule): checkpointing trades
an extra forward pass during backward for not storing activations.

    python benchmarks/ss07_grad_ckpt.py --predicted "..." \
        2>&1 | tee benchmarks/artifacts/ss07/run_$SLURM_JOB_ID.log
"""
from __future__ import annotations

import argparse
import json
import os
import time


def run_arm(ckpt: bool, args) -> dict:
    import torch

    from shoprl.data.catalog import generate_catalog
    from shoprl.data.prompts_v2 import SYSTEM_PROMPT_V2
    from shoprl.data.sft_v2 import generate_sft_v2_dialogues
    from shoprl.train.build import build_model
    from shoprl.train.sft import build_example, collate

    torch.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    built = build_model("Qwen/Qwen2.5-1.5B-Instruct", method="lora",
                        dtype=torch.float16, device="cuda",
                        grad_checkpointing=ckpt)
    model, tok = built.policy, built.tokenizer
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    scaler = torch.amp.GradScaler("cuda")
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-5)

    catalog = generate_catalog(n=300, seed=0)
    demos = generate_sft_v2_dialogues(catalog, n=args.steps * args.batch_size,
                                      seed=0)
    examples = [build_example(tok, d, args.max_len, system=SYSTEM_PROMPT_V2)
                for d in demos]
    model.train()
    torch.cuda.synchronize()
    t0 = time.time()
    tokens = 0
    for step in range(args.steps):
        batch = collate(examples[step * args.batch_size:(step + 1) * args.batch_size],
                        tok.pad_token_id, "cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(**batch)
        scaler.scale(out.loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        tokens += int(batch["attention_mask"].sum().item())
    torch.cuda.synchronize()
    wall = time.time() - t0
    result = {"grad_checkpointing": ckpt, "steps": args.steps,
              "step_seconds": round(wall / args.steps, 3),
              "tokens_per_sec": round(tokens / wall, 1),
              "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    del model, built, optimizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predicted", default="")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", default="benchmarks/artifacts/ss07")
    args = ap.parse_args()

    from shoprl.profiling import require_prediction, write_manifest
    predicted = require_prediction(args.predicted)
    os.makedirs(args.out, exist_ok=True)

    on = run_arm(True, args)
    off = run_arm(False, args)
    ratio_mem = round(off["peak_mem_gb"] / on["peak_mem_gb"], 2)
    ratio_speed = round(off["tokens_per_sec"] / on["tokens_per_sec"], 2)
    measured = (f"ckpt-ON {on['peak_mem_gb']} GB @ {on['tokens_per_sec']} tok/s; "
                f"ckpt-OFF {off['peak_mem_gb']} GB @ {off['tokens_per_sec']} tok/s "
                f"(off = {ratio_mem}× memory, {ratio_speed}× speed)")
    _bars(on, off, args.out, predicted, measured)
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"on": on, "off": off, "mem_ratio_off_over_on": ratio_mem,
                   "speed_ratio_off_over_on": ratio_speed}, f, indent=2)
    write_manifest(args.out, "SS7", "gradient-checkpointing toggle",
                   predicted=predicted, measured=measured,
                   mechanism=("checkpointing recomputes activations in backward "
                              "(extra forward ≈ +30-40% step time) instead of "
                              "storing them (activation memory ∝ layers)"),
                   config=vars(args),
                   rerun_cmd="python benchmarks/ss07_grad_ckpt.py --predicted '...'")
    print(f"[SS7] {measured}")


def _bars(on, off, out, predicted, measured) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2), facecolor="#fcfcfb")
    for ax, key, label in ((ax1, "peak_mem_gb", "peak memory (GB)"),
                           (ax2, "tokens_per_sec", "tokens/sec")):
        vals = [on[key], off[key]]
        bars = ax.bar(["ckpt ON", "ckpt OFF"], vals, color="#2a78d6", width=0.5)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=10, color="#0b0b0b")
        ax.set_ylabel(label, color="#52514e")
        ax.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, axis="y", color="#e7e7e4", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(f"SS7 — grad checkpointing | predicted: {predicted}",
                 fontsize=9.5, x=0.01, ha="left")
    ax1.set_title(f"measured: {measured}", fontsize=8, loc="left",
                  color="#52514e", pad=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(f"{out}/ss07_toggle.png", dpi=300, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
