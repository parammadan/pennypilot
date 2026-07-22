"""SS9 — SEQUENCE PACKING (length-bucketing): real-vs-pad tokens + tokens/sec.

    python benchmarks/ss09_packing.py --predicted "..." --out <dir> \
        2>&1 | tee <dir>/run_$SLURM_JOB_ID.log

Two arms over identical demos and step count:
  shuffled  — batches in dataset order (current SFT behaviour)
  bucketed  — batches of similar length (sort by length, then batch)
Honest labeling: this is length-BUCKETING, not block-diagonal packing — same
goal (kill pad waste), none of the attention-mask surgery.
"""
from __future__ import annotations

import argparse
import json
import os
import time


def run_arm(bucketed: bool, examples, tok, args) -> dict:
    import torch

    from shoprl.train.build import build_model
    from shoprl.train.sft import collate

    torch.manual_seed(0)
    torch.cuda.empty_cache()
    built = build_model("Qwen/Qwen2.5-1.5B-Instruct", method="lora",
                        dtype=torch.float16, device="cuda",
                        grad_checkpointing=False)   # SS7: ckpt-off adopted
    model = built.policy
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    scaler = torch.amp.GradScaler("cuda")
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-5)

    order = (sorted(range(len(examples)), key=lambda i: len(examples[i]["input_ids"]))
             if bucketed else list(range(len(examples))))
    model.train()
    real = pad = 0
    torch.cuda.synchronize()
    t0 = time.time()
    steps = 0
    for i in range(0, len(order), args.batch_size):
        batch_ex = [examples[j] for j in order[i:i + args.batch_size]]
        batch = collate(batch_ex, tok.pad_token_id, "cuda")
        n_real = int(batch["attention_mask"].sum().item())
        real += n_real
        pad += batch["input_ids"].numel() - n_real
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(**batch)
        scaler.scale(out.loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
        if steps >= args.steps:
            break
    torch.cuda.synchronize()
    wall = time.time() - t0
    result = {"arm": "bucketed" if bucketed else "shuffled", "steps": steps,
              "real_tokens": real, "pad_tokens": pad,
              "pad_fraction": round(pad / (real + pad), 4),
              "real_tokens_per_sec": round(real / wall, 1),
              "wall_seconds": round(wall, 1)}
    del model, built, optimizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predicted", default="")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", default="benchmarks/artifacts/ss09")
    args = ap.parse_args()

    from shoprl.profiling import require_prediction, write_manifest
    predicted = require_prediction(args.predicted)
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer

    from shoprl.data.catalog import generate_catalog
    from shoprl.data.prompts_v2 import SYSTEM_PROMPT_V2
    from shoprl.data.sft_v2 import generate_sft_v2_dialogues
    from shoprl.train.sft import build_example

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    catalog = generate_catalog(n=300, seed=0)
    demos = generate_sft_v2_dialogues(
        catalog, n=args.steps * args.batch_size + args.batch_size, seed=0)
    examples = [build_example(tok, d, args.max_len, system=SYSTEM_PROMPT_V2)
                for d in demos]

    shuffled = run_arm(False, examples, tok, args)
    bucketed = run_arm(True, examples, tok, args)
    measured = (f"pad waste {shuffled['pad_fraction']*100:.1f}% → "
                f"{bucketed['pad_fraction']*100:.1f}%; real-token throughput "
                f"{shuffled['real_tokens_per_sec']} → "
                f"{bucketed['real_tokens_per_sec']} tok/s "
                f"({bucketed['real_tokens_per_sec']/shuffled['real_tokens_per_sec']:.2f}×)")
    _bars(shuffled, bucketed, args.out, predicted, measured)
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"shuffled": shuffled, "bucketed": bucketed}, f, indent=2)
    write_manifest(args.out, "SS9", "length-bucketing vs shuffled padding",
                   predicted=predicted, measured=measured,
                   mechanism=("batch width = longest member; mixing lengths "
                              "pads everyone to the outlier — sorting by "
                              "length makes width ≈ members' own length"),
                   config=vars(args),
                   rerun_cmd="python benchmarks/ss09_packing.py --predicted '...'")
    print(f"[SS9] {measured}")


def _bars(sh, bu, out, predicted, measured) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.4), facecolor="#fcfcfb")
    arms = [sh, bu]
    x = range(2)
    real = [a["real_tokens"] / 1e3 for a in arms]
    pad = [a["pad_tokens"] / 1e3 for a in arms]
    ax.bar(x, real, 0.5, label="real tokens (k)", color="#2a78d6")
    ax.bar(x, pad, 0.5, bottom=real, label="pad tokens (k)", color="#eda100")
    for i, a in enumerate(arms):
        ax.annotate(f"{a['pad_fraction']*100:.1f}% pad\n"
                    f"{a['real_tokens_per_sec']} real tok/s",
                    (i, (real[i] + pad[i])), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=9,
                    color="#0b0b0b")
    ax.set_xticks(list(x), ["shuffled", "length-bucketed"])
    ax.set_ylabel("tokens per 50 steps (thousands)", color="#52514e")
    ax.legend(frameon=False, fontsize=9)
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color="#e7e7e4", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.suptitle(f"SS9 — pad waste | predicted: {predicted}", fontsize=9.5,
                 x=0.01, ha="left")
    ax.set_title(f"measured: {measured}", fontsize=8, loc="left",
                 color="#52514e", pad=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(f"{out}/ss09_padding.png", dpi=300, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
