"""SS10 — DATA-PATH GAP: is the dataloader ever the bottleneck? (GPU)

    python benchmarks/ss10_prefetch.py --predicted "..." --out <dir> \
        2>&1 | tee <dir>/run_$SLURM_JOB_ID.log

Our SFT pre-tokenizes everything up front, so the expected result is a
NO-GAP artifact ("a profile showing no hotspot is itself reportable").
Measures per-step: data time (collate + H2D) vs compute time (fwd/bwd/step).
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predicted", default="")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", default="benchmarks/artifacts/ss10")
    args = ap.parse_args()

    from shoprl.profiling import require_prediction, write_manifest
    predicted = require_prediction(args.predicted)
    os.makedirs(args.out, exist_ok=True)

    import torch

    from shoprl.data.catalog import generate_catalog
    from shoprl.data.prompts_v2 import SYSTEM_PROMPT_V2
    from shoprl.data.sft_v2 import generate_sft_v2_dialogues
    from shoprl.train.build import build_model
    from shoprl.train.sft import build_example, collate

    built = build_model("Qwen/Qwen2.5-1.5B-Instruct", method="lora",
                        dtype=torch.float16, device="cuda",
                        grad_checkpointing=False)
    model, tok = built.policy, built.tokenizer
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    scaler = torch.amp.GradScaler("cuda")
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-5)

    catalog = generate_catalog(n=300, seed=0)
    t_tok0 = time.time()
    demos = generate_sft_v2_dialogues(
        catalog, n=args.steps * args.batch_size, seed=0)
    examples = [build_example(tok, d, args.max_len, system=SYSTEM_PROMPT_V2)
                for d in demos]
    pretokenize_s = round(time.time() - t_tok0, 1)

    model.train()
    data_s = compute_s = 0.0
    for step in range(args.steps):
        t0 = time.time()
        batch = collate(examples[step * args.batch_size:(step + 1) * args.batch_size],
                        tok.pad_token_id, "cuda")
        torch.cuda.synchronize()
        data_s += time.time() - t0
        t1 = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(**batch)
        scaler.scale(out.loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        compute_s += time.time() - t1

    frac = data_s / (data_s + compute_s)
    measured = (f"data path {data_s:.2f}s vs compute {compute_s:.2f}s over "
                f"{args.steps} steps = {frac*100:.2f}% of step time "
                f"(one-time pre-tokenization {pretokenize_s}s)")
    verdict = ("NO-GAP artifact: the dataloader is not a hotspot — "
               "pre-tokenization already eliminated it; prefetch machinery "
               "would optimize " + f"{frac*100:.2f}% of the loop"
               if frac < 0.03 else
               "data path IS visible — prefetch/pipelining justified")
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"data_seconds": round(data_s, 2),
                   "compute_seconds": round(compute_s, 2),
                   "data_fraction": round(frac, 4),
                   "pretokenize_seconds": pretokenize_s,
                   "verdict": verdict}, f, indent=2)
    write_manifest(args.out, "SS10", "dataloader-gap measurement",
                   predicted=predicted, measured=f"{measured} → {verdict}",
                   mechanism=("all tokenization happens once up front; per-step "
                              "data work is a python list slice + pad + one H2D "
                              "copy of ~4k ints"),
                   config=vars(args),
                   rerun_cmd="python benchmarks/ss10_prefetch.py --predicted '...'")
    print(f"[SS10] {measured}\n[SS10] {verdict}")


if __name__ == "__main__":
    main()
