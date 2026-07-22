"""SS8 — LoRA vs FULL-FT memory anatomy: fixed footprint + optimizer state.

    python benchmarks/ss08_lora_vs_full.py --predicted "..." \
        2>&1 | tee benchmarks/artifacts/ss08/run_$SLURM_JOB_ID.log

Measures per method: weights footprint, trainable params, optimizer-state
bytes after one step (AdamW moments materialize lazily), backward peak.
"""
from __future__ import annotations

import argparse
import json
import os


def run_arm(method: str, args) -> dict:
    import torch

    from shoprl.data.catalog import generate_catalog
    from shoprl.data.prompts_v2 import SYSTEM_PROMPT_V2
    from shoprl.data.sft_v2 import generate_sft_v2_dialogues
    from shoprl.train.build import build_model
    from shoprl.train.sft import build_example, collate

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    built = build_model("Qwen/Qwen2.5-1.5B-Instruct", method=method,
                        dtype=torch.float16, device="cuda",
                        grad_checkpointing=True)
    model, tok = built.policy, built.tokenizer
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    weights_gb = round((torch.cuda.memory_allocated() - base_mem) / 2**30, 2)

    if method == "lora":                    # pinned fp16 recipe
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=method == "lora")

    catalog = generate_catalog(n=300, seed=0)
    demos = generate_sft_v2_dialogues(catalog, n=2, seed=0)
    examples = [build_example(tok, d, 1024, system=SYSTEM_PROMPT_V2)
                for d in demos]
    model.train()
    batch = collate(examples, tok.pad_token_id, "cuda")
    pre_step = torch.cuda.memory_allocated()
    with torch.autocast("cuda", dtype=torch.float16,
                        enabled=method == "lora"):
        out = model(**batch)
    scaler.scale(out.loss).backward()
    scaler.step(optimizer)
    scaler.update()
    opt_bytes = sum(t.numel() * t.element_size()
                    for st in optimizer.state.values()
                    for t in st.values() if torch.is_tensor(t))
    result = {
        "method": method,
        "trainable_params_m": round(trainable / 1e6, 1),
        "weights_gb": weights_gb,
        "optimizer_state_gb": round(opt_bytes / 2**30, 3),
        "post_step_allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
        "backward_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    del model, built, optimizer, out, batch
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predicted", default="")
    ap.add_argument("--out", default="benchmarks/artifacts/ss08")
    args = ap.parse_args()

    from shoprl.profiling import require_prediction, write_manifest
    predicted = require_prediction(args.predicted)
    os.makedirs(args.out, exist_ok=True)

    lora = run_arm("lora", args)
    full = run_arm("full", args)
    ratio = round(full["optimizer_state_gb"] / max(lora["optimizer_state_gb"], 1e-6))
    measured = (f"optimizer state: LoRA {lora['optimizer_state_gb']} GB vs "
                f"full-FT {full['optimizer_state_gb']} GB ({ratio}×); "
                f"trainable {lora['trainable_params_m']}M vs "
                f"{full['trainable_params_m']}M; backward peak "
                f"{lora['backward_peak_gb']} vs {full['backward_peak_gb']} GB")
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"lora": lora, "full": full,
                   "optimizer_ratio": ratio}, f, indent=2)
    write_manifest(args.out, "SS8", "LoRA vs full-FT memory anatomy",
                   predicted=predicted, measured=measured,
                   mechanism=("AdamW keeps 2 fp32 moments per trainable param "
                              "(+fp32 master under mixed precision): full-FT "
                              "pays it for 1.54B params, LoRA for ~18M — the "
                              "note that full-FT also needs a SEPARATE frozen "
                              "reference for KL while LoRA-from-base gets it "
                              "free stays in ARCHITECTURE.md"),
                   config=vars(args),
                   rerun_cmd="python benchmarks/ss08_lora_vs_full.py --predicted '...'")
    print(f"[SS8] {measured}")


if __name__ == "__main__":
    main()
