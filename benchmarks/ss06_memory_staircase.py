"""SS6 — MEMORY STAIRCASE: annotated NVML timeline of one RL iteration,
sampled from BEFORE model load (GPU, trainer env).

    python benchmarks/ss06_memory_staircase.py --sft-adapter <adapter> \
        --predicted "..." 2>&1 | tee benchmarks/artifacts/ss06/run_$SLURM_JOB_ID.log

Steps annotated: policy load → reference load → fp32 masters → rollout
(KV/cache growth) → update (backward peak) → optimizer step → post-step.
Reports the staircase figure + torch.cuda peak.
"""
from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--predicted", default="")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="benchmarks/artifacts/ss06")
    args = ap.parse_args()

    from shoprl.profiling import (EventLog, GPUSampler, render_timeline,
                                  require_prediction, write_manifest)
    predicted = require_prediction(args.predicted)
    os.makedirs(args.out, exist_ok=True)

    sampler = GPUSampler().start()          # BEFORE any CUDA allocation
    events = EventLog(t0=sampler.t0)

    import torch

    from shoprl.data.catalog import catalog_index, generate_catalog
    from shoprl.data.prompts_v2 import SYSTEM_PROMPT_V2
    from shoprl.env.scenario import generate_hard_scenarios
    from shoprl.profiling.bench_common import load_hf_policy
    from shoprl.train.algo import RLConfigV2, rl_step

    events.mark("policy_load:start")
    policy, tok = load_hf_policy("Qwen/Qwen2.5-1.5B-Instruct", args.sft_adapter)
    events.mark("policy_load:end")
    events.mark("reference_load:start")
    reference, _ = load_hf_policy("Qwen/Qwen2.5-1.5B-Instruct", args.sft_adapter)
    for p in reference.parameters():
        p.requires_grad_(False)
    events.mark("reference_load:end")
    events.mark("fp32_masters:start")
    for n, p in policy.named_parameters():
        if "lora" in n.lower():
            p.requires_grad_(True)
            p.data = p.data.float()
    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad), lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    events.mark("fp32_masters:end")

    catalog = generate_catalog(n=300, seed=0)
    idx = catalog_index(catalog)
    scen = generate_hard_scenarios(catalog, n=1, seed=3000,
                                   n_must_haves=(3, 4),
                                   valid_target_range=(3, 10))
    torch.cuda.reset_peak_memory_stats()
    cfg = RLConfigV2(algo="rloo", k=args.k, language="es-en")
    m = rl_step(policy, reference, tok, optimizer, scaler, catalog, scen, idx,
                cfg, SYSTEM_PROMPT_V2, annotate=events.mark)
    events.mark("post_step")
    samples = sampler.stop()

    peak_torch = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    peak_nvml = round(max(s[2] for s in samples), 2)
    measured = (f"NVML peak {peak_nvml} GB (torch alloc peak {peak_torch} GB); "
                f"staircase: 2 model loads then rollout/update plateaus")
    render_timeline(samples, events.events,
                    os.path.join(args.out, "ss06_staircase.png"),
                    f"SS6 — memory staircase | predicted: {predicted} | measured: {measured}",
                    subtitle=f"one RL iteration k={args.k}, LoRA fp16 recipe, "
                             "sampled from before model load")
    events.to_csv(os.path.join(args.out, "events.csv"))
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"peak_nvml_gb": peak_nvml, "peak_torch_gb": peak_torch,
                   "step_metrics": m}, f, indent=2)
    write_manifest(args.out, "SS6", "memory staircase of one RL iteration",
                   predicted=predicted, measured=measured,
                   mechanism=("two fp16 model copies dominate the floor; "
                              "rollout adds KV cache; backward adds activation "
                              "peak; optimizer state is tiny under LoRA"),
                   config=vars(args),
                   rerun_cmd=f"python benchmarks/ss06_memory_staircase.py "
                             f"--sft-adapter {args.sft_adapter} --predicted '...'")
    print(f"[SS6] {measured}")


if __name__ == "__main__":
    main()
