"""SS0 — BASELINE: annotated profile of ONE full RL iteration (GPU).

    python benchmarks/ss00_baseline.py --sft-adapter <adapter> \
        --predicted "rollout >= 70% of iteration wall-clock" \
        2>&1 | tee benchmarks/artifacts/ss00/run_$(date +%Y%m%d_%H%M).log

Captures: NVML timeline (util/mem, 200 ms) with phase annotations, wall-clock
% per phase (the headline), torch.profiler chrome trace of the update phase,
MANIFEST.md. Refuses to run without --predicted (campaign rule).
"""
from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--predicted", required=False, default="")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--language", default="es-en")
    ap.add_argument("--n-must-haves", type=int, nargs=2, default=[3, 4])
    ap.add_argument("--valid-range", type=int, nargs=2, default=[3, 10])
    ap.add_argument("--out", default="benchmarks/artifacts/ss00")
    args = ap.parse_args()

    from shoprl.profiling import (EventLog, GPUSampler, phase_breakdown,
                                  render_phase_bar, render_timeline,
                                  require_prediction, write_manifest)
    predicted = require_prediction(args.predicted)
    os.makedirs(args.out, exist_ok=True)

    import torch

    from shoprl.data.catalog import catalog_index, generate_catalog
    from shoprl.data.prompts_v2 import SYSTEM_PROMPT_V2
    from shoprl.env.scenario import generate_hard_scenarios
    from shoprl.profiling.bench_common import load_hf_policy
    from shoprl.train.algo import RLConfigV2, rl_step

    policy, tok = load_hf_policy("Qwen/Qwen2.5-1.5B-Instruct", args.sft_adapter)
    reference, _ = load_hf_policy("Qwen/Qwen2.5-1.5B-Instruct", args.sft_adapter)
    for p in reference.parameters():
        p.requires_grad_(False)
    # rl_step trains the policy adapter: re-enable + fp32 masters.
    for n, p in policy.named_parameters():
        if "lora" in n.lower():
            p.requires_grad_(True)
            p.data = p.data.float()
    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad), lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    catalog = generate_catalog(n=300, seed=0)
    idx = catalog_index(catalog)
    scen = generate_hard_scenarios(catalog, n=1, seed=3000,
                                   n_must_haves=tuple(args.n_must_haves),
                                   valid_target_range=tuple(args.valid_range))
    cfg = RLConfigV2(algo="rloo", k=args.k, language=args.language)

    sampler = GPUSampler().start()
    events = EventLog(t0=sampler.t0)
    m = rl_step(policy, reference, tok, optimizer, scaler, catalog, scen, idx,
                cfg, SYSTEM_PROMPT_V2, annotate=events.mark)
    samples = sampler.stop()
    events.to_csv(os.path.join(args.out, "events.csv"))

    b = phase_breakdown(events.events)
    measured = " / ".join(f"{k} {v*100:.1f}%" for k, v in b["fractions"].items())
    render_timeline(samples, events.events,
                    os.path.join(args.out, "ss00_timeline.png"),
                    f"SS0 — one RL iteration | predicted: {predicted} | measured: {measured}",
                    subtitle="NVML 200ms; phases annotated; V100, HF generate rollouts")
    render_phase_bar(b, os.path.join(args.out, "ss00_phases.png"),
                     f"SS0 — wall-clock per phase | predicted: {predicted} | measured: {measured}")
    with open(os.path.join(args.out, "step_metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    write_manifest(args.out, "SS0", "baseline RL-iteration phase split",
                   predicted=predicted, measured=measured,
                   mechanism=("autoregressive rollout decode is memory-"
                              "bandwidth-bound and sequential per turn; the "
                              "update is a handful of dense fwd/bwd passes"),
                   config=vars(args), rerun_cmd="python benchmarks/ss00_baseline.py "
                   f"--sft-adapter {args.sft_adapter} --predicted '...'")
    print(f"[SS0] measured: {measured}")
    print(f"[SS0] artifacts -> {args.out}")


if __name__ == "__main__":
    main()
