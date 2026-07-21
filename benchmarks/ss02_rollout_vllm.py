"""SS2 — ROLLOUT ENGINE: vLLM tokens/sec on the SAME env loop as SS1 (GPU).

Run inside the pinned venv (vllm==0.7.3 + transformers==4.49.0), which also
needs the repo installed once:
    /scratch/madan.pa/venvs/vllm-073/bin/pip install -e ~/pennywise-v100-infra

    VLLM_USE_V1=0 /scratch/madan.pa/venvs/vllm-073/bin/python \
        benchmarks/ss02_rollout_vllm.py --sft-adapter <adapter> \
        --predicted "<engine tok/s guess>" \
        2>&1 | tee benchmarks/artifacts/ss02/run_$(date +%Y%m%d_%H%M).log

NO cuda module loaded (pip wheels carry their own CUDA libs — measured gotcha).
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--sft-adapter", default=None,
                    help="LoRA adapter dir (rides along as a LoRARequest)")
    ap.add_argument("--predicted", default="")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--language", default="es-en")
    ap.add_argument("--gpu-mem-util", type=float, default=0.45)
    ap.add_argument("--prefix-caching", action="store_true")
    ap.add_argument("--out", default="benchmarks/artifacts/ss02")
    args = ap.parse_args()

    from shoprl.profiling import (EventLog, GPUSampler, render_timeline,
                                  require_prediction, write_manifest)
    predicted = require_prediction(args.predicted)
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    from vllm import LLM

    from shoprl.data.catalog import catalog_index, generate_catalog
    from shoprl.data.prompts_v2 import SYSTEM_PROMPT_V2
    from shoprl.env.catalog_env import SyntheticCatalogEnvironment
    from shoprl.env.scenario import generate_hard_scenarios
    from shoprl.profiling.bench_common import GenMeter, vllm_agent_fn
    from shoprl.train.algo import rollout_v2

    tok = AutoTokenizer.from_pretrained(args.sft_adapter or args.model)
    llm = LLM(model=args.model, dtype="float16",
              gpu_memory_utilization=args.gpu_mem_util,
              enable_prefix_caching=args.prefix_caching,
              enable_lora=args.sft_adapter is not None, max_lora_rank=16,
              enforce_eager=True)
    meter = GenMeter()
    agent_fn = vllm_agent_fn(llm, tok, meter, adapter=args.sft_adapter)

    catalog = generate_catalog(n=300, seed=0)
    idx = catalog_index(catalog)
    scens = generate_hard_scenarios(catalog, n=args.episodes, seed=3000,
                                    n_must_haves=(3, 4),
                                    valid_target_range=(3, 10))
    sampler = GPUSampler().start()
    events = EventLog(t0=sampler.t0)
    t0 = time.time()
    rewards = []
    for i, scen in enumerate(scens):
        events.mark(f"ep{i}:start")
        env = SyntheticCatalogEnvironment(catalog, scen, idx=idx,
                                          language=args.language)
        rewards.append(rollout_v2(agent_fn, env, SYSTEM_PROMPT_V2).reward)
        events.mark(f"ep{i}:end")
    wall = time.time() - t0
    samples = sampler.stop()

    result = {
        "episodes": args.episodes,
        "prefix_caching": args.prefix_caching,
        "gen_tokens": meter.gen_tokens,
        "engine_tokens_per_sec": meter.engine_tps(),
        "loop_tokens_per_sec": round(meter.gen_tokens / wall, 1),
        "gen_seconds": round(meter.gen_seconds, 1),
        "wall_seconds": round(wall, 1),
        "reward_mean": round(sum(rewards) / len(rewards), 3),
    }
    measured = (f"engine {result['engine_tokens_per_sec']} tok/s, loop "
                f"{result['loop_tokens_per_sec']} tok/s"
                + (" (APC on)" if args.prefix_caching else ""))
    render_timeline(samples, [], os.path.join(args.out, "ss02_timeline.png"),
                    f"SS2 — vLLM rollout engine | predicted: {predicted} | measured: {measured}",
                    subtitle=f"vLLM 0.7.3 V0 engine, fp16, LoRA={'on' if args.sft_adapter else 'off'}, "
                             f"same {args.episodes}-episode loop as SS1")
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    write_manifest(args.out, "SS2", "vLLM rollout throughput (same loop as SS1)",
                   predicted=predicted, measured=measured,
                   mechanism=("PagedAttention KV management + persistent "
                              "engine remove per-call model setup; batch=1 "
                              "sequential turns still leave batching headroom "
                              "(SS4's subject)"),
                   config=vars(args),
                   rerun_cmd="VLLM_USE_V1=0 <venv>/bin/python "
                             "benchmarks/ss02_rollout_vllm.py --predicted '...'")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
