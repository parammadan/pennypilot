"""SS1 — ROLLOUT ENGINE BASELINE: HF `.generate()` tokens/sec (GPU).

    python benchmarks/ss01_rollout_hf.py --sft-adapter <adapter> \
        --predicted "<engine tok/s guess>" \
        2>&1 | tee benchmarks/artifacts/ss01/run_$(date +%Y%m%d_%H%M).log

N full conversations through the SAME env loop SS2 uses (rollout_v2); reports
engine tokens/sec (generation time only) AND loop-effective tokens/sec (wall
incl. env steps) — the gap between them is SS4b's subject.
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--predicted", default="")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--language", default="es-en")
    ap.add_argument("--out", default="benchmarks/artifacts/ss01")
    args = ap.parse_args()

    from shoprl.profiling import (EventLog, GPUSampler, render_timeline,
                                  require_prediction, write_manifest)
    predicted = require_prediction(args.predicted)
    os.makedirs(args.out, exist_ok=True)

    from shoprl.data.catalog import catalog_index, generate_catalog
    from shoprl.data.prompts_v2 import SYSTEM_PROMPT_V2
    from shoprl.env.catalog_env import SyntheticCatalogEnvironment
    from shoprl.env.scenario import generate_hard_scenarios
    from shoprl.profiling.bench_common import (GenMeter, hf_agent_fn,
                                               load_hf_policy)
    from shoprl.train.algo import rollout_v2

    model, tok = load_hf_policy("Qwen/Qwen2.5-1.5B-Instruct", args.sft_adapter)
    meter = GenMeter()
    agent_fn = hf_agent_fn(model, tok, meter)

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
        "gen_tokens": meter.gen_tokens,
        "engine_tokens_per_sec": meter.engine_tps(),
        "loop_tokens_per_sec": round(meter.gen_tokens / wall, 1),
        "gen_seconds": round(meter.gen_seconds, 1),
        "wall_seconds": round(wall, 1),
        "gen_fraction_of_wall": round(meter.gen_seconds / wall, 3),
        "reward_mean": round(sum(rewards) / len(rewards), 3),
    }
    measured = (f"engine {result['engine_tokens_per_sec']} tok/s, loop "
                f"{result['loop_tokens_per_sec']} tok/s")
    render_timeline(samples, [], os.path.join(args.out, "ss01_timeline.png"),
                    f"SS1 — HF rollout engine | predicted: {predicted} | measured: {measured}",
                    subtitle=f"{args.episodes} sequential conversations, HF greedy-batch=1 generate")
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    write_manifest(args.out, "SS1", "HF .generate() rollout throughput",
                   predicted=predicted, measured=measured,
                   mechanism=("batch-1 sequential decode: every token streams "
                              "all weights; no batching to amortize reads"),
                   config=vars(args),
                   rerun_cmd=f"python benchmarks/ss01_rollout_hf.py "
                             f"--sft-adapter {args.sft_adapter} --predicted '...'")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
