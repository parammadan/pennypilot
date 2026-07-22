"""SS4 + SS4b — CONCURRENCY & RL-LOOP UTILIZATION (vLLM venv).

Runs N ∈ {1, 4, 16} full episodes through the batched lockstep driver
(shoprl.train.batch_rollout) against one vLLM engine: one episode's env-step
gap is filled by other episodes' generation. Emits SS4 artifacts (aggregate
tokens/sec by N, N=1 vs N=16 GPU timelines — the SS5 interleave evidence) and
SS4b artifacts (utilization % + effective rollout throughput, sequential vs
batched — the single-GPU version of rollout/trainer overlap).

    VLLM_USE_V1=0 <venv>/bin/python benchmarks/ss04_concurrency.py \
        --sft-adapter <adapter> --predicted-ss4 "..." --predicted-ss4b "..." \
        2>&1 | tee benchmarks/artifacts/ss04/run_$SLURM_JOB_ID.log
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--sft-adapter", default=None)
    ap.add_argument("--predicted-ss4", default="")
    ap.add_argument("--predicted-ss4b", default="")
    ap.add_argument("--gpu-mem-util", type=float, default=0.45)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--prefix-caching", action="store_true", default=True)
    ap.add_argument("--out4", default="benchmarks/artifacts/ss04")
    ap.add_argument("--out4b", default="benchmarks/artifacts/ss04b")
    args = ap.parse_args()

    from shoprl.profiling import (EventLog, GPUSampler, render_timeline,
                                  require_prediction, write_manifest)
    p4 = require_prediction(args.predicted_ss4)
    p4b = require_prediction(args.predicted_ss4b)
    os.makedirs(args.out4, exist_ok=True)
    os.makedirs(args.out4b, exist_ok=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from shoprl.data.catalog import catalog_index, generate_catalog
    from shoprl.data.prompts_v2 import SYSTEM_PROMPT_V2
    from shoprl.env.catalog_env import SyntheticCatalogEnvironment
    from shoprl.env.scenario import generate_hard_scenarios
    from shoprl.train.batch_rollout import batched_rollouts

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="float16",
              gpu_memory_utilization=args.gpu_mem_util,
              enable_prefix_caching=args.prefix_caching,
              enable_lora=args.sft_adapter is not None, max_lora_rank=16,
              enforce_eager=True)
    lora_req = None
    if args.sft_adapter:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("policy", 1, args.sft_adapter)
    sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=64)

    meter = {"tokens": 0, "gen_s": 0.0}

    def generate_batch(message_lists):
        prompts = [tok.apply_chat_template(m, tokenize=False,
                                           add_generation_prompt=True)
                   for m in message_lists]
        t0 = time.time()
        outs = llm.generate(prompts, sp, lora_request=lora_req, use_tqdm=False)
        meter["gen_s"] += time.time() - t0
        meter["tokens"] += sum(len(o.outputs[0].token_ids) for o in outs)
        return [o.outputs[0].text.strip() for o in outs]

    catalog = generate_catalog(n=300, seed=0)
    idx = catalog_index(catalog)
    pool = generate_hard_scenarios(catalog, n=64, seed=3000,
                                   n_must_haves=(3, 4),
                                   valid_target_range=(3, 10))

    rows = []
    for n in args.levels:
        envs = [SyntheticCatalogEnvironment(catalog, pool[i], idx=idx,
                                            language="es-en")
                for i in range(n)]
        meter["tokens"] = 0
        meter["gen_s"] = 0.0
        sampler = GPUSampler().start()
        t0 = time.time()
        trajs = batched_rollouts(generate_batch, envs, SYSTEM_PROMPT_V2)
        wall = time.time() - t0
        samples = sampler.stop()
        util = [s[1] for s in samples]
        row = {
            "n": n, "episodes": len(trajs),
            "gen_tokens": meter["tokens"],
            "agg_tokens_per_sec": round(meter["tokens"] / wall, 1),
            "engine_tokens_per_sec": round(meter["tokens"] / meter["gen_s"], 1)
                                     if meter["gen_s"] else 0.0,
            "episodes_per_min": round(60 * len(trajs) / wall, 2),
            "gpu_util_mean": round(sum(util) / len(util), 1) if util else 0.0,
            "wall_seconds": round(wall, 1),
            "reward_mean": round(sum(t.reward for t in trajs) / len(trajs), 3),
            "violations": sum(t.violation for t in trajs),
        }
        rows.append(row)
        print(f"[SS4] {json.dumps(row)}")
        render_timeline(samples, [],
                        os.path.join(args.out4, f"ss04_timeline_n{n}.png"),
                        f"SS4 — {n} concurrent episodes | util mean "
                        f"{row['gpu_util_mean']}% | {row['agg_tokens_per_sec']} tok/s",
                        subtitle="lockstep batched rollouts, vLLM 0.7.3, APC on")

    by_n = {r["n"]: r for r in rows}
    speedup = round(by_n[16]["agg_tokens_per_sec"] /
                    max(by_n[1]["agg_tokens_per_sec"], 0.1), 2)
    m4 = (f"1→4→16-way agg tok/s: {by_n[1]['agg_tokens_per_sec']} → "
          f"{by_n[4]['agg_tokens_per_sec']} → {by_n[16]['agg_tokens_per_sec']} "
          f"({speedup}× at 16)")
    m4b = (f"GPU util {by_n[1]['gpu_util_mean']}% → {by_n[16]['gpu_util_mean']}%; "
           f"episodes/min {by_n[1]['episodes_per_min']} → "
           f"{by_n[16]['episodes_per_min']}")
    _bars(rows, args.out4, p4, m4)
    with open(os.path.join(args.out4, "result.json"), "w") as f:
        json.dump(rows, f, indent=2)
    write_manifest(args.out4, "SS4", "concurrent conversations vs throughput",
                   predicted=p4, measured=m4,
                   mechanism=("decode is memory-bandwidth-bound: batching N "
                              "sequences amortizes each weight read; env-step "
                              "gaps of one episode are filled by others'"),
                   config=vars(args),
                   rerun_cmd="VLLM_USE_V1=0 <venv>/bin/python "
                             "benchmarks/ss04_concurrency.py --predicted-ss4 '...' "
                             "--predicted-ss4b '...'")
    with open(os.path.join(args.out4b, "result.json"), "w") as f:
        json.dump({"sequential": by_n[1], "batched_16": by_n[16],
                   "speedup_tokens": speedup}, f, indent=2)
    write_manifest(args.out4b, "SS4b", "RL-loop utilization: sequential vs batched",
                   predicted=p4b, measured=m4b,
                   mechanism=("the single-GPU version of rollout/trainer "
                              "overlap: parallel episodes against one engine "
                              "fill env-step idle with useful decode"),
                   config=vars(args),
                   rerun_cmd="(same as SS4 — one run emits both)")
    print(f"[SS4] {m4}\n[SS4b] {m4b}")


def _bars(rows, out, predicted, measured) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = [str(r["n"]) for r in rows]
    tps = [r["agg_tokens_per_sec"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor="#fcfcfb")
    bars = ax.bar(ns, tps, color="#2a78d6", width=0.5)
    for b, r in zip(bars, rows):
        ax.annotate(f"{r['agg_tokens_per_sec']} tok/s\nutil {r['gpu_util_mean']}%",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9, color="#0b0b0b")
    ax.set_xlabel("concurrent episodes", color="#52514e")
    ax.set_ylabel("aggregate tokens/sec", color="#52514e")
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color="#e7e7e4", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.suptitle(f"SS4 — concurrency | predicted: {predicted}",
                 fontsize=10, x=0.01, ha="left")
    ax.set_title(f"measured: {measured}", fontsize=8.5, loc="left",
                 color="#52514e", pad=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(f"{out}/ss04_throughput.png", dpi=300, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
