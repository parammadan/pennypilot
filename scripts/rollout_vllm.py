"""vLLM rollout-throughput benchmark (Phase 3).

Drives multi-turn PennyEnv rollouts against a vLLM OpenAI-compatible server (the
served policy): N scenarios, C concurrent conversations. Each trajectory's turns
are sequential (turn t+1 needs turn t), so C = number of in-flight conversations
= concurrent requests vLLM batches via continuous batching.

Measures trajectories/sec and agent-generations/sec. For Step 3, pass multiple
comma-separated --endpoints to round-robin conversations across data-parallel
servers and measure 1-GPU vs 2-GPU scaling.

    python scripts/rollout_vllm.py --endpoints http://localhost:8000/v1 \
        --model pennywise-rloo --n 128 --concurrency 32
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import time

import httpx

from shoprl.data.catalog import catalog_index, generate_catalog
from shoprl.env.pennyenv import PennyEnv
from shoprl.env.scenario import generate_scenarios


async def gen_action(client, base_url, model, messages, max_tokens, temperature):
    r = await client.post(f"{base_url}/chat/completions", json={
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "top_p": 0.95})
    r.raise_for_status()
    txt = (r.json()["choices"][0]["message"]["content"] or "").strip()
    return txt.splitlines()[0].strip() if txt else ""


async def rollout(client, base_url, model, catalog, scenario, idx, *, max_turns,
                  max_tokens, temperature):
    env = PennyEnv(catalog, scenario, idx=idx, max_turns=max_turns)
    opener = env.reset()
    messages = [{"role": "user", "content": opener}]
    done, gens = False, 0
    while not done:
        action = await gen_action(client, base_url, model, messages, max_tokens,
                                  temperature)
        gens += 1
        user, done, _ = env.step(action)
        messages += [{"role": "assistant", "content": action},
                     {"role": "user", "content": user}]
    r = env.reward()
    return {"value": r.value_quality, "gens": gens, "reward": r.total,
            "violation": r.acted_without_permission}


async def main_async(args):
    catalog = generate_catalog(n=args.catalog_size, seed=0)
    idx = catalog_index(catalog)
    scen = generate_scenarios(catalog, n=args.n, seed=args.seed)
    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    ep = itertools.cycle(endpoints)
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=180,
                                 limits=httpx.Limits(max_connections=args.concurrency + 8)) as client:
        async def worker(s, base):
            async with sem:
                return await rollout(client, base, args.model, catalog, s, idx,
                                     max_turns=args.max_turns,
                                     max_tokens=args.max_tokens,
                                     temperature=args.temperature)
        # warmup one trajectory (loads/pages the model) not counted
        await worker(scen[0], endpoints[0])
        t0 = time.time()
        results = await asyncio.gather(*[asyncio.create_task(worker(s, next(ep)))
                                         for s in scen])
        dt = time.time() - t0

    n = len(results)
    gens = sum(r["gens"] for r in results)
    val = sum(r["value"] for r in results) / n
    viol = sum(r["violation"] for r in results) / n
    print(f"[vllm-rollout] endpoints={len(endpoints)} concurrency={args.concurrency} n={n}")
    print(f"  wall={dt:.1f}s  trajectories/sec={n/dt:.3f}  agent-gens/sec={gens/dt:.2f}")
    print(f"  mean_value={val:.3f}  violation_rate={viol:.3f}  (sanity: behaviour preserved via vLLM)")
    return {"endpoints": len(endpoints), "concurrency": args.concurrency, "n": n,
            "wall_s": round(dt, 2), "traj_per_s": round(n / dt, 3),
            "gens_per_s": round(gens / dt, 2), "mean_value": round(val, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", default="http://localhost:8000/v1",
                    help="comma-separated vLLM OpenAI base URLs (>=1 for data-parallel)")
    ap.add_argument("--model", default="pennywise-rloo")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--catalog-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=3000)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
