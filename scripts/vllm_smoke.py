"""vLLM-on-V100 smoke test (GPU) — run once per candidate pin, in ITS venv.

    python scripts/vllm_smoke.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --gpu-mem-util 0.45 --out runs/gate/vllm_smoke_<pin>.json

PASS = engine initializes on Volta, generates a 16-prompt batch, and prefix
caching toggles without error. Exit 0 on pass, 2 on fail — the session
orchestrator walks the candidate list until one passes. Record the winning
pin + flags in the docs repo (RUN_ON_SLURM.md §9) and FREEZE it for Stage 6.
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gpu-mem-util", type=float, default=0.45,
                    help="low on purpose: the trainer must coexist in 32GB")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", default="runs/gate/vllm_smoke.json")
    args = ap.parse_args()

    report: dict = {"model": args.model, "gpu_mem_util": args.gpu_mem_util}
    try:
        import vllm
        report["vllm_version"] = vllm.__version__
        from vllm import LLM, SamplingParams

        def build(prefix_caching: bool) -> "LLM":
            return LLM(model=args.model, dtype="float16",
                       gpu_memory_utilization=args.gpu_mem_util,
                       enable_prefix_caching=prefix_caching,
                       enforce_eager=True)  # skip CUDA-graph capture on smoke

        t0 = time.time()
        llm = build(prefix_caching=False)
        report["engine_init_seconds"] = round(time.time() - t0, 1)

        prompts = [f"List one budget laptop for user {i} in one sentence."
                   for i in range(args.n)]
        sp = SamplingParams(temperature=0.8, max_tokens=args.max_tokens)
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        wall = time.time() - t0
        toks = sum(len(o.outputs[0].token_ids) for o in outs)
        report["batch_generate"] = {
            "prompts": args.n, "gen_tokens": toks,
            "tokens_per_sec": round(toks / wall, 1),
            "wall_seconds": round(wall, 2),
        }
        del llm

        t0 = time.time()
        llm = build(prefix_caching=True)
        llm.generate(prompts[:2], sp)
        report["prefix_caching"] = {"init_and_generate_ok": True,
                                    "wall_seconds": round(time.time() - t0, 1)}
        report["verdict"] = "PASS"
        rc = 0
    except Exception as e:  # noqa: BLE001 — the failure IS the measurement
        report["verdict"] = f"FAIL: {type(e).__name__}: {e}"
        rc = 2

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())
