# SS2 — vLLM rollout throughput (same loop as SS1)

- **Predicted (stated before the run):** 1.5-3x SS1 engine tok/s at batch=1; big wins deferred to concurrency (SS4)
- **Measured:** engine 19.3 tok/s, loop 19.3 tok/s
- **Mechanism:** PagedAttention KV management + persistent engine remove per-call model setup; batch=1 sequential turns still leave batching headroom (SS4's subject)
- **Config:** `{"episodes": 16, "gpu_mem_util": 0.45, "language": "es-en", "model": "Qwen/Qwen2.5-1.5B-Instruct", "out": "benchmarks/artifacts/ss02", "predicted": "1.5-3x SS1 engine tok/s at batch=1; big wins deferred to concurrency (SS4)", "prefix_caching": false, "sft_adapter": "/scratch/madan.pa/pennypilot/rloo50_v2/policy"}` (hash `20e84e12`)
- **W&B:** (offline / not logged)
- **Rerun:** `VLLM_USE_V1=0 <venv>/bin/python benchmarks/ss02_rollout_vllm.py --predicted '...'`
- **Captured:** 2026-07-21 21:37:47

## Artifacts
- `result.json`
- `run_8536377.log`
- `run_8536566.log`
- `ss02_timeline.csv`
- `ss02_timeline.png`
