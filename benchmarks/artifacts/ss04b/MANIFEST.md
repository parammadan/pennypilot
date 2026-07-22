# SS4b — RL-loop utilization: sequential vs batched

- **Predicted (stated before the run):** GPU util <35% sequential -> >=70% at 16-way; episodes/min >= 5x
- **Measured:** GPU util 48.5% → 68.4%; episodes/min 8.12 → 51.75
- **Mechanism:** the single-GPU version of rollout/trainer overlap: parallel episodes against one engine fill env-step idle with useful decode
- **Config:** `{"gpu_mem_util": 0.45, "levels": [1, 4, 16], "model": "Qwen/Qwen2.5-1.5B-Instruct", "out4": "benchmarks/artifacts/ss04", "out4b": "benchmarks/artifacts/ss04b", "predicted_ss4": "16-way >= 5x aggregate tok/s vs 1-way (decode batching amortizes weight reads)", "predicted_ss4b": "GPU util <35% sequential -> >=70% at 16-way; episodes/min >= 5x", "prefix_caching": false, "sft_adapter": null}` (hash `b151f7e9`)
- **W&B:** (offline / not logged)
- **Rerun:** `(same as SS4 — one run emits both)`
- **Captured:** 2026-07-22 06:55:31

## Artifacts
- `result.json`
