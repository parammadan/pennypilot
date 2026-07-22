# SS4 — concurrent conversations vs throughput

- **Predicted (stated before the run):** 16-way >= 5x aggregate tok/s vs 1-way (decode batching amortizes weight reads)
- **Measured:** 1→4→16-way agg tok/s: 62.7 → 148.0 → 312.1 (4.98× at 16)
- **Mechanism:** decode is memory-bandwidth-bound: batching N sequences amortizes each weight read; env-step gaps of one episode are filled by others'
- **Config:** `{"gpu_mem_util": 0.45, "levels": [1, 4, 16], "model": "Qwen/Qwen2.5-1.5B-Instruct", "out4": "benchmarks/artifacts/ss04", "out4b": "benchmarks/artifacts/ss04b", "predicted_ss4": "16-way >= 5x aggregate tok/s vs 1-way (decode batching amortizes weight reads)", "predicted_ss4b": "GPU util <35% sequential -> >=70% at 16-way; episodes/min >= 5x", "prefix_caching": false, "sft_adapter": null}` (hash `b151f7e9`)
- **W&B:** (offline / not logged)
- **Rerun:** `VLLM_USE_V1=0 <venv>/bin/python benchmarks/ss04_concurrency.py --predicted-ss4 '...' --predicted-ss4b '...'`
- **Captured:** 2026-07-22 06:55:31

## Artifacts
- `result.json`
- `run_8548876.log`
- `ss04_throughput.png`
- `ss04_timeline_n1.csv`
- `ss04_timeline_n1.png`
- `ss04_timeline_n16.csv`
- `ss04_timeline_n16.png`
- `ss04_timeline_n4.csv`
- `ss04_timeline_n4.png`
