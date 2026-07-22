# SS1 — HF .generate() rollout throughput

- **Predicted (stated before the run):** engine 30-60 tok/s single-stream (1.5B fp16 batch-1 on V100)
- **Measured:** engine 18.3 tok/s, loop 18.3 tok/s
- **Mechanism:** batch-1 sequential decode: every token streams all weights; no batching to amortize reads
- **Config:** `{"episodes": 16, "language": "es-en", "out": "benchmarks/artifacts/ss01", "predicted": "engine 30-60 tok/s single-stream (1.5B fp16 batch-1 on V100)", "sft_adapter": "/scratch/madan.pa/pennypilot/rloo50_v2/policy"}` (hash `d645595b`)
- **W&B:** (offline / not logged)
- **Rerun:** `python benchmarks/ss01_rollout_hf.py --sft-adapter /scratch/madan.pa/pennypilot/rloo50_v2/policy --predicted '...'`
- **Captured:** 2026-07-21 21:33:13

## Artifacts
- `result.json`
- `run_8536377.log`
- `run_8536566.log`
- `ss01_timeline.csv`
- `ss01_timeline.png`
