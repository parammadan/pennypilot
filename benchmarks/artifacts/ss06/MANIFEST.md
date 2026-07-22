# SS6 — memory staircase of one RL iteration

- **Predicted (stated before the run):** peak <= 14 GB; staircase dominated by two ~3 GB fp16 model loads; LoRA optimizer state invisible at GB scale
- **Measured:** NVML peak 10.77 GB (torch alloc peak 7.92 GB); staircase: 2 model loads then rollout/update plateaus
- **Mechanism:** two fp16 model copies dominate the floor; rollout adds KV cache; backward adds activation peak; optimizer state is tiny under LoRA
- **Config:** `{"k": 4, "out": "/scratch/madan.pa/pennypilot/artifacts/ss06", "predicted": "peak <= 14 GB; staircase dominated by two ~3 GB fp16 model loads; LoRA optimizer state invisible at GB scale", "sft_adapter": "/scratch/madan.pa/pennypilot/rloo50_v2/policy"}` (hash `c35de9fb`)
- **W&B:** (offline / not logged)
- **Rerun:** `python benchmarks/ss06_memory_staircase.py --sft-adapter /scratch/madan.pa/pennypilot/rloo50_v2/policy --predicted '...'`
- **Captured:** 2026-07-22 07:13:00

## Artifacts
- `events.csv`
- `result.json`
- `run_8549109.log`
- `ss06_staircase.csv`
- `ss06_staircase.png`
