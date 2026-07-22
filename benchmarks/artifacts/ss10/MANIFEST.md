# SS10 — dataloader-gap measurement

- **Predicted (stated before the run):** data path <3% of step time (pre-tokenized up front) -> expected NO-GAP artifact; prefetch not justified
- **Measured:** data path 0.10s vs compute 52.74s over 50 steps = 0.20% of step time (one-time pre-tokenization 4.7s) → NO-GAP artifact: the dataloader is not a hotspot — pre-tokenization already eliminated it; prefetch machinery would optimize 0.20% of the loop
- **Mechanism:** all tokenization happens once up front; per-step data work is a python list slice + pad + one H2D copy of ~4k ints
- **Config:** `{"batch_size": 4, "max_len": 1024, "out": "/scratch/madan.pa/pennypilot/artifacts/ss10", "predicted": "data path <3% of step time (pre-tokenized up front) -> expected NO-GAP artifact; prefetch not justified", "steps": 50}` (hash `addd9985`)
- **W&B:** (offline / not logged)
- **Rerun:** `python benchmarks/ss10_prefetch.py --predicted '...'`
- **Captured:** 2026-07-22 07:28:03

## Artifacts
- `result.json`
- `run_8549823.log`
