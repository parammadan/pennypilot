# SS0 — baseline RL-iteration phase split

- **Predicted (stated before the run):** rollout >= 80% of iteration wall-clock; update < 15%
- **Measured:** rollout 95.8% / update 4.0% / optimizer 0.1%
- **Mechanism:** autoregressive rollout decode is memory-bandwidth-bound and sequential per turn; the update is a handful of dense fwd/bwd passes
- **Config:** `{"k": 8, "language": "es-en", "n_must_haves": [3, 4], "out": "benchmarks/artifacts/ss00", "predicted": "rollout >= 80% of iteration wall-clock; update < 15%", "sft_adapter": "/scratch/madan.pa/pennypilot/rloo50_v2/policy", "valid_range": [3, 10]}` (hash `90d63c9`)
- **W&B:** (offline / not logged)
- **Rerun:** `python benchmarks/ss00_baseline.py --sft-adapter /scratch/madan.pa/pennypilot/rloo50_v2/policy --predicted '...'`
- **Captured:** 2026-07-21 21:29:17

## Artifacts
- `events.csv`
- `run_8536377.log`
- `run_8536566.log`
- `ss00_phases.json`
- `ss00_phases.png`
- `ss00_timeline.csv`
- `ss00_timeline.png`
- `step_metrics.json`
