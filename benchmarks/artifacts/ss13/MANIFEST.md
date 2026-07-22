# SS13 — iteration waterfall, rung by rung

- **Predicted (stated before the run):** final rung >= 4x faster than baseline; batched rollouts contribute nearly all of it; engine swap alone ~nothing
- **Measured:** 93.2s → 42.7s → 15.1s → 13.4s (6.96× total)
- **Mechanism:** nearly all the win is batched rollouts filling env gaps + amortizing weight reads; engine swap alone and data-path tweaks measured ≈ nothing (SS2/SS9/SS10); ckpt-off trims the update
- **Config:** `{"k": 8, "merged": true}` (hash `badbc14b`)
- **W&B:** (offline / not logged)
- **Rerun:** `scripts/ss13_session.sh (staged)`
- **Captured:** 2026-07-22 07:58:08

## Artifacts
- `result.json`
- `rollout_hf.json`
- `rollout_vllm.json`
- `ss13_waterfall.png`
- `update.json`
