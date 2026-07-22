# SS7 — gradient-checkpointing toggle

- **Predicted (stated before the run):** ckpt-OFF >= 1.5x peak memory of ckpt-ON; ckpt-OFF 1.25-1.45x faster (no recompute forward)
- **Measured:** ckpt-ON 7.34 GB @ 3028.1 tok/s; ckpt-OFF 16.36 GB @ 4485.7 tok/s (off = 2.23× memory, 1.48× speed)
- **Mechanism:** checkpointing recomputes activations in backward (extra forward ≈ +30-40% step time) instead of storing them (activation memory ∝ layers)
- **Config:** `{"batch_size": 2, "max_len": 1024, "out": "/scratch/madan.pa/pennypilot/artifacts/ss07", "predicted": "ckpt-OFF >= 1.5x peak memory of ckpt-ON; ckpt-OFF 1.25-1.45x faster (no recompute forward)", "steps": 20}` (hash `6803d53c`)
- **W&B:** (offline / not logged)
- **Rerun:** `python benchmarks/ss07_grad_ckpt.py --predicted '...'`
- **Captured:** 2026-07-22 07:13:50

## Artifacts
- `result.json`
- `run_8549109.log`
- `ss07_toggle.png`
