# SS9 — length-bucketing vs shuffled padding

- **Predicted (stated before the run):** shuffled pad waste 25-40%; bucketing cuts it under 10% and lifts real-token throughput 1.15-1.3x
- **Measured:** pad waste 8.2% → 0.3%; real-token throughput 3418.2 → 3378.0 tok/s (0.99×)
- **Mechanism:** batch width = longest member; mixing lengths pads everyone to the outlier — sorting by length makes width ≈ members' own length
- **Config:** `{"batch_size": 4, "max_len": 1024, "out": "/scratch/madan.pa/pennypilot/artifacts/ss09", "predicted": "shuffled pad waste 25-40%; bucketing cuts it under 10% and lifts real-token throughput 1.15-1.3x", "steps": 50}` (hash `126a7ce2`)
- **W&B:** (offline / not logged)
- **Rerun:** `python benchmarks/ss09_packing.py --predicted '...'`
- **Captured:** 2026-07-22 07:26:15

## Artifacts
- `result.json`
- `run_8549188.log`
- `run_8549823.log`
- `ss09_padding.png`
