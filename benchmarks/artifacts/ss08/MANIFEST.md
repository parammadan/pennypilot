# SS8 — LoRA vs full-FT memory anatomy

- **Predicted (stated before the run):** optimizer state ratio >= 40x (full-FT AdamW fp32 moments for 1.54B vs LoRA ~18M); full-FT backward peak >= 2x LoRA
- **Measured:** optimizer state: LoRA 0.138 GB vs full-FT 5.751 GB (42×); trainable 18.5M vs 1543.7M; backward peak 7.19 vs 17.85 GB
- **Mechanism:** AdamW keeps 2 fp32 moments per trainable param (+fp32 master under mixed precision): full-FT pays it for 1.54B params, LoRA for ~18M — the note that full-FT also needs a SEPARATE frozen reference for KL while LoRA-from-base gets it free stays in ARCHITECTURE.md
- **Config:** `{"out": "/scratch/madan.pa/pennypilot/artifacts/ss08", "predicted": "optimizer state ratio >= 40x (full-FT AdamW fp32 moments for 1.54B vs LoRA ~18M); full-FT backward peak >= 2x LoRA"}` (hash `adb55d96`)
- **W&B:** (offline / not logged)
- **Rerun:** `python benchmarks/ss08_lora_vs_full.py --predicted '...'`
- **Captured:** 2026-07-22 07:14:23

## Artifacts
- `result.json`
- `run_8549109.log`
