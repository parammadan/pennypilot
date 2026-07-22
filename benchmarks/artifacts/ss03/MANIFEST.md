# SS3 — per-turn latency (fixed 64-tok decode), APC off vs on

- **Predicted (stated before the run):** APC-off per-turn latency grows with turns once decode is constant; APC-on Volta-gated (documented)
- **Measured:** APC-off per-turn latency grows 879.7→936.7 ms (turns 2→9); **APC-on UNAVAILABLE on Volta**: vLLM 0.7.3's prefix-cache prefill kernel (Triton) aborts — 'mma layout conversion only supported on Ampere'
- **Mechanism:** prefill work grows with conversation length (measured); the caching fix requires a prefix-prefill kernel this hardware cannot compile — the optimization is documented as hardware-gated, per the campaign rule that a blocked path is itself a reportable artifact
- **Config:** `{"gpu_mem_util": 0.45, "mode": "plot", "model": "Qwen/Qwen2.5-1.5B-Instruct", "out": "benchmarks/artifacts/ss03", "predicted": "APC-off per-turn latency grows with turns once decode is constant; APC-on Volta-gated (documented)", "sft_adapter": null}` (hash `e08002f0`)
- **W&B:** (offline / not logged)
- **Rerun:** `VLLM_USE_V1=0 <venv>/bin/python benchmarks/ss03_prefix_cache.py --mode apc_off|apc_on|plot --predicted '...'`
- **Captured:** 2026-07-22 06:58:54

## Artifacts
- `apc_off.json`
- `result.json`
- `run_8548620.log`
- `run_8548661.log`
- `run_8548778.log`
- `run_8548825.log`
- `run_off_8548876.log`
- `run_on_8548876.log`
- `ss03_prefill.csv`
- `ss03_prefill.png`
