# SS11 — update-phase re-profile (kernel gate)

- **Predicted (stated before the run):** PG-math chain < 1% of the full iteration (update itself is ~4%) -> Triton kernel NOT justified; the documented no-go IS the artifact
- **Measured:** update 11.6s; PG-math chain ≈ 20.0% of update CUDA time ≈ 0.84% of the full iteration → NO KERNEL-WORTHY HOTSPOT — the PG-math chain is 0.84% of the iteration; SS12 = the documented decision NOT to write the kernel
- **Mechanism:** the update is model fwd/bwd GEMM-dominated; the elementwise PG chain is tiny and already fused reasonably by eager kernels
- **Config:** `{"k": 8}` (hash `1b3c249b`)
- **W&B:** (offline / not logged)
- **Rerun:** `python benchmarks/ss11_update_reprofile.py --sft-adapter ... --predicted '...'`
- **Captured:** 2026-07-22 07:42:06

## Artifacts
- `result.json`
- `run_8553227.log`
- `update_trace.json`
