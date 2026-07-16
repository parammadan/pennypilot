# Phase 3 status — efficiency + observability (AWS A10G)

Honest record of Phase 3 (rollout throughput + observability). Companion to
`PHASE2_RESULTS.md`. _Last updated: 2026-07-16._

## Done ✅

### Step 1 — S3 bridge (NEU ↔ AWS)
- Bucket `s3://pennywise-794528241070-us-east-2` (us-east-2, private).
- RLOO checkpoint pushed cluster→S3 (`checkpoints/rloo50/policy/`, `model.safetensors`
  full 3,087,467,144 B), verified by a fresh-dir pull-back.
- Push done **directly cluster→S3** via a **scoped, S3-only IAM user**
  (`pennywise-cluster`) — not root creds. (Mac relay failed repeatedly on the
  home link; direct cluster→S3 at ~65 MB/s is the reliable path and matches the
  intended `V100 → S3` design.)

### Step 4 — observability dashboard
- `src/shoprl/observability/` + `results/dashboard.html` (self-contained, replay
  on real Phase-2 metrics). MetricsSource abstraction (static / live-tail /
  AWS-stub) so live data connects by swapping the source, not a rebuild. Training
  dynamics fully populated; system-health panels are live-ready placeholders.
  Alerts fire correctly (KL-blowup→CRITICAL, reward-stall→WARNING). 31 tests.

### Guardrails
- AWS Budget `pennywise-phase3-guard` $25/mo, email alerts at $10/$20/forecast.
- Every launch: spot, **EBS delete-on-terminate**, **`instance-initiated-shutdown-behavior=terminate`
  + user-data `shutdown -h +90`** (dead-man's switch — self-terminates ≤90 min
  regardless of failures).

## Blocked ⚠️ — Steps 2 & 3 (A10G rollout throughput)

Two independent external blockers:

1. **Spot capacity.** Spot G quota is 8 vCPU in **us-east-2 only** (every other
   region defaults to 0; increase requests + on-demand appeal all PENDING/DENIED).
   us-east-2 A10G spot capacity is **intermittent** — dry during the day; the one
   successful launch was **07:30 ET (off-peak)**. Background auto-hunt loops are
   reaped by the dev sandbox within seconds, so hunting is manual bursts at
   off-peak.
2. **vLLM serving.** On the one box we caught: instance role → S3 pull of
   checkpoint + code, bootstrap, and model load **all worked**, but the DLAMI's
   default `pip install vllm` (v0.25.1, bleeding edge) failed engine init — its
   flashinfer JIT sampler wouldn't build, and the v1 engine hard-requires
   flashinfer. Terminated (~$0.35) rather than burn the billing clock.

## Staged for a clean retry (next off-peak box → fast result)
- Checkpoint + code (both `scripts/rollout_vllm.py` and guaranteed
  `scripts/rollout_hf.py`) in S3.
- vLLM fix to try first: `TORCH_CUDA_ARCH_LIST=8.6` (A10G arch) for the flashinfer
  JIT build; if it still balks, `rollout_hf.py` (transformers batched generation,
  no vLLM) guarantees a throughput number.
- Reusable AWS scaffolding: IAM instance role (S3 read), key pair, security group,
  pinned DLAMI `ami-032b37e4db407994d`.

## Honest takeaway
The **core deliverable (the RLOO post-training pipeline) is complete** (Phase 2).
Phase 3's bridge + dashboard are done; the A10G **throughput benchmark is a
nice-to-have gated on intermittent AWS spot capacity** — attempted, documented,
and staged to finish quickly the next time an off-peak GPU is available.
