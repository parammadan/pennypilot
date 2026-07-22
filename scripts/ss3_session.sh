#!/bin/bash
# SS3 + SS4/SS4b measurement session (vLLM venv, ~30 min). Predictions below,
# stated before the run per the campaign rule.
set -uo pipefail
cd "$HOME/pennywise-v100-infra"
export HF_HOME=/scratch/madan.pa/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH
VPY=/scratch/madan.pa/venvs/vllm-073/bin/python
POLICY=/scratch/madan.pa/pennypilot/rloo50_v2/policy
mkdir -p benchmarks/artifacts/ss03 benchmarks/artifacts/ss04 benchmarks/artifacts/ss04b

echo "=== SS3: prefix caching ==="
# NOTE: no --sft-adapter — vLLM 0.7.3 LoRA Triton kernels abort on Volta
# (mma->mma layout needs Ampere). SS3/SS4 measure engine physics (prefill
# growth, decode batching), identical with/without an adapter; documented.
PRED3="APC-off per-turn latency GROWS with turns; APC-on flattens it OR is Volta-blocked (Triton kernel needs Ampere) — either outcome reported"
VLLM_USE_V1=0 $VPY benchmarks/ss03_prefix_cache.py --mode apc_off --predicted "$PRED3" \
  2>&1 | tee benchmarks/artifacts/ss03/run_off_${SLURM_JOB_ID:-x}.log || exit 1
VLLM_USE_V1=0 $VPY benchmarks/ss03_prefix_cache.py --mode apc_on --predicted "$PRED3" \
  2>&1 | tee benchmarks/artifacts/ss03/run_on_${SLURM_JOB_ID:-x}.log \
  || echo "APC-on crashed as predicted-possible — documenting as hardware-gated"
VLLM_USE_V1=0 $VPY benchmarks/ss03_prefix_cache.py --mode plot --predicted "$PRED3" || exit 1

echo "=== SS4 + SS4b: concurrency + utilization ==="
VLLM_USE_V1=0 $VPY benchmarks/ss04_concurrency.py --no-prefix-caching \
  --predicted-ss4 "16-way >= 5x aggregate tok/s vs 1-way (decode batching amortizes weight reads)" \
  --predicted-ss4b "GPU util <35% sequential -> >=70% at 16-way; episodes/min >= 5x" \
  2>&1 | tee benchmarks/artifacts/ss04/run_${SLURM_JOB_ID:-x}.log || exit 1

echo "=== SS3/4 SESSION COMPLETE ==="
ls benchmarks/artifacts/ss03 benchmarks/artifacts/ss04 benchmarks/artifacts/ss04b
