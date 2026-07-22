#!/bin/bash
# SS6-SS8 memory measurement session (trainer env, HF path — no Volta-Triton
# exposure). Predictions stated below, before the run.
set -uo pipefail
cd "$HOME/pennywise-v100-infra"
source env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
POLICY=/scratch/madan.pa/pennypilot/rloo50_v2/policy
mkdir -p benchmarks/artifacts/ss06 benchmarks/artifacts/ss07 benchmarks/artifacts/ss08

echo "=== SS6: memory staircase ==="
python benchmarks/ss06_memory_staircase.py --sft-adapter "$POLICY" \
  --predicted "peak <= 14 GB; staircase dominated by two ~3 GB fp16 model loads; LoRA optimizer state invisible at GB scale" \
  2>&1 | tee benchmarks/artifacts/ss06/run_${SLURM_JOB_ID:-x}.log || exit 1

echo "=== SS7: gradient-checkpointing toggle ==="
python benchmarks/ss07_grad_ckpt.py \
  --predicted "ckpt-OFF >= 1.5x peak memory of ckpt-ON; ckpt-OFF 1.25-1.45x faster (no recompute forward)" \
  2>&1 | tee benchmarks/artifacts/ss07/run_${SLURM_JOB_ID:-x}.log || exit 1

echo "=== SS8: LoRA vs full-FT optimizer memory ==="
python benchmarks/ss08_lora_vs_full.py \
  --predicted "optimizer state ratio >= 40x (full-FT AdamW fp32 moments for 1.54B vs LoRA ~18M); full-FT backward peak >= 2x LoRA" \
  2>&1 | tee benchmarks/artifacts/ss08/run_${SLURM_JOB_ID:-x}.log || exit 1

echo "=== SS6-8 SESSION COMPLETE ==="
ls benchmarks/artifacts/ss06 benchmarks/artifacts/ss07 benchmarks/artifacts/ss08
