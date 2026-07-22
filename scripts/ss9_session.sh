#!/bin/bash
# SS9-SS10 data-path session (trainer env). Predictions stated before the run.
set -uo pipefail
cd "$HOME/pennywise-v100-infra"
source env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
ART=/scratch/madan.pa/pennypilot/artifacts
mkdir -p "$ART/ss09" "$ART/ss10"

echo "=== SS9: length-bucketing vs shuffled padding ==="
python benchmarks/ss09_packing.py --out "$ART/ss09" \
  --predicted "shuffled pad waste 25-40%; bucketing cuts it under 10% and lifts real-token throughput 1.15-1.3x" \
  2>&1 | tee "$ART/ss09/run_${SLURM_JOB_ID:-x}.log" || exit 1

echo "=== SS10: dataloader gap ==="
python benchmarks/ss10_prefetch.py --out "$ART/ss10" \
  --predicted "data path <3% of step time (pre-tokenized up front) -> expected NO-GAP artifact; prefetch not justified" \
  2>&1 | tee "$ART/ss10/run_${SLURM_JOB_ID:-x}.log" || exit 1

echo "=== SS9-10 SESSION COMPLETE ==="
ls "$ART/ss09" "$ART/ss10"
