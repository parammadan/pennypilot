#!/bin/bash
# SS13 remaining stages (merge + rollout_hf already banked on scratch).
set -uo pipefail
cd "$HOME/pennywise-v100-infra"
source env.sh
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
ART=/scratch/madan.pa/pennypilot/artifacts
env -u LD_LIBRARY_PATH VLLM_USE_V1=0 HF_HOME=/scratch/madan.pa/hf_cache HF_HUB_OFFLINE=1 \
  /scratch/madan.pa/venvs/vllm-073/bin/python benchmarks/ss13_waterfall.py \
  --stage rollout_vllm --out "$ART/ss13" || exit 1
python benchmarks/ss13_waterfall.py --stage update --out "$ART/ss13" || exit 1
python benchmarks/ss13_waterfall.py --stage assemble --out "$ART/ss13" \
  --predicted "final rung >= 4x faster than baseline; batched rollouts contribute nearly all of it; engine swap alone ~nothing" || exit 1
echo "=== SS13 COMPLETE ==="
