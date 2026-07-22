#!/bin/bash
# SS0-SS2 measurement session (sbatch). Predictions were stated before this
# run (2026-07-21, see each --predicted below and STAGE6_BENCHMARKS.md).
#   sbatch --partition=gpu-short --gres=gpu:v100-sxm2:1 --cpus-per-task=8 \
#          --mem=64G --time=01:00:00 --job-name=ss00-02 \
#          --output=/scratch/madan.pa/pennypilot/ss_session_%j.log \
#          ~/pennywise-v100-infra/scripts/benchmarks_session.sh
set -uo pipefail
cd "$HOME/pennywise-v100-infra"
source env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
POLICY=/scratch/madan.pa/pennypilot/rloo50_v2/policy
ART=benchmarks/artifacts
STAMP=${SLURM_JOB_ID:-$(date +%s)}   # unique per job: same-minute twins collided once
mkdir -p "$ART/ss00" "$ART/ss01" "$ART/ss02"

echo "=== SS0: baseline RL-iteration phase split ==="
python benchmarks/ss00_baseline.py --sft-adapter "$POLICY" \
  --predicted "rollout >= 80% of iteration wall-clock; update < 15%" \
  2>&1 | tee "$ART/ss00/run_$STAMP.log" || exit 1

echo "=== SS1: HF rollout engine tokens/sec ==="
python benchmarks/ss01_rollout_hf.py --sft-adapter "$POLICY" \
  --predicted "engine 30-60 tok/s single-stream (1.5B fp16 batch-1 on V100)" \
  2>&1 | tee "$ART/ss01/run_$STAMP.log" || exit 1

echo "=== SS2: vLLM rollout engine, same loop (pinned venv, no cuda module) ==="
env -u LD_LIBRARY_PATH VLLM_USE_V1=0 HF_HOME=/scratch/madan.pa/hf_cache HF_HUB_OFFLINE=1 \
  /scratch/madan.pa/venvs/vllm-073/bin/python benchmarks/ss02_rollout_vllm.py \
  --sft-adapter "$POLICY" \
  --predicted "1.5-3x SS1 engine tok/s at batch=1; big wins deferred to concurrency (SS4)" \
  2>&1 | tee "$ART/ss02/run_$STAMP.log" || exit 1

echo "=== SS SESSION COMPLETE ==="
for d in ss00 ss01 ss02; do echo "--- $d"; ls "$ART/$d"; done
