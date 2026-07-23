#!/bin/bash
# Evaluate one H3 arm/stage: general-chat RETENTION + shopping SUCCESS.
#   sbatch ... scripts/eval7b.sh <adapter-dir> <label>
set -euo pipefail
export HF_HOME=/scratch/madan.pa/hf_cache HF_HUB_OFFLINE=1 \
       PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd "$HOME/pennywise-v100-infra"
PY=/home/madan.pa/.conda/envs/shoprl/bin/python
ADAPTER="$1"; LABEL="$2"
RET_ARGS=(); CKPT_ARGS=()
if [ "$ADAPTER" != "base" ]; then
  # resolve shared-ref nested layout: <dir> or <dir>/policy
  if [ ! -f "$ADAPTER/adapter_config.json" ] && [ -f "$ADAPTER/policy/adapter_config.json" ]; then
    ADAPTER="$ADAPTER/policy"
  fi
  RET_ARGS=(--adapter "$ADAPTER"); CKPT_ARGS=(--ckpt "$ADAPTER")
fi
echo "[eval7b] $LABEL adapter=$ADAPTER"
$PY scripts/eval_retention.py --label "$LABEL" ${RET_ARGS[@]+"${RET_ARGS[@]}"}
# shopping success on the REGISTERED H3 protocol: seed-5001 RL-difficulty split
$PY scripts/eval_v2.py --model Qwen/Qwen2.5-7B-Instruct ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"} \
    --system chat --languages es-en --n "${3:-128}" --max-new-tokens 128 \
    --scenario-seed 5001 --n-must-haves 3 4 --valid-range 3 10 \
    --out /scratch/madan.pa/pennypilot/eval/${LABEL}_shop.json || true
