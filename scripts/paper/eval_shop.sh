#!/bin/bash
# Paper protocol shopping eval — ONE checkpoint x ONE condition per job.
#   sbatch [slurm flags] scripts/paper/eval_shop.sh <model> <system> <label> <ckpt|base> [eval_v2 args...]
# e.g. eval_shop.sh Qwen/Qwen2.5-1.5B-Instruct v2 base15_paper base --n 128
# Outputs (json + manifest.json) -> /scratch/madan.pa/paper/eval/<label>/
set -euo pipefail
export HF_HOME=/scratch/madan.pa/.hf HF_HUB_OFFLINE=1 \
       PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd "$HOME/pennywise-v100-infra"
PY=/home/madan.pa/.conda/envs/shoprl/bin/python
MODEL="$1"; SYS="$2"; LABEL="$3"; CKPT="$4"; shift 4
CKPT_ARGS=()
if [ "$CKPT" != "base" ]; then
  # shared-ref nested layout: <dir> or <dir>/policy
  if [ ! -f "$CKPT/adapter_config.json" ] && [ -f "$CKPT/policy/adapter_config.json" ]; then
    CKPT="$CKPT/policy"
  fi
  CKPT_ARGS=(--ckpt "$CKPT")
fi
MNT=64; [[ "$MODEL" == *7B* ]] && MNT=128      # 7B chat answers run longer
OUT=/scratch/madan.pa/paper/eval/$LABEL
echo "[paper-eval] $LABEL model=$MODEL system=$SYS ckpt=$CKPT job=${SLURM_JOB_ID:-none}"
$PY scripts/eval_v2.py --model "$MODEL" ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"} \
    --system "$SYS" --languages es-en --max-new-tokens "$MNT" \
    --out "$OUT/shop.json" "$@" || true      # violations are data, not a crash
