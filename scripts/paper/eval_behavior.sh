#!/bin/bash
# Tool-decision + prompt probes + answer correctness + LoRA sweep for one ckpt.
#   sbatch [flags] eval_behavior.sh <arm> <seed> <sft|rl|base> [model]
set -euo pipefail
source "$HOME/pennywise-v100-infra/scripts/paper/common.sh"
ARM="$1"; SEED="$2"; STAGE="$3"; MODEL="${4:-$M7}"
LABEL=${ARM}-${STAGE}__s${SEED}__paper
if [ "$STAGE" = base ]; then A=base; else
  A=$KEEP/${ARM}__s${SEED}/$STAGE
  [ -f "$A/adapter_config.json" ] || A=$A/policy
  [ -f "$A/adapter_config.json" ] || A=$A/policy
fi
$PY scripts/paper/eval_behavior.py --model "$MODEL" --adapter "$A" \
    --out "$SCR/eval/$LABEL/behavior/behavior.json"
