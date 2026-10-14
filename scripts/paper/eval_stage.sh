#!/bin/bash
# All paper metrics for one checkpoint: shopping (held-out catalog), 40-Q
# retention, MMLU + IFEval.   eval_stage.sh <arm> <seed> <sft|rl|base>
set -euo pipefail
source "$HOME/pennywise-v100-infra/scripts/paper/common.sh"
ARM="$1"; SEED="$2"; STAGE="$3"
LABEL=${ARM}-${STAGE}__s${SEED}__paper
if [ "$STAGE" = base ]; then CK=base; else CK=$SCR/runs/${ARM}__s${SEED}/$STAGE/policy; fi
bash scripts/paper/eval_shop.sh $M7 chat "$LABEL" "$CK" --split paper --n 128 --seed "$SEED"
RA=(); [ "$CK" != base ] && { A=$CK; [ -f "$A/adapter_config.json" ] || A=$A/policy; RA=(--adapter "$A"); }
$PY scripts/eval_retention.py --model $M7 --label "$LABEL" ${RA[@]+"${RA[@]}"} \
    --out "$SCR/eval/$LABEL/retention.json"
bash scripts/paper/eval_general.sh $M7 "$LABEL" "$CK" 10 200
