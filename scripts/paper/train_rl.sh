#!/bin/bash
# 50-step shared-ref RLOO from this arm x seed's SFT adapter. Requeue-safe.
#   train_rl.sh <arm> <seed>
set -euo pipefail
source "$HOME/pennywise-v100-infra/scripts/paper/common.sh"
ARM="$1"; SEED="$2"
RUN=$SCR/runs/${ARM}__s${SEED}
$PY scripts/run_rl_v2.py --model $M7 --sft-adapter "$RUN/sft/policy" \
    --system chat --shared-ref --algo rloo --steps 50 --k 8 --lr 1e-6 \
    --beta 0.04 --max-new-tokens 128 --train-seed "$SEED" \
    --out-dir "$RUN/rl" --resume latest
mirror "$RUN/rl"
