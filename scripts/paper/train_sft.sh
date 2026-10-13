#!/bin/bash
# SFT one arm x seed (7B LoRA, chat system prompt).
#   train_sft.sh <arm> <seed> <rehearsal_n>   (rehearsal_n=0 => specialist)
set -euo pipefail
source "$HOME/pennywise-v100-infra/scripts/paper/common.sh"
ARM="$1"; SEED="$2"; RN="$3"
OUT=$SCR/runs/${ARM}__s${SEED}/sft
REH_ARGS=(); [ "$RN" -gt 0 ] && REH_ARGS=(--rehearsal "$REH" --rehearsal-n "$RN")
$PY scripts/run_sft_v2.py --model $M7 --method lora --system chat \
    --demos 540 --seed 0 --train-seed "$SEED" --lr 1e-5 --epochs 1 \
    --batch-size 2 --max-len 1024 --out-dir "$OUT" ${REH_ARGS[@]+"${REH_ARGS[@]}"}
mirror "$OUT"
