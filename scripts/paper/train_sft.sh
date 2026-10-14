#!/bin/bash
# SFT one arm x seed (7B LoRA, chat system prompt).
# max-len 2048: measured untruncated demo lengths p99 1574 / max 1606 tokens
# (1024 truncated 219/540 demos, cutting the permission + cart turns).
#   train_sft.sh <arm> <seed> <rehearsal_n>   (rehearsal_n=0 => specialist)
set -euo pipefail
source "$HOME/pennywise-v100-infra/scripts/paper/common.sh"
ARM="$1"; SEED="$2"; RN="$3"
OUT=$SCR/runs/${ARM}__s${SEED}/sft
REH_ARGS=(); [ "$RN" -gt 0 ] && REH_ARGS=(--rehearsal "$REH" --rehearsal-n "$RN")
KL=${KL:-0}                                  # baseline E: KL-to-base during SFT
[ "$KL" != 0 ] && REH_ARGS+=(--kl-base "$KL")
MAXS=(); [ -n "${MAX_STEPS:-}" ] && MAXS=(--max-steps "$MAX_STEPS")   # smoke only
$PY scripts/run_sft_v2.py --model $M7 --method lora --system chat \
    --demos 540 --seed 0 --train-seed "$SEED" --lr 1e-5 --epochs 1 \
    --batch-size 2 --max-len 2048 --out-dir "$OUT" ${REH_ARGS[@]+"${REH_ARGS[@]}"} ${MAXS[@]+"${MAXS[@]}"}
mirror "$OUT"
