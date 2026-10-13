#!/bin/bash
# 2-job chain (fits the gpu QOS: 8 submitted / 4 running per user).
#   submit_arm2.sh <arm> <seed> <rehearsal_n>
set -euo pipefail
ARM="$1"; SEED="$2"; RN="$3"
D=$HOME/pennywise-v100-infra/scripts/paper
S="sbatch --parsable --partition=gpu --gres=gpu:v100-sxm2:1 --cpus-per-task=8 --mem=64G --output=/scratch/madan.pa/paper/logs/%x_%j.log"
a=$($S --time=03:00:00 --job-name=A-$ARM-s$SEED $D/sft_and_eval.sh $ARM $SEED $RN)
b=$($S --time=07:00:00 --requeue --dependency=afterok:$a --job-name=B-$ARM-s$SEED $D/rl_and_eval.sh $ARM $SEED) \
  || { scancel $a; echo "submit B failed; cancelled A=$a"; exit 1; }
echo "$(date +%F_%T) $ARM s$SEED rn=$RN: A=$a B=$b" | tee -a /scratch/madan.pa/paper/chains.txt
