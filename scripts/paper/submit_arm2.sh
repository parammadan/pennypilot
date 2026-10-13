#!/bin/bash
# 2-job chain (fits the gpu QOS: 8 submitted / 4 running per user).
#   submit_arm2.sh <arm> <seed> <rehearsal_n>
#   PART_A / PART_B pick partitions (same V100-SXM2-32GB hardware everywhere):
#     PART_A=gpu-short (own QOS, 2h cap: A ~1.1h) PART_B=short (d1027, own QOS)
set -euo pipefail
ARM="$1"; SEED="$2"; RN="$3"
D=$HOME/pennywise-v100-infra/scripts/paper
PART_A=${PART_A:-gpu}; PART_B=${PART_B:-gpu}
S="sbatch --parsable --gres=gpu:v100-sxm2:1 --cpus-per-task=8 --mem=64G --output=/scratch/madan.pa/paper/logs/%x_%j.log"
TA=03:00:00; [ "$PART_A" = gpu-short ] && TA=02:00:00
a=$($S --partition=$PART_A --time=$TA --job-name=A-$ARM-s$SEED $D/sft_and_eval.sh $ARM $SEED $RN)
b=$($S --partition=$PART_B --time=07:00:00 --requeue --dependency=afterok:$a --job-name=B-$ARM-s$SEED $D/rl_and_eval.sh $ARM $SEED) \
  || { scancel $a; echo "submit B failed; cancelled A=$a"; exit 1; }
echo "$(date +%F_%T) $ARM s$SEED rn=$RN: A=$a($PART_A) B=$b($PART_B)" | tee -a /scratch/madan.pa/paper/chains.txt
