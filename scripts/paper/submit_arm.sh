#!/bin/bash
# Chain one arm x seed: SFT -> eval(sft) ; SFT -> RL -> eval(rl).  afterok only.
#   submit_arm.sh <arm> <seed> <rehearsal_n> [dependency-for-SFT, e.g. afterany:123]
set -euo pipefail
ARM="$1"; SEED="$2"; RN="$3"; DEP="${4:-}"
D=$HOME/pennywise-v100-infra/scripts/paper
S="sbatch --parsable --gres=gpu:v100-sxm2:1 --cpus-per-task=8 --mem=64G --output=/scratch/madan.pa/paper/logs/%x_%j.log"
mkdir -p /scratch/madan.pa/paper/logs
j1=$($S --partition=gpu --time=02:00:00 ${DEP:+--dependency=$DEP} --job-name=sft-$ARM-s$SEED $D/train_sft.sh $ARM $SEED $RN)
j2=$($S --partition=gpu --time=04:00:00 --dependency=afterok:$j1 --job-name=ev-sft-$ARM-s$SEED $D/eval_stage.sh $ARM $SEED sft)
j3=$($S --partition=gpu --time=06:00:00 --requeue --dependency=afterok:$j1 --job-name=rl-$ARM-s$SEED $D/train_rl.sh $ARM $SEED)
j4=$($S --partition=gpu --time=04:00:00 --dependency=afterok:$j3 --job-name=ev-rl-$ARM-s$SEED $D/eval_stage.sh $ARM $SEED rl)
echo "$j4" > /scratch/madan.pa/paper/last_chain_tail
echo "$ARM s$SEED rn=$RN: sft=$j1 ev-sft=$j2 rl=$j3 ev-rl=$j4" | tee -a /scratch/madan.pa/paper/chains.txt
