#!/bin/bash
# General-ability retention (paper metric, complements eval_retention.py):
#   MMLU   — loglikelihood, format-independent => is KNOWLEDGE retained?
#   IFEval — generative, chat template + the deployment system prompt
#            => does the model still BEHAVE as a general assistant?
#   sbatch [slurm flags] scripts/paper/eval_general.sh <model> <label> <adapter|base> [mmlu_limit] [ifeval_limit]
# lm-eval runs in its own venv (vllm-free, transformers 4.49) — never the shoprl env.
set -euo pipefail
export HF_HOME=/scratch/madan.pa/.hf HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} \
       HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1} \
       PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd "$HOME/pennywise-v100-infra"
V=/scratch/madan.pa/venvs/lmeval
MODEL="$1"; LABEL="$2"; ADAPTER="$3"
MMLU_LIMIT="${4:-10}"      # per subject (57 subjects => ~570 items)
IFEVAL_LIMIT="${5:-200}"
MARGS="pretrained=$MODEL,dtype=float16"
if [ "$ADAPTER" != "base" ]; then
  if [ ! -f "$ADAPTER/adapter_config.json" ] && [ -f "$ADAPTER/policy/adapter_config.json" ]; then
    ADAPTER="$ADAPTER/policy"
  fi
  MARGS="$MARGS,peft=$ADAPTER"
fi
OUT=/scratch/madan.pa/paper/eval/$LABEL
mkdir -p "$OUT"
SYS=$(/home/madan.pa/.conda/envs/shoprl/bin/python -c \
  "from shoprl.data.prompts_v2 import SYSTEM_PROMPT_CHAT_MIN as s; print(s)")
echo "[paper-general] $LABEL model=$MODEL adapter=$ADAPTER job=${SLURM_JOB_ID:-none}"
$V/bin/lm_eval --model hf --model_args "$MARGS" --tasks mmlu \
    --num_fewshot 0 --limit "$MMLU_LIMIT" --batch_size 8 --seed 0 \
    --output_path "$OUT/mmlu"
$V/bin/lm_eval --model hf --model_args "$MARGS" --tasks ifeval \
    --apply_chat_template --system_instruction "$SYS" \
    --gen_kwargs max_gen_toks=512 --limit "$IFEVAL_LIMIT" --batch_size 8 \
    --seed 0 --output_path "$OUT/ifeval"
