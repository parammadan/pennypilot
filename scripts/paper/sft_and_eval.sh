#!/bin/bash
# Job A of a 2-job chain: SFT, then its eval. An eval failure is logged but
# does NOT fail the job (RL in job B must still run; eval can be redone).
set -euo pipefail
D=$HOME/pennywise-v100-infra/scripts/paper
bash $D/train_sft.sh "$1" "$2" "$3"
bash $D/eval_stage.sh "$1" "$2" sft || echo "[chain] EVAL-SFT FAILED for $1 s$2 (rerun eval_stage.sh)"
