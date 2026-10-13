#!/bin/bash
# Job B of a 2-job chain: RL (requeue-safe, resumes latest), then its eval.
set -euo pipefail
D=$HOME/pennywise-v100-infra/scripts/paper
bash $D/train_rl.sh "$1" "$2"
bash $D/eval_stage.sh "$1" "$2" rl || echo "[chain] EVAL-RL FAILED for $1 s$2 (rerun eval_stage.sh)"
