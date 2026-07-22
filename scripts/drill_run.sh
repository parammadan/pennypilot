#!/bin/bash
# SIGTERM/resume drill runner (item 9): 20-step throwaway RLOO from frozen
# SFT v2.1; --resume latest makes the same script serve both legs.
set -uo pipefail
cd "$HOME/pennywise-v100-infra"
source env.sh
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
python scripts/run_rl_v2.py \
  --sft-adapter /scratch/madan.pa/pennypilot/sft_v21/policy \
  --algo rloo --steps 20 --k 8 --save-every 4 \
  --out-dir /scratch/madan.pa/pennypilot/drill_rloo --resume latest
