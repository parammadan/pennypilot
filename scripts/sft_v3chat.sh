#!/bin/bash
# v3 chat+actions SFT retrain -> quarantined sft_v3_chat (frozen ckpts untouched).
set -uo pipefail
cd "$HOME/pennywise-v100-infra"
source env.sh
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
BASE=/scratch/madan.pa/pennypilot
python scripts/run_sft_v2.py --demos 64 --max-steps 20 --batch-size 2 \
  --out-dir "$BASE/sft_v3chat_tiny" || exit 1
python scripts/run_sft_v2.py --demos 1000 --epochs 1 --batch-size 2 \
  --save-every 200 --out-dir "$BASE/sft_v3_chat"
echo "=== sft_v3_chat COMPLETE ==="
