#!/bin/bash
# Serve BOTH policies on ONE V100 for the live contrast demo (~18GB total):
#   port 8765 — trained 1.5B RL (rloo50_v2): shops expertly, FORGOT how to
#               chat (CHALLENGES #27, live). Drive with:
#                 python scripts/demo_human.py
#   port 8766 — 7B + rehearsal RL adapter (H3 Arm B line): chats AND shops.
#               Drive with:
#                 python scripts/demo_human.py --chat --chat-min \
#                     --policy-url http://localhost:8766
# Launch:
#   sbatch --partition=sharing --gres=gpu:v100:1 --cpus-per-task=4 --mem=48G \
#          --time=01:00:00 scripts/serve_duo.sh [7b-adapter-dir]
# then tunnel BOTH ports:  ssh -L 8765:<node>:8765 -L 8766:<node>:8766 explorer
set -euo pipefail
export HF_HOME=/scratch/madan.pa/hf_cache HF_HUB_OFFLINE=1 \
       PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd "$HOME/pennywise-v100-infra"
PY=/home/madan.pa/.conda/envs/shoprl/bin/python
D=/scratch/madan.pa/pennypilot
ADAPTER_7B="${1:-$D/rl7b_B2/policy}"
if [ ! -f "$ADAPTER_7B/adapter_config.json" ] && [ -f "$ADAPTER_7B/policy/adapter_config.json" ]; then
  ADAPTER_7B="$ADAPTER_7B/policy"
fi
$PY scripts/serve_policy.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --ckpt $D/rloo50_v2/policy --port 8765 --max-new-tokens 128 \
    --peak-flops 125e12 &
exec $PY scripts/serve_policy.py --model Qwen/Qwen2.5-7B-Instruct \
    --ckpt "$ADAPTER_7B" --port 8766 --max-new-tokens 256 --peak-flops 125e12
