#!/bin/bash
# Serve the Qwen2.5-7B-Instruct "chat face" for the LIVE demo (no adapter).
# The bigger base chats naturally AND emits shopping JSON actions; pair it with
# RemotePolicyV2(system=SYSTEM_PROMPT_CHAT). Launch on a single V100 (~15 GB):
#   sbatch --partition=sharing --gres=gpu:v100:1 --cpus-per-task=4 --mem=40G \
#          --time=01:00:00 scripts/serve_7b.sh
# then from the laptop tunnel the node it prints:  ssh -L 8765:<node>:8765 explorer
set -euo pipefail
export HF_HOME=/scratch/madan.pa/hf_cache HF_HUB_OFFLINE=1 \
       PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd "$HOME/pennywise-v100-infra"
/home/madan.pa/.conda/envs/shoprl/bin/python scripts/serve_policy.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8765 --max-new-tokens 160 --peak-flops 125e12
