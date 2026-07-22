#!/bin/bash
# Generic env wrapper for 7B training/eval jobs (H3 experiment). Runs any python
# command in the shoprl conda env with the right HF cache + no user-site shadow.
#   sbatch ... scripts/train7b.sh scripts/run_sft_v2.py --model Qwen/... ...
set -euo pipefail
export HF_HOME=/scratch/madan.pa/hf_cache HF_HUB_OFFLINE=1 \
       PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd "$HOME/pennywise-v100-infra"
exec /home/madan.pa/.conda/envs/shoprl/bin/python "$@"
