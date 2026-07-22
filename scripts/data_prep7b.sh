#!/bin/bash
# H3 data prep (one GPU job): self-distil rehearsal set from base 7B + measure
# the base-model general-chat retention BASELINE.
set -euo pipefail
export HF_HOME=/scratch/madan.pa/hf_cache HF_HUB_OFFLINE=1 \
       PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd "$HOME/pennywise-v100-infra"
PY=/home/madan.pa/.conda/envs/shoprl/bin/python
$PY scripts/make_rehearsal.py --n 180 --out /scratch/madan.pa/pennypilot/rehearsal.jsonl
$PY scripts/eval_retention.py --label base
