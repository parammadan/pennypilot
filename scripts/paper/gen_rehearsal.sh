#!/bin/bash
# Self-distil the paper rehearsal set: base 7B answers the 900 Dolly prompts.
set -euo pipefail
source "$HOME/pennywise-v100-infra/scripts/paper/common.sh"
$PY scripts/make_rehearsal.py --model $M7 --n 900 \
    --prompts-file src/shoprl/data/rehearsal_prompts_dolly900.jsonl \
    --batch-size 16 --max-new-tokens 256 --out "$REH"
mkdir -p "$KEEP" && cp "$REH" "$KEEP/"
N=$(wc -l < "$REH"); echo "[gen-rehearsal] $N exemplars"
[ "$N" -ge 540 ] || { echo "need >=540 for the 50% dose arm"; exit 1; }
